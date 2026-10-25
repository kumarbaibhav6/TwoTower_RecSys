"""Two-tower retrieval model: a user tower and an item tower mapping into one L2-normalised space.

Scores are cosine similarities divided by a temperature: `user @ item.T / tau`.

The user tower summarises purchase history in one of two ways:
- history_mode="id": mean of the history items' ID embeddings, shared with the item tower's ID table.
- history_mode="tower": mean of the item tower's *outputs* for the history items (content-aware; required when
  the item ID embedding is switched off). Needs the item feature table attached with `attach_item_table`.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.dataset import ITEM_CATEGORICAL_KEYS
from src.features import PAD_ID

USER_ATTRIBUTE_KEYS = ["age_bucket", "club_status", "news_frequency"]
ITEM_ATTRIBUTE_KEYS = [k for k in ITEM_CATEGORICAL_KEYS if k != "item_vocab"]
N_NUMERIC_USER_FEATURES = 3  # log1p(prior purchases), log1p(days since last), has_prior


@dataclass
class ModelConfig:
    vocab_sizes: dict
    out_dim: int = 64  # shared embedding space
    id_dim: int = 64  # user / item ID embeddings
    cat_dim: int = 16  # small categoricals (colour, age bucket, ...)
    text_dim: int = 384  # all-MiniLM-L6-v2
    hidden_dim: int = 256
    dropout: float = 0.1
    temperature: float = 0.05
    learn_temperature: bool = False
    # Ablation switches
    use_user_id: bool = True
    use_item_id: bool = True
    use_history: bool = True
    use_text: bool = True
    history_mode: str = "id"  # "id" or "tower"; forced to "tower" when use_item_id is False

    def __post_init__(self):
        if not self.use_item_id:
            self.history_mode = "tower"
        if self.history_mode not in ("id", "tower"):
            raise ValueError(f"unknown history_mode {self.history_mode!r}")

    @classmethod
    def from_meta(cls, meta: dict, **overrides) -> "ModelConfig":
        return cls(vocab_sizes=meta["vocab_sizes"], **overrides)

    def to_dict(self) -> dict:
        return asdict(self)


def mlp(in_dim: int, cfg: ModelConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, cfg.hidden_dim),
        nn.GELU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden_dim, cfg.out_dim),
    )


class ItemTower(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.cat_dim * len(ITEM_ATTRIBUTE_KEYS)
        if cfg.use_item_id:
            # padding_idx: PAD_ID rows stay zero and get no gradient (history padding in the user tower).
            self.id_emb = nn.Embedding(cfg.vocab_sizes["item_vocab"], cfg.id_dim, padding_idx=PAD_ID)
            in_dim += cfg.id_dim
        self.attr_emb = nn.ModuleDict({k: nn.Embedding(cfg.vocab_sizes[k], cfg.cat_dim) for k in ITEM_ATTRIBUTE_KEYS})
        if cfg.use_text:
            self.text_proj = nn.Linear(cfg.text_dim, cfg.id_dim)
            in_dim += cfg.id_dim
        self.mlp = mlp(in_dim, cfg)
        self.table_keys: list[str] = []

    def attach_item_table(self, table: dict[str, torch.Tensor]) -> None:
        """Keep every item's input features on the model's device so items can be encoded from row numbers."""
        self.table_keys = [k for k in table if k != "text_emb" or self.cfg.use_text]
        for key in self.table_keys:
            self.register_buffer(f"table_{key}", table[key], persistent=False)  # not saved in checkpoints

    def forward(self, item: dict) -> torch.Tensor:
        # Input order (ID, attributes, text) is part of the checkpoint format: the MLP weights depend on it.
        parts = [self.id_emb(item["item_vocab"])] if self.cfg.use_item_id else []
        parts += [self.attr_emb[k](item[k]) for k in ITEM_ATTRIBUTE_KEYS]
        if self.cfg.use_text:
            parts.append(self.text_proj(item["text_emb"].float()))
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)

    def forward_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Encode items by row number using the attached feature table."""
        if not self.table_keys:
            raise RuntimeError("attach_item_table() must be called before encoding items by row")
        return self({key: getattr(self, f"table_{key}")[rows] for key in self.table_keys})


class UserTower(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.attr_emb = nn.ModuleDict({k: nn.Embedding(cfg.vocab_sizes[k], cfg.cat_dim) for k in USER_ATTRIBUTE_KEYS})
        in_dim = cfg.cat_dim * len(USER_ATTRIBUTE_KEYS) + N_NUMERIC_USER_FEATURES
        if cfg.use_user_id:
            self.id_emb = nn.Embedding(cfg.vocab_sizes["user_vocab"], cfg.id_dim)
            in_dim += cfg.id_dim
        if cfg.use_history:
            in_dim += cfg.id_dim if cfg.history_mode == "id" else cfg.out_dim
        self.mlp = mlp(in_dim, cfg)

    @staticmethod
    def masked_mean(vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over the history axis of [B, L, D] vectors; all-zero for an empty history."""
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (vectors * mask.unsqueeze(-1)).sum(dim=1) / count

    def pooled_history(self, user: dict, item_tower: ItemTower) -> torch.Tensor:
        mask = user["hist_mask"]
        if self.cfg.history_mode == "id":
            return self.masked_mean(item_tower.id_emb(user["hist_ids"]), mask)
        # Encode each distinct history item once, then scatter back to the [B, L] slots.
        rows = torch.where(mask, user["hist_rows"], torch.zeros_like(user["hist_rows"]))
        unique, inverse = torch.unique(rows, return_inverse=True)
        vectors = item_tower.forward_rows(unique)[inverse]
        return self.masked_mean(vectors, mask)

    def forward(self, user: dict, item_tower: ItemTower) -> torch.Tensor:
        parts = [self.attr_emb[k](user[k]) for k in USER_ATTRIBUTE_KEYS]
        parts.append(user["numeric"])
        if self.cfg.use_user_id:
            parts.append(self.id_emb(user["user_vocab"]))
        if self.cfg.use_history:
            parts.append(self.pooled_history(user, item_tower))
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.item_tower = ItemTower(cfg)
        self.user_tower = UserTower(cfg)
        log_tau = torch.tensor(cfg.temperature).log()
        if cfg.learn_temperature:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)

    @property
    def needs_item_table(self) -> bool:
        return self.cfg.use_history and self.cfg.history_mode == "tower"

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp()

    def encode_users(self, user: dict) -> torch.Tensor:
        return self.user_tower(user, self.item_tower)

    def encode_items(self, item: dict) -> torch.Tensor:
        return self.item_tower(item)

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (user vectors [B, D], item vectors [B, D]), both unit length."""
        return self.encode_users(batch["user"]), self.encode_items(batch["item"])

    def score(self, user_vecs: torch.Tensor, item_vecs: torch.Tensor) -> torch.Tensor:
        """[B_users, B_items] logits: cosine similarity / temperature."""
        return user_vecs @ item_vecs.T / self.temperature


def build_model(cfg: ModelConfig, store, device: torch.device, state_dict: dict | None = None) -> TwoTowerModel:
    """Create (and optionally load) a model on `device`, attaching the item table when the config needs it."""
    model = TwoTowerModel(cfg)
    if state_dict is not None:
        # Checkpoints from before the history refactor also stored the shared item ID table under the user tower.
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("user_tower.item_id_emb.")}
        model.load_state_dict(state_dict)
    if model.needs_item_table:
        model.item_tower.attach_item_table(store.item_table())
    return model.to(device)


def parameter_counts(model: TwoTowerModel) -> dict:
    """Parameters per component."""
    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    item, user = model.item_tower, model.user_tower
    counts = {
        "item_id_embedding": count(item.id_emb) if model.cfg.use_item_id else 0,
        "user_id_embedding": count(user.id_emb) if model.cfg.use_user_id else 0,
    }
    counts["item_other"] = count(item) - counts["item_id_embedding"]
    counts["user_other"] = count(user) - counts["user_id_embedding"]
    counts["total"] = sum(p.numel() for p in model.parameters())
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total") + (
        1 if model.cfg.learn_temperature else 0
    )
    return counts
