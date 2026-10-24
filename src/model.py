"""Two-tower retrieval model: a user tower and an item tower mapping into one L2-normalised space.

Scores are cosine similarities divided by a temperature: `user @ item.T / tau`.
The user tower's purchase history is mean-pooled over the item tower's ID embedding table, so a bought
item and a recommendable item share the same vector.
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
    use_history: bool = True
    use_text: bool = True

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
        # padding_idx: PAD_ID rows stay zero and get no gradient (used by history padding in the user tower).
        self.id_emb = nn.Embedding(cfg.vocab_sizes["item_vocab"], cfg.id_dim, padding_idx=PAD_ID)
        self.attr_emb = nn.ModuleDict({k: nn.Embedding(cfg.vocab_sizes[k], cfg.cat_dim) for k in ITEM_ATTRIBUTE_KEYS})
        in_dim = cfg.id_dim + cfg.cat_dim * len(ITEM_ATTRIBUTE_KEYS)
        if cfg.use_text:
            self.text_proj = nn.Linear(cfg.text_dim, cfg.id_dim)
            in_dim += cfg.id_dim
        self.mlp = mlp(in_dim, cfg)

    def forward(self, item: dict) -> torch.Tensor:
        parts = [self.id_emb(item["item_vocab"])]
        parts += [self.attr_emb[k](item[k]) for k in ITEM_ATTRIBUTE_KEYS]
        if self.cfg.use_text:
            parts.append(self.text_proj(item["text_emb"]))
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class UserTower(nn.Module):
    def __init__(self, cfg: ModelConfig, item_id_emb: nn.Embedding):
        super().__init__()
        self.cfg = cfg
        self.item_id_emb = item_id_emb  # shared with ItemTower, not a copy
        self.attr_emb = nn.ModuleDict({k: nn.Embedding(cfg.vocab_sizes[k], cfg.cat_dim) for k in USER_ATTRIBUTE_KEYS})
        in_dim = cfg.cat_dim * len(USER_ATTRIBUTE_KEYS) + N_NUMERIC_USER_FEATURES
        if cfg.use_user_id:
            self.id_emb = nn.Embedding(cfg.vocab_sizes["user_vocab"], cfg.id_dim)
            in_dim += cfg.id_dim
        if cfg.use_history:
            in_dim += cfg.id_dim
        self.mlp = mlp(in_dim, cfg)

    def pooled_history(self, hist_ids: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        """Mean of the history items' ID embeddings; all-zero for an empty history."""
        emb = self.item_id_emb(hist_ids)  # [B, L, D]; padding rows are zero
        count = hist_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (emb * hist_mask.unsqueeze(-1)).sum(dim=1) / count

    def forward(self, user: dict) -> torch.Tensor:
        parts = [self.attr_emb[k](user[k]) for k in USER_ATTRIBUTE_KEYS]
        parts.append(user["numeric"])
        if self.cfg.use_user_id:
            parts.append(self.id_emb(user["user_vocab"]))
        if self.cfg.use_history:
            parts.append(self.pooled_history(user["hist_ids"], user["hist_mask"]))
        return F.normalize(self.mlp(torch.cat(parts, dim=-1)), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.item_tower = ItemTower(cfg)
        self.user_tower = UserTower(cfg, self.item_tower.id_emb)
        log_tau = torch.tensor(cfg.temperature).log()
        if cfg.learn_temperature:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp()

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (user vectors [B, D], item vectors [B, D]), both unit length."""
        return self.user_tower(batch["user"]), self.item_tower(batch["item"])

    def score(self, user_vecs: torch.Tensor, item_vecs: torch.Tensor) -> torch.Tensor:
        """[B_users, B_items] logits: cosine similarity / temperature."""
        return user_vecs @ item_vecs.T / self.temperature


def parameter_counts(model: TwoTowerModel) -> dict:
    """Parameters per component; the shared item ID table is counted once, under the item tower."""
    def count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    user = model.user_tower
    counts = {
        "item_id_embedding": count(model.item_tower.id_emb),
        "item_other": count(model.item_tower) - count(model.item_tower.id_emb),
        "user_id_embedding": count(user.id_emb) if model.cfg.use_user_id else 0,
    }
    counts["user_other"] = count(user) - counts["user_id_embedding"] - count(user.item_id_emb)
    counts["total"] = sum(p.numel() for p in model.parameters())
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total") + (
        1 if model.cfg.learn_temperature else 0
    )
    return counts
