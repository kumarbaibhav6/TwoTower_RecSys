from pathlib import Path

import pytest
import torch

from src.features import HIST_LEN, PAD_ID
from src.model import ITEM_ATTRIBUTE_KEYS, USER_ATTRIBUTE_KEYS, ModelConfig, TwoTowerModel, parameter_counts

VOCAB = {
    "user_vocab": 50, "item_vocab": 40, "age_bucket": 12, "club_status": 4, "news_frequency": 4,
    "product_group_name": 20, "colour_group_name": 51, "product_type_name": 132, "index_group_name": 6,
}
B = 8
N_ITEMS = 30
_g = torch.Generator().manual_seed(1)
ITEM_TABLE = {k: torch.randint(1 if k == "item_vocab" else 0, VOCAB[k], (N_ITEMS,), generator=_g)
              for k in ["item_vocab", *ITEM_ATTRIBUTE_KEYS]}
ITEM_TABLE["text_emb"] = torch.randn(N_ITEMS, 384, generator=_g).half()


def random_batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    hist_len = torch.randint(0, HIST_LEN + 1, (B,), generator=g)
    hist_mask = torch.arange(HIST_LEN)[None, :] >= (HIST_LEN - hist_len[:, None])  # left-padded like the dataset
    hist_rows = torch.where(hist_mask, torch.randint(0, N_ITEMS, (B, HIST_LEN), generator=g), 0)
    hist_ids = torch.where(hist_mask, ITEM_TABLE["item_vocab"][hist_rows], PAD_ID)
    user = {k: torch.randint(0, VOCAB[k], (B,), generator=g) for k in ["user_vocab", *USER_ATTRIBUTE_KEYS]}
    user.update(hist_ids=hist_ids, hist_rows=hist_rows, hist_mask=hist_mask, numeric=torch.rand(B, 3, generator=g))
    item = {k: torch.randint(0, VOCAB[k], (B,), generator=g) for k in ["item_vocab", *ITEM_ATTRIBUTE_KEYS]}
    item["text_emb"] = torch.randn(B, 384, generator=g)
    return {"user": user, "item": item}


def make_model(**overrides) -> TwoTowerModel:
    torch.manual_seed(0)
    model = TwoTowerModel(ModelConfig(vocab_sizes=VOCAB, **overrides)).eval()
    if model.needs_item_table:
        model.item_tower.attach_item_table(ITEM_TABLE)
    return model


def test_forward_shapes_and_unit_norm():
    model = make_model()
    u, i = model(random_batch())
    assert u.shape == i.shape == (B, 64)
    torch.testing.assert_close(u.norm(dim=-1), torch.ones(B))
    torch.testing.assert_close(i.norm(dim=-1), torch.ones(B))
    scores = model.score(u, i)
    assert scores.shape == (B, B)
    assert scores.abs().max() <= 1 / 0.05 + 1e-4  # cosine / tau


def test_history_shares_item_id_embedding():
    model = make_model()
    names = [n for n, _ in model.named_parameters()]
    assert names.count("item_tower.id_emb.weight") == 1
    assert not any("user_tower" in n and "id_emb" in n and "item" in n for n in names)


def test_id_history_is_masked_mean_and_ignores_padding():
    model = make_model()
    batch = random_batch()
    user = batch["user"]
    ids, mask = user["hist_ids"], user["hist_mask"]
    pooled = model.user_tower.pooled_history(user, model.item_tower)
    emb = model.item_tower.id_emb.weight
    for b in range(B):
        expected = emb[ids[b][mask[b]]].mean(0) if mask[b].any() else torch.zeros(64)
        torch.testing.assert_close(pooled[b], expected)
    # Whatever sits in masked-out slots must not matter.
    noisy = dict(user, hist_ids=torch.where(mask, ids, torch.randint(1, VOCAB["item_vocab"], ids.shape)))
    torch.testing.assert_close(model.user_tower.pooled_history(noisy, model.item_tower), pooled)


def test_tower_history_is_mean_of_item_tower_outputs():
    model = make_model(history_mode="tower")
    user = random_batch()["user"]
    pooled = model.user_tower.pooled_history(user, model.item_tower)
    all_items = model.item_tower.forward_rows(torch.arange(N_ITEMS))
    for b in range(B):
        m = user["hist_mask"][b]
        expected = all_items[user["hist_rows"][b][m]].mean(0) if m.any() else torch.zeros(64)
        torch.testing.assert_close(pooled[b], expected, atol=1e-5, rtol=1e-5)


def test_no_item_id_uses_content_only_and_tower_history():
    model = make_model(use_item_id=False)
    assert model.cfg.history_mode == "tower"
    assert parameter_counts(model)["item_id_embedding"] == 0
    batch = random_batch()
    # Changing an item's ID must not change its vector when IDs are off.
    item_a = {k: v.clone() for k, v in batch["item"].items()}
    item_b = dict(item_a, item_vocab=torch.randint(1, VOCAB["item_vocab"], (B,)))
    torch.testing.assert_close(model.encode_items(item_a), model.encode_items(item_b))
    u, i = model(batch)
    assert u.shape == i.shape == (B, 64)


def test_old_checkpoint_reproduces_its_recorded_val_score():
    """A checkpoint saved before the history refactor must load and score exactly as recorded."""
    path, features = Path("models/sample-smoke/best.pt"), Path("data/features/sample_0.05")
    if not (path.exists() and (features / "meta.json").exists()):
        pytest.skip("local sample checkpoint/features not available")
    from src.dataset import FeatureStore
    from src.evaluate import WeekEvaluator
    from src.model import build_model
    from src.split import VAL

    ck = torch.load(path, weights_only=False)
    store = FeatureStore(features)
    model = build_model(ModelConfig(**ck["model_config"]), store, torch.device("cpu"), ck["model"]).eval()
    result = WeekEvaluator(store, VAL).evaluate(model, torch.device("cpu"))
    assert result["map@12"] == pytest.approx(ck["val"]["map@12"], rel=1e-6)
    assert result["recall@100"] == pytest.approx(ck["val"]["recall@100"], rel=1e-6)


@pytest.mark.parametrize("switch", ["use_user_id", "use_item_id", "use_history", "use_text"])
def test_ablation_variants_run(switch):
    model = make_model(**{switch: False})
    batch = random_batch()
    if switch == "use_text":
        del batch["item"]["text_emb"]  # the variant must not need text
    u, i = model(batch)
    assert u.shape == i.shape == (B, 64)
    if switch == "use_user_id":
        assert parameter_counts(model)["user_id_embedding"] == 0


@pytest.mark.parametrize("overrides", [{}, {"use_item_id": False}])
def test_every_parameter_gets_a_gradient(overrides):
    model = make_model(learn_temperature=True, **overrides).train()
    u, i = model(random_batch())
    loss = torch.nn.functional.cross_entropy(model.score(u, i), torch.arange(B))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert missing == []
    if model.cfg.use_item_id:
        assert model.item_tower.id_emb.weight.grad[PAD_ID].abs().sum() == 0  # padding row stays frozen


def test_parameter_counts_add_up():
    model = make_model()
    counts = parameter_counts(model)
    assert counts["item_id_embedding"] == VOCAB["item_vocab"] * 64
    assert counts["user_id_embedding"] == VOCAB["user_vocab"] * 64
    assert counts["total"] == sum(p.numel() for p in model.parameters())
