import pytest
import torch

from src.features import HIST_LEN, PAD_ID
from src.model import ITEM_ATTRIBUTE_KEYS, USER_ATTRIBUTE_KEYS, ModelConfig, TwoTowerModel, parameter_counts

VOCAB = {
    "user_vocab": 50, "item_vocab": 40, "age_bucket": 12, "club_status": 4, "news_frequency": 4,
    "product_group_name": 20, "colour_group_name": 51, "product_type_name": 132, "index_group_name": 6,
}
B = 8


def random_batch(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    hist_len = torch.randint(0, HIST_LEN + 1, (B,), generator=g)
    hist_mask = torch.arange(HIST_LEN)[None, :] >= (HIST_LEN - hist_len[:, None])  # left-padded like the dataset
    hist_ids = torch.where(hist_mask, torch.randint(1, VOCAB["item_vocab"], (B, HIST_LEN), generator=g), PAD_ID)
    user = {k: torch.randint(0, VOCAB[k], (B,), generator=g) for k in ["user_vocab", *USER_ATTRIBUTE_KEYS]}
    user.update(hist_ids=hist_ids, hist_mask=hist_mask, numeric=torch.rand(B, 3, generator=g))
    item = {k: torch.randint(0, VOCAB[k], (B,), generator=g) for k in ["item_vocab", *ITEM_ATTRIBUTE_KEYS]}
    item["text_emb"] = torch.randn(B, 384, generator=g)
    return {"user": user, "item": item}


def make_model(**overrides) -> TwoTowerModel:
    torch.manual_seed(0)
    return TwoTowerModel(ModelConfig(vocab_sizes=VOCAB, **overrides)).eval()


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
    assert model.user_tower.item_id_emb is model.item_tower.id_emb
    names = [n for n, _ in model.named_parameters()]
    assert sum("id_emb" in n and "item" in n for n in names) == 1  # registered once


def test_pooled_history_is_masked_mean_and_ignores_padding():
    model = make_model()
    batch = random_batch()
    ids, mask = batch["user"]["hist_ids"], batch["user"]["hist_mask"]
    pooled = model.user_tower.pooled_history(ids, mask)
    emb = model.item_tower.id_emb.weight
    for b in range(B):
        expected = emb[ids[b][mask[b]]].mean(0) if mask[b].any() else torch.zeros(64)
        torch.testing.assert_close(pooled[b], expected)
    # Whatever sits in masked-out slots must not matter.
    noisy = torch.where(mask, ids, torch.randint(1, VOCAB["item_vocab"], ids.shape))
    torch.testing.assert_close(model.user_tower.pooled_history(noisy, mask), pooled)


@pytest.mark.parametrize("switch", ["use_user_id", "use_history", "use_text"])
def test_ablation_variants_run(switch):
    model = make_model(**{switch: False})
    batch = random_batch()
    if switch == "use_text":
        del batch["item"]["text_emb"]  # the variant must not need text
    u, i = model(batch)
    assert u.shape == i.shape == (B, 64)
    if switch == "use_user_id":
        assert parameter_counts(model)["user_id_embedding"] == 0


def test_every_parameter_gets_a_gradient():
    model = make_model(learn_temperature=True).train()
    u, i = model(random_batch())
    loss = torch.nn.functional.cross_entropy(model.score(u, i), torch.arange(B))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert missing == []
    assert model.item_tower.id_emb.weight.grad[PAD_ID].abs().sum() == 0  # padding row stays frozen


def test_parameter_counts_add_up():
    model = make_model()
    counts = parameter_counts(model)
    assert counts["item_id_embedding"] == VOCAB["item_vocab"] * 64
    assert counts["user_id_embedding"] == VOCAB["user_vocab"] * 64
    assert counts["total"] == sum(p.numel() for p in model.parameters())
