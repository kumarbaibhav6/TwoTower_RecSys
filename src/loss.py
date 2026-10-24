"""In-batch softmax (InfoNCE) loss with logQ correction and accidental-hit masking."""

import torch
import torch.nn.functional as F


def in_batch_softmax_loss(
    user_vecs: torch.Tensor,
    item_vecs: torch.Tensor,
    item_rows: torch.Tensor,
    item_log_q: torch.Tensor,
    temperature: torch.Tensor | float,
    use_logq: bool = True,
    mask_accidental_hits: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Each user's positive is the item on the same row; every other item in the batch is a negative.

    item_rows identify articles (not vocab ids), so two rows holding the same article are recognised
    as the same item. `item_vecs`, `item_rows` and `item_log_q` may cover more items than there are users
    (e.g. gathered from all GPUs); user b's positive is then column `offset + b` with offset given by the
    caller arranging items so that the local batch comes first.
    """
    n_users = user_vecs.shape[0]
    logits = (user_vecs @ item_vecs.T).float() / temperature  # [B_users, B_items]
    if use_logq:
        # Yi et al. (2019): subtract log sampling probability per item column, positives included.
        logits = logits - item_log_q.float()[None, :]
    labels = torch.arange(n_users, device=logits.device)
    if mask_accidental_hits:
        same_item = item_rows[:n_users, None] == item_rows[None, :]
        same_item[labels, labels] = False  # keep the true positive
        logits = logits.masked_fill(same_item, float("-inf"))
    loss = F.cross_entropy(logits, labels)
    with torch.no_grad():
        stats = {
            "in_batch_accuracy": (logits.argmax(dim=1) == labels).float().mean().item(),
            "accidental_hits_per_row": same_item.sum().item() / n_users if mask_accidental_hits else 0.0,
        }
    return loss, stats
