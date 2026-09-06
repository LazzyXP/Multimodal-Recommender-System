"""Shared tensor utilities for the graph recommender family.

These helpers implement the low-level building blocks that the individual
papers use (symmetric graph normalization, sparse message passing, BPR and
InfoNCE losses, k-NN graph construction). They are generic infrastructure, not
model-specific logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

MAX_SIMILARITY_BYTES = 128 * 1024 * 1024


def bpr_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    users: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
) -> torch.Tensor:
    """Bayesian Personalized Ranking loss over (user, positive, negative)."""
    user_vec = user_emb[users]
    positive_vec = item_emb[positives]
    negative_vec = item_emb[negatives]
    return -F.logsigmoid((user_vec * (positive_vec - negative_vec)).sum(dim=1)).mean()


class _SymmetricInfoNCE(torch.autograd.Function):
    """Chunked symmetric InfoNCE with a chunked, memory-bounded backward pass."""

    @staticmethod
    def forward(  # noqa: D102
        ctx: Any,
        first: torch.Tensor,
        second: torch.Tensor,
        temperature: float,
        chunk: int,
    ) -> torch.Tensor:
        count = first.size(0)
        inv_temperature = 1.0 / temperature
        positives = (first * second).sum(dim=1) * inv_temperature  # diagonal logits
        row_lse = torch.empty(count, device=first.device, dtype=first.dtype)
        column_lse: torch.Tensor | None = None
        for start in range(0, count, chunk):
            block = (first[start : start + chunk] @ second.T) * inv_temperature
            row_lse[start : start + chunk] = block.logsumexp(dim=1)
            column_block = block.logsumexp(dim=0)
            column_lse = (
                column_block
                if column_lse is None
                else torch.logaddexp(column_lse, column_block)
            )
        loss = 0.5 * ((row_lse - positives).mean() + (column_lse - positives).mean())
        ctx.save_for_backward(first, second, row_lse, column_lse)
        ctx.inv_temperature = inv_temperature
        ctx.chunk = chunk
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:  # noqa: D102
        first, second, row_lse, column_lse = ctx.saved_tensors
        count = first.size(0)
        chunk = ctx.chunk
        inv_temperature = ctx.inv_temperature
        needs_first = ctx.needs_input_grad[0]
        needs_second = ctx.needs_input_grad[1]
        if not needs_first and not needs_second:
            return None, None, None, None
        grad_first = torch.zeros_like(first) if needs_first else None
        grad_second = torch.zeros_like(second) if needs_second else None
        for start in range(0, count, chunk):
            rows = first[start : start + chunk]  # (c, D)
            block = (rows @ second.T) * inv_temperature  # (c, N)
            # dL/dlogits = 0.5/N * (softmax_row + softmax_col) - 1/N * [i == j]
            softmax_row = torch.exp(block - row_lse[start : start + chunk, None])
            softmax_col = torch.exp(block - column_lse[None, :])
            combined = (0.5 / count) * (softmax_row + softmax_col)
            local = torch.arange(rows.size(0), device=first.device)
            combined[local, start + local] -= 1.0 / count
            if needs_first:
                grad_first[start : start + rows.size(0)] = (
                    (combined @ second) * inv_temperature * grad_output
                )
            if needs_second:
                grad_second += (combined.T @ rows) * inv_temperature
        if needs_second:
            # Fold grad_output in-place once instead of allocating a full copy.
            grad_second.mul_(grad_output)  # type: ignore[union-attr]
        return grad_first, grad_second, None, None


def info_nce(
    first: torch.Tensor,
    second: torch.Tensor,
    temperature: float = 0.5,
    chunk: int = 2048,
) -> torch.Tensor:
    """Symmetric InfoNCE between two views of the same batch of nodes.

    Exact full-batch InfoNCE (every other node is a negative) in the
    numerically-stable log-sum-exp form. The ``N x N`` logits matrix is never
    materialized: a custom autograd Function evaluates it in row chunks and
    recomputes each chunk's softmax during backward, so memory stays bounded at
    ``chunk x N`` in both passes instead of the quadratic dense form.
    """
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    count = first.size(0)
    if count == 0:
        return torch.zeros((), device=first.device)
    # Keep the temporary similarity block bounded even when the caller passes
    # a large default chunk and the node count is large.
    chunk = min(
        chunk,
        max(1, MAX_SIMILARITY_BYTES // max(1, count * first.element_size())),
    )
    if count <= chunk:
        logits = (first @ second.T) / temperature
        labels = torch.arange(count, device=first.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return _SymmetricInfoNCE.apply(first, second, temperature, chunk)


def symmetric_edge_weight(src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return the D^{-1/2} A D^{-1/2} weight for each directed (src -> dst) edge."""
    degree = torch.zeros(num_nodes, device=src.device, dtype=torch.float32)
    degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    return (degree[src].clamp_min(1.0) * degree[dst].clamp_min(1.0)).pow(-0.5)


def propagate(
    embeddings: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_weight: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Sparse message passing: out[dst] += edge_weight * embeddings[src]."""
    messages = edge_weight.unsqueeze(-1) * embeddings[src]
    out = torch.zeros(
        num_nodes,
        embeddings.size(1),
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    out.index_add_(0, dst, messages)
    return out


def knn_graph(
    features: torch.Tensor, k: int, chunk: int = 4096
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an undirected k-NN item-item graph, returning (src, dst) tensors.

    Similarities are computed in row chunks so an N x N matrix is never
    materialized. Self edges are excluded.
    """
    features = F.normalize(features, dim=-1)
    count = features.size(0)
    chunk = min(
        chunk,
        max(1, MAX_SIMILARITY_BYTES // max(1, count * features.element_size())),
    )
    neighbors = min(k + 1, count)
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        block = features[start:stop] @ features.T
        _, indices = torch.topk(block, k=neighbors, dim=1)
        indices = indices[:, 1:]  # drop self
        src = (
            torch.arange(start, stop, device=features.device)[:, None]
            .expand(-1, indices.size(1))
            .reshape(-1)
        )
        dst_parts.append(indices.reshape(-1))
        src_parts.append(src)
    src = torch.cat(src_parts)
    dst = torch.cat(dst_parts)
    keep = src != dst
    return src[keep], dst[keep]


def cooccurrence_topk(
    left: torch.Tensor,
    right: torch.Tensor,
    num_left: int,
    num_right: int,
    k: int,
    degree_cap: int = 500,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a top-``k`` co-occurrence graph on the ``left`` side of a bipartite
    edge list.

    Two left-nodes co-occur when they share a right-node; each shared right-node
    ``r`` contributes ``1 / degree(r)^2`` to their weight (the DRAGON
    user-user / item-item normalization). Contributions from multiple shared
    right-nodes are *summed* before the top-``k`` selection, and self pairs are
    excluded. All ordered pairs are packed into a preallocated ``(key, weight)``
    buffer and aggregated with a single ``np.unique`` + ``bincount`` pass, so
    the dense ``num_left x num_left`` matrix is never materialized. Right-nodes
    above ``degree_cap`` are skipped because their ``1 / degree^2`` weight (at
    most ``1 / degree_cap^2``) is negligible next to rarer shared right-nodes;
    this is the only bounded-memory approximation and only affects catalogs with
    very popular right-nodes.
    """
    left_np = left.detach().cpu().numpy().astype(np.int64)
    right_np = right.detach().cpu().numpy().astype(np.int64)
    order = np.argsort(right_np, kind="stable")
    right_sorted = right_np[order]
    left_sorted = left_np[order]

    counts = np.bincount(right_sorted, minlength=num_right).astype(np.int64)
    inv_deg_sq = np.zeros(num_right, dtype=np.float32)
    np.divide(1.0, counts.astype(np.float32), out=inv_deg_sq, where=counts > 0)
    inv_deg_sq *= inv_deg_sq
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)[:-1]))

    active = np.flatnonzero((counts >= 2) & (counts <= degree_cap))
    sizes = counts[active]
    total = int(np.sum(sizes * (sizes - 1)))
    if total == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    keys = np.empty(total, dtype=np.int64)
    weights = np.empty(total, dtype=np.float32)
    position = 0
    for index, right_node in enumerate(active):
        size = int(sizes[index])
        block = left_sorted[offsets[right_node] : offsets[right_node] + size].astype(np.int64)
        weight = float(inv_deg_sq[right_node])
        packed = (block[:, None] * num_left + block[None, :]).ravel()
        # Mask every position where src == dst *by value* (not just the positional
        # diagonal), so duplicate (left, right) edges cannot smuggle in self-loops.
        keep = ~(block[:, None] == block[None, :]).ravel()
        count = int(keep.sum())
        keys[position : position + count] = packed[keep]
        weights[position : position + count] = weight
        position += count

    if position < total:
        # With duplicate (left, right) edges the value-based self-mask excludes
        # more than the positional diagonal, so fewer entries are written than
        # the d*(d-1) upper bound; trim the buffer instead of feeding
        # uninitialized memory to np.unique.
        keys = keys[:position]
        weights = weights[:position]

    unique_keys, inverse = np.unique(keys, return_inverse=True)
    summed = np.bincount(inverse, weights=weights).astype(np.float32)
    del keys, weights, inverse

    src = (unique_keys // num_left).astype(np.int64)
    dst = (unique_keys % num_left).astype(np.int64)
    order = np.lexsort((-summed, src))
    src = src[order]
    dst = dst[order]
    start = np.searchsorted(src, np.arange(num_left), side="left")
    rank = np.arange(src.size) - start[src]
    keep = rank < k
    return torch.as_tensor(src[keep]), torch.as_tensor(dst[keep])


def scatter_attention(
    modality_embeddings: list[torch.Tensor],
    query: torch.Tensor,
) -> torch.Tensor:
    """Weighted average of per-modality embeddings using a softmax attention.

    ``query`` provides one query vector per node; each modality embedding must
    have the same shape as ``query``.
    """
    stacked = torch.stack(modality_embeddings, dim=0)  # (M, N, D)
    scores = (stacked * query.unsqueeze(0)).sum(dim=-1)  # (M, N)
    weights = torch.softmax(scores / (query.size(-1) ** 0.5), dim=0)  # (M, N)
    return (stacked * weights.unsqueeze(-1)).sum(dim=0)


def build_positive_mask(
    edge_users: torch.Tensor,
    edge_items: torch.Tensor,
    num_users: int,
    num_items: int,
) -> np.ndarray:
    """Build a bit-packed (n_users, ceil(n_items/8)) uint8 positive mask.

    Each user's liked items are stored as set bits, so the collision check in
    ``sample_negatives`` is a vectorized gather + bitwise-and instead of a
    per-sample Python set lookup.
    """
    users = edge_users.detach().cpu().numpy().astype(np.int64)
    items = edge_items.detach().cpu().numpy().astype(np.int64)
    width = (num_items + 7) >> 3
    mask = np.zeros((num_users, width), dtype=np.uint8)
    np.bitwise_or.at(mask, (users, items >> 3), np.uint8(1) << (items & 7))
    return mask


def sample_negatives(
    users: torch.Tensor,
    num_items: int,
    positive_mask: np.ndarray,
    rng: torch.Generator,
) -> torch.Tensor:
    """Sample one negative item per user, rejecting items the user already liked.

    Fully vectorized: the collision check reads the bit-packed positive mask in
    one gather and one bitwise-and, resampling only the (rare) collided entries.
    """
    count = users.numel()
    users_np = users.detach().cpu().numpy().astype(np.int64)
    negatives = torch.randint(0, num_items, (count,), generator=rng)
    for _ in range(10):
        negative_np = negatives.numpy()
        word = negative_np >> 3
        bit = np.uint8(1) << (negative_np & 7)
        collided = (positive_mask[users_np, word] & bit) != 0
        hit = int(collided.sum())
        if hit == 0:
            break
        indices = np.nonzero(collided)[0]
        negatives[torch.as_tensor(indices)] = torch.randint(0, num_items, (hit,), generator=rng)
    # Dense users can still collide after the bounded retry loop. Resolve only
    # those rare entries by a deterministic circular scan, and fail explicitly
    # when a user has no legal negative instead of silently corrupting BPR pairs.
    negative_np = negatives.numpy()
    word = negative_np >> 3
    bit = np.uint8(1) << (negative_np & 7)
    collided = (positive_mask[users_np, word] & bit) != 0
    for index in np.nonzero(collided)[0]:
        start = int(negative_np[index])
        user = int(users_np[index])
        for offset in range(num_items):
            candidate = (start + offset) % num_items
            if not (positive_mask[user, candidate >> 3] & (np.uint8(1) << (candidate & 7))):
                negative_np[index] = candidate
                break
        else:
            raise ValueError(
                "Cannot sample a negative item for a user who has seen the full catalog."
            )
    return negatives


def normalize_features(blocks: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert encoded feature blocks into row-normalized float tensors."""
    return {
        name: torch.nn.functional.normalize(torch.as_tensor(matrix, dtype=torch.float32), dim=1)
        for name, matrix in blocks.items()
    }
