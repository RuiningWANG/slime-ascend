"""Frozen training indexer using the same public NPU operators as SGLang.

Each packed document gets a temporary paged pool cache. Keys are already
projected and normalized, including the inference BF16 rounding boundary.
Only the final token indices are consumed by the differentiable attention.
"""

import torch

from .packed_sequence import packed_boundaries


@torch.no_grad()
def pooled_indexer_topk(q, k, weights, gates, ape, cu_seqlens, topk, pool_size, *, query_start=0):
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or q.shape[-1] != 128:
        raise ValueError("SGLang KPool requires BF16 Q/K and head dimension 128")
    if pool_size <= 0 or topk <= 0 or topk % pool_size:
        raise ValueError("KPool topk must be positive and divisible by pool_size")
    boundaries = packed_boundaries(cu_seqlens, k.shape[0])
    if query_start < 0 or query_start + q.shape[0] > k.shape[0]:
        raise ValueError("KPool query interval lies outside the packed stream")

    from sgl_kernel_npu.operators.indexer import (
        pack_pool_key_indexer_indices,
        pool_key_indexer,
        prepare_pool_key_indexer_inputs,
    )
    from sgl_kernel_npu.operators.mem_cache import (
        kpool_assemble_softmax_rotate_write_cache,
    )

    width = ((topk + pool_size - 1 + 63) // 64) * 64
    output = torch.full((q.shape[0], 1, width), -1, dtype=torch.int32, device=q.device)
    slots_per_page = 128
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        first = max(start - query_start, 0)
        last = min(end - query_start, q.shape[0])
        if last <= first:
            continue
        prefix = query_start + last - start
        positions = torch.arange(query_start + first - start, prefix, device=q.device, dtype=torch.int64)
        pools = prefix // pool_size
        if pools == 0:
            tail = torch.arange(pool_size - 1, device=q.device)
            output[first:last, 0, topk : topk + pool_size - 1] = torch.where(
                tail[None] <= positions[:, None], tail[None] + start, -1
            ).int()
            continue

        # Page zero is reserved, matching inference's nonzero pool page table.
        pages = (pools + slots_per_page - 1) // slots_per_page
        cache = k.new_zeros((pages + 1, slots_per_page, 1, k.shape[-1]))
        tail_k = k.new_zeros((1, pool_size, k.shape[-1]))
        tail_score = gates.new_zeros(tail_k.shape)
        zeros = torch.zeros(pools, device=q.device, dtype=torch.int64)
        pool_ids = torch.arange(pools, device=q.device, dtype=torch.int64)
        kpool_assemble_softmax_rotate_write_cache(
            index_k_cache=cache,
            chunk_k=k[start : start + prefix].contiguous(),
            chunk_score=gates[start : start + prefix].contiguous(),
            tail_k=tail_k,
            tail_score=tail_score,
            req_pool_idx=zeros,
            n_from_tail=zeros.to(torch.int32),
            chunk_src_start=pool_ids * pool_size,
            tail_logical_base=zeros,
            ape=ape.float().contiguous(),
            loc=slots_per_page + pool_ids,
        )
        query_ends = torch.tensor([last - first], device=q.device, dtype=torch.int64)
        pool_lengths = torch.tensor([pools], device=q.device, dtype=torch.int64)
        pool_tails = torch.tensor([prefix % pool_size], device=q.device, dtype=torch.int64)
        table = torch.arange(1, pages + 1, device=q.device, dtype=torch.int32)[None]
        qs, ws, es, ls, ts, bs = prepare_pool_key_indexer_inputs(
            q[first:last].contiguous(),
            weights[first:last].contiguous(),
            query_ends,
            pool_lengths,
            pool_tails,
            table,
        )
        selected, _ = pool_key_indexer(
            query=qs,
            pool_key=cache,
            weights=ws,
            pool_tail_k=ts,
            actual_seq_q=es,
            actual_seq_k=ls,
            block_table=bs,
            layout_q="TND",
            layout_k="PA_BBND",
            topk=topk,
            pool_size=pool_size,
            mask_mode=3,
            quant_mode=-1,
            return_value=False,
        )
        selected = pack_pool_key_indexer_indices(
            indices=selected[: last - first],
            positions=positions,
            query_ends=query_ends,
            topk=topk,
            pool_size=pool_size,
        )
        output[first:last, 0, : selected.shape[-1]] = torch.where(selected >= 0, selected + start, -1).int()
    return output
