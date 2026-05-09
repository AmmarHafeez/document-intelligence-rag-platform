from __future__ import annotations

from collections.abc import Iterable, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than 0.")


def _top_k(ids: Sequence[str], k: int) -> list[str]:
    _validate_k(k)
    return list(ids[:k])


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved_relevant = set(_top_k(retrieved_ids, k)) & relevant
    return len(retrieved_relevant) / len(relevant)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    top_ids = _top_k(retrieved_ids, k)
    if not top_ids:
        return 0.0
    return len(set(top_ids) & relevant) / k


def mean_reciprocal_rank(
    ranked_id_lists: Sequence[Sequence[str]],
    relevant_id_sets: Sequence[Iterable[str]],
    *,
    k: int | None = None,
) -> float:
    if len(ranked_id_lists) != len(relevant_id_sets):
        raise ValueError("ranked_id_lists and relevant_id_sets must have the same length.")
    if not ranked_id_lists:
        return 0.0
    if k is not None:
        _validate_k(k)

    reciprocal_ranks: list[float] = []
    for ranked_ids, relevant_ids in zip(ranked_id_lists, relevant_id_sets, strict=True):
        relevant = set(relevant_ids)
        search_space = ranked_ids[:k] if k is not None else ranked_ids
        reciprocal_rank = 0.0
        for index, item_id in enumerate(search_space, start=1):
            if item_id in relevant:
                reciprocal_rank = 1.0 / index
                break
        reciprocal_ranks.append(reciprocal_rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
