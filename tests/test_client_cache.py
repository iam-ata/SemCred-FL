"""Tests for the resumable schedule-bound Phase 6B client cache."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from conftest import Phase4CorpusFixture

from semcred_fl.data.client_cache import (
    ClientCacheBuildRequest,
    build_selected_client_cache,
    iter_cached_client_batches,
    load_selected_client_cache,
)


def test_client_cache_build_resume_and_exact_stream(
    phase4_corpus: Phase4CorpusFixture, tmp_path: Path
) -> None:
    output = tmp_path / "cache"
    request = ClientCacheBuildRequest(
        preprocessing_report=phase4_corpus.preprocessing_report,
        client_partition_manifest=phase4_corpus.client_partition_manifest,
        profile="iid",
        selected_client_indices=(0,),
        schedule_id="a" * 64,
        output_directory=output,
        work_database=output / "cache.sqlite3",
        batch_size=17,
    )
    first = build_selected_client_cache(request)
    second = build_selected_client_cache(request)
    assert first == second
    assert first["records"] == 40
    assert first["client_totals"] == [40]
    assert first["family_client_counts"] == [[20], [20]]
    gates = cast(dict[str, object], first["gates"])
    assert gates["validation_test_and_anchors_excluded"] is True

    cache = load_selected_client_cache(output / "client_cache.json")
    batches = list(iter_cached_client_batches(cache, 0, batch_size=13, shuffle_seed=20260816))
    repeated = list(iter_cached_client_batches(cache, 0, batch_size=13, shuffle_seed=20260816))
    assert sum(len(batch.record_ids) for batch in batches) == 40
    assert len({record_id for batch in batches for record_id in batch.record_ids}) == 40
    assert sum(int((batch.targets == 0).sum()) for batch in batches) == 20
    assert sum(int((batch.targets == 1).sum()) for batch in batches) == 20
    assert [batch.record_ids for batch in batches] == [batch.record_ids for batch in repeated]
    assert all(torch.isfinite(batch.features).all().item() for batch in batches)
    assert all(batch.training_batch().records == len(batch.record_ids) for batch in batches)
