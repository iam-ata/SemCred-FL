"""Synthetic integration tests for resumable Phase 6C attack execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from conftest import Phase4CorpusFixture

from semcred_fl.attacks.artifacts import (
    Phase6BResolution,
    ResolvedCoordinate,
    TriggerCoordinate,
)
from semcred_fl.attacks.execution import Phase6CAttackRequest, run_phase6c_attack
from semcred_fl.attacks.execution_contracts import (
    AttackRunSpec,
    phase6c_execution_config,
)
from semcred_fl.identity import to_primitive


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upgrade_to_eight_clients(fixture: Phase4CorpusFixture) -> tuple[str, str]:
    corpus_id = "a" * 64
    preprocessor_id = "b" * 64
    split_id = "c" * 64
    anchor_id = "d" * 64
    partition_id = "e" * 64
    report = cast(
        dict[str, object],
        json.loads(fixture.preprocessing_report.read_text(encoding="utf-8")),
    )
    preprocessor_record = cast(dict[str, object], report["preprocessor"])
    preprocessor_path = fixture.preprocessing_report.parent / cast(str, preprocessor_record["path"])
    preprocessor = cast(
        dict[str, object], json.loads(preprocessor_path.read_text(encoding="utf-8"))
    )
    preprocessor.update(
        {
            "preprocessor_id": preprocessor_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
        }
    )
    _write_json(preprocessor_path, preprocessor)
    report.update(
        {
            "preprocessed_corpus_id": corpus_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
        }
    )
    preprocessor_record.update(
        {"preprocessor_id": preprocessor_id, "sha256": _sha256(preprocessor_path)}
    )
    _write_json(fixture.preprocessing_report, report)

    manifest = cast(
        dict[str, object],
        json.loads(fixture.client_partition_manifest.read_text(encoding="utf-8")),
    )
    assignment_files = cast(list[dict[str, object]], manifest["assignment_files"])
    assignment_path = fixture.client_partition_manifest.parent / cast(
        str, assignment_files[0]["path"]
    )
    assignment = pq.read_table(assignment_path)
    client_ids = np.asarray([(index // 2) % 8 for index in range(80)], dtype=np.int16)
    rewritten = assignment.set_column(
        assignment.schema.get_field_index("iid_client_id"),
        "iid_client_id",
        pa.array(client_ids, type=pa.int16()),
    )
    pq.write_table(rewritten, assignment_path)
    assignment_files[0].update(
        {
            "byte_size": assignment_path.stat().st_size,
            "sha256": _sha256(assignment_path),
        }
    )
    manifest.update(
        {
            "client_partition_id": partition_id,
            "preprocessed_corpus_id": corpus_id,
            "preprocessor_id": preprocessor_id,
            "split_id": split_id,
            "anchor_id": anchor_id,
            "num_clients": 8,
            "profiles": {
                "iid": {
                    "client_totals": [10] * 8,
                    "family_client_counts": {
                        "benign": [5] * 8,
                        "attack": [5] * 8,
                    },
                }
            },
        }
    )
    _write_json(fixture.client_partition_manifest, manifest)
    return corpus_id, partition_id


def _resolution(path: Path, corpus_id: str, partition_id: str) -> Path:
    ranked = tuple(
        ResolvedCoordinate(
            coordinate=f"x{index:04d}",
            index=index,
            source_feature=f"feature-{index}",
            lower_bound=-2.0,
            upper_bound=2.0,
            source_median=1.0,
            benign_median=-1.0,
            absolute_median_gap=2.0,
        )
        for index in range(4)
    )
    resolution = Phase6BResolution(
        dataset="edge_iiotset",
        phase6a_contract_id="f" * 64,
        preprocessed_corpus_id=corpus_id,
        preprocessor_id="b" * 64,
        split_id="c" * 64,
        anchor_id="d" * 64,
        client_partition_id=partition_id,
        source_family="attack",
        source_family_id=1,
        target_family="benign",
        target_family_id=0,
        source_training_records=40,
        target_training_records=40,
        federated_training_records=80,
        ranked_coordinates=ranked,
        feature_coordinates=ranked[:3],
        trigger_coordinates=tuple(
            TriggerCoordinate(
                coordinate=item.coordinate,
                index=item.index,
                source_feature=item.source_feature,
                endpoint="lower",
                value=-2.0,
            )
            for item in ranked[:3]
        ),
        trigger_training_collisions=0,
    )
    payload = {
        "resolution_id": resolution.resolution_id,
        **cast(dict[str, object], to_primitive(resolution)),
        "gates": {
            "source_and_benign_medians_exact_over_federated_train": True,
            "feature_coordinates_nonconstant_and_train_ranked": True,
            "trigger_from_top_twelve_coordinates": True,
            "trigger_has_zero_federated_training_collisions": True,
            "scanner_filter_excludes_validation_test_and_anchors": True,
            "test_partition_accessed": False,
            "performance_claims_allowed": False,
        },
    }
    _write_json(path, payload)
    return path


def test_phase6c_resume_noop_equivalence_and_population_selection(
    phase4_corpus: Phase4CorpusFixture, tmp_path: Path
) -> None:
    corpus_id, partition_id = _upgrade_to_eight_clients(phase4_corpus)
    resolution_path = _resolution(tmp_path / "resolution.json", corpus_id, partition_id)
    config = replace(
        phase6c_execution_config(),
        profiles=("iid",),
        malicious_client_ratios=(0.25,),
        poison_fractions=(0.5,),
        total_clients=8,
        clients_per_round=4,
        total_rounds=2,
        local_epochs=1,
        batch_size=5,
        hidden_dimensions=(8,),
        dropout=0.0,
    )
    cache = tmp_path / "cache"

    def request(
        name: str,
        *,
        poison_fraction: float = 0.0,
        maximum_rounds: int | None = None,
    ) -> Phase6CAttackRequest:
        output = tmp_path / name
        return Phase6CAttackRequest(
            preprocessing_report=phase4_corpus.preprocessing_report,
            client_partition_manifest=phase4_corpus.client_partition_manifest,
            resolution_path=resolution_path,
            output_directory=output,
            cache_directory=cache,
            cache_work_database=cache / "cache.sqlite3",
            spec=AttackRunSpec(
                dataset="edge_iiotset",
                profile="iid",
                attack_name="clean" if name == "clean" else "label_flip",
                malicious_ratio=0.25,
                seed=17,
                poison_fraction=poison_fraction,
            ),
            device="cpu",
            cache_scan_batch_size=17,
            max_rounds_this_call=maximum_rounds,
            config=config,
        )

    clean = run_phase6c_attack(request("clean"))
    interrupted = run_phase6c_attack(request("zero-flip", maximum_rounds=1))
    assert interrupted["status"] == "interrupted"
    assert interrupted["next_round"] == 1
    zero_flip = run_phase6c_attack(request("zero-flip"))
    assert run_phase6c_attack(request("zero-flip")) == zero_flip
    assert clean["final_parameters_sha256"] == zero_flip["final_parameters_sha256"]
    assert clean["realization_id"] == zero_flip["realization_id"]

    poisoned = run_phase6c_attack(request("half-flip", poison_fraction=0.5, maximum_rounds=1))
    assert poisoned["status"] == "interrupted"
    selections = cast(
        dict[str, object],
        json.loads(
            (tmp_path / "half-flip" / "poison_selection_manifest.json").read_text(encoding="utf-8")
        ),
    )
    clients = cast(list[dict[str, object]], selections["client_selections"])
    assert len(clients) == 2
    assert {record["eligible_count"] for record in clients} == {5}
    assert {record["selected_count"] for record in clients} == {2}
    audit = cast(
        dict[str, object],
        json.loads(
            (tmp_path / "half-flip" / "attack_rounds" / "round-0000.json").read_text(
                encoding="utf-8"
            )
        ),
    )
    assert len(cast(list[str], audit["active_malicious_client_ids"])) == 1
    updates = cast(list[dict[str, object]], audit["client_updates"])
    assert sum(int(record["selected_poison_records"]) for record in updates) == 2
