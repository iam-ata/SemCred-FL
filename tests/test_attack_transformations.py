"""Exact tests for Phase 6B artifacts and poisoning transformations."""

from __future__ import annotations

import math

import pytest
import torch

from semcred_fl.attacks.artifacts import (
    AttackArtifactError,
    Phase6BResolution,
    ResolvedCoordinate,
    TriggerCoordinate,
    benign_norm_reference,
)
from semcred_fl.attacks.transformations import (
    backdoor_poison,
    backdoor_poison_selected,
    clean_label_poison,
    coalition_model_replacement,
    constrain_update_norm,
    reverse_update,
    scale_update,
    selected_source_indices,
    targeted_label_flip,
    targeted_label_flip_selected,
    triggered_test_batch,
)

IDENTITY = "a" * 64


def _coordinate(index: int) -> ResolvedCoordinate:
    source_median = float(index + 2)
    benign_median = float(index) / 10.0
    return ResolvedCoordinate(
        coordinate=f"x{index:04d}",
        index=index,
        source_feature=f"feature-{index}",
        lower_bound=-2.0,
        upper_bound=2.0,
        source_median=source_median,
        benign_median=benign_median,
        absolute_median_gap=abs(source_median - benign_median),
    )


def _resolution() -> Phase6BResolution:
    ranked = tuple(_coordinate(index) for index in range(12))
    trigger = tuple(
        TriggerCoordinate(
            coordinate=ranked[index].coordinate,
            index=ranked[index].index,
            source_feature=ranked[index].source_feature,
            endpoint="lower",
            value=-2.0,
        )
        for index in range(3)
    )
    return Phase6BResolution(
        dataset="fixture",
        phase6a_contract_id=IDENTITY,
        preprocessed_corpus_id=IDENTITY,
        preprocessor_id=IDENTITY,
        split_id=IDENTITY,
        anchor_id=IDENTITY,
        client_partition_id=IDENTITY,
        source_family="source",
        source_family_id=1,
        target_family="benign",
        target_family_id=0,
        source_training_records=20,
        target_training_records=30,
        federated_training_records=100,
        ranked_coordinates=ranked,
        feature_coordinates=ranked[:3],
        trigger_coordinates=trigger,
        trigger_training_collisions=0,
    )


def test_sha256_selection_is_deterministic_without_replacement() -> None:
    targets = torch.tensor([1, 0, 1, 1, 2, 1, 1, 0])
    record_ids = tuple(f"row-{index}" for index in range(len(targets)))
    first = selected_source_indices(
        targets=targets,
        record_ids=record_ids,
        source_family_id=1,
        poison_fraction=0.5,
        run_seed=7,
        attack_id="flip",
        client_id="client-03",
    )
    second = selected_source_indices(
        targets=targets,
        record_ids=record_ids,
        source_family_id=1,
        poison_fraction=0.5,
        run_seed=7,
        attack_id="flip",
        client_id="client-03",
    )
    assert first == second
    assert len(first) == 2
    assert len(set(first)) == 2
    assert all(targets[index].item() == 1 for index in first)


def test_zero_fraction_is_an_exact_tensor_no_op() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    targets = torch.tensor([1, 0, 1, 2])
    result = targeted_label_flip(
        features=features,
        targets=targets,
        record_ids=("a", "b", "c", "d"),
        source_family_id=1,
        target_family_id=0,
        poison_fraction=0.0,
        run_seed=1,
        attack_id="flip",
        client_id="client-00",
    )
    assert result.features is features
    assert result.targets is targets
    assert result.selected_indices == ()


def test_targeted_label_flip_changes_only_selected_labels() -> None:
    features = torch.zeros((4, 12))
    targets = torch.tensor([1, 1, 0, 2])
    result = targeted_label_flip(
        features=features,
        targets=targets,
        record_ids=("a", "b", "c", "d"),
        source_family_id=1,
        target_family_id=0,
        poison_fraction=1.0,
        run_seed=1,
        attack_id="flip",
        client_id="client-00",
    )
    assert result.features is features
    assert torch.equal(result.targets, torch.tensor([0, 0, 0, 2]))
    assert result.selected_indices == (0, 1)


def test_population_selection_is_reused_exactly_across_streamed_batches() -> None:
    selected_ids = ("source-1", "source-3")
    first_features = torch.zeros((3, 12))
    first_targets = torch.tensor([1, 0, 1])
    first = targeted_label_flip_selected(
        features=first_features,
        targets=first_targets,
        record_ids=("source-1", "benign-1", "source-2"),
        selected_record_ids=selected_ids,
        source_family_id=1,
        target_family_id=0,
        poison_fraction=0.5,
        attack_id="flip",
    )
    second_features = torch.ones((2, 12))
    second_targets = torch.tensor([1, 0])
    second = targeted_label_flip_selected(
        features=second_features,
        targets=second_targets,
        record_ids=("source-3", "benign-2"),
        selected_record_ids=selected_ids,
        source_family_id=1,
        target_family_id=0,
        poison_fraction=0.5,
        attack_id="flip",
    )
    assert first.selected_indices == (0,)
    assert second.selected_indices == (0,)
    assert torch.equal(first.targets, torch.tensor([0, 0, 1]))
    assert torch.equal(second.targets, torch.tensor([0, 0]))
    assert first.features is first_features
    assert second.features is second_features


def test_empty_population_selection_is_an_exact_streaming_no_op() -> None:
    features = torch.zeros((2, 12))
    targets = torch.tensor([1, 0])
    result = backdoor_poison_selected(
        features=features,
        targets=targets,
        record_ids=("source", "benign"),
        selected_record_ids=(),
        resolution=_resolution(),
        poison_fraction=0.0,
        attack_id="backdoor",
    )
    assert result.features is features
    assert result.targets is targets
    assert result.selected_indices == ()


def test_clean_label_poison_preserves_labels_and_projects_bounds() -> None:
    resolution = _resolution()
    features = torch.full((2, 12), 10.0)
    targets = torch.tensor([1, 0])
    result = clean_label_poison(
        features=features,
        targets=targets,
        record_ids=("source", "benign"),
        resolution=resolution,
        poison_fraction=1.0,
        strength=1.0,
        run_seed=2,
        attack_id="clean-label",
        client_id="client-01",
    )
    assert result.targets is targets
    assert result.selected_indices == (0,)
    assert torch.allclose(result.features[0, :3], torch.tensor([0.0, 0.1, 0.2]))
    assert torch.equal(result.features[1], features[1])


def test_backdoor_sets_trigger_and_target_only_on_selected_rows() -> None:
    resolution = _resolution()
    features = torch.zeros((3, 12))
    targets = torch.tensor([1, 2, 1])
    result = backdoor_poison(
        features=features,
        targets=targets,
        record_ids=("a", "b", "c"),
        resolution=resolution,
        poison_fraction=1.0,
        run_seed=3,
        attack_id="backdoor",
        client_id="client-02",
    )
    assert result.selected_indices == (0, 2)
    assert torch.equal(result.targets, torch.tensor([0, 2, 0]))
    assert torch.all(result.features[[0, 2], :3] == -2.0).item()
    assert torch.equal(result.features[1], features[1])


def test_triggered_test_generator_excludes_natural_collisions() -> None:
    resolution = _resolution()
    features = torch.zeros((4, 12))
    features[0, :3] = -2.0
    targets = torch.tensor([1, 1, 0, 2])
    result = triggered_test_batch(features=features, targets=targets, resolution=resolution)
    assert result.source_records == 2
    assert result.natural_collisions == 1
    assert result.asr_denominator == 1
    assert result.original_indices == (1,)
    assert torch.all(result.features[:, :3] == -2.0).item()
    assert torch.equal(result.source_targets, torch.tensor([1]))


def test_direct_model_formulas_match_analytical_values() -> None:
    delta = torch.tensor([1.0, -2.0])
    assert torch.equal(scale_update(delta, 5.0), torch.tensor([5.0, -10.0]))
    assert torch.equal(reverse_update(delta, 5.0), torch.tensor([-5.0, 10.0]))
    replacement = coalition_model_replacement(
        target_parameters=torch.tensor([3.0, 5.0]),
        global_parameters=torch.tensor([1.0, 1.0]),
        participants_per_round=20,
        coalition_size=4,
    )
    assert torch.equal(replacement, torch.tensor([10.0, 20.0]))


def test_adaptive_norm_projection_clips_both_sides() -> None:
    assert math.isclose(
        float(torch.linalg.vector_norm(constrain_update_norm(torch.tensor([3.0, 4.0]), 1, 2))),
        2.0,
    )
    assert math.isclose(
        float(torch.linalg.vector_norm(constrain_update_norm(torch.tensor([0.3, 0.4]), 1, 2))),
        1.0,
    )
    unchanged = torch.tensor([0.6, 0.8])
    assert constrain_update_norm(unchanged, 0.5, 2.0) is unchanged
    with pytest.raises(ValueError, match="no direction"):
        constrain_update_norm(torch.zeros(2), 1.0, 2.0)


def test_norm_reference_is_order_invariant_and_uses_linear_quartiles() -> None:
    ids = tuple(f"{value:064x}" for value in range(1, 5))
    first = benign_norm_reference(
        dataset="fixture",
        profile="iid",
        preprocessed_corpus_id=IDENTITY,
        client_partition_id=IDENTITY,
        detector_config_id=IDENTITY,
        local_training_config_id=IDENTITY,
        development_run_ids=ids,
        update_norms=(1.0, 2.0, 3.0, 4.0),
    )
    second = benign_norm_reference(
        dataset="fixture",
        profile="iid",
        preprocessed_corpus_id=IDENTITY,
        client_partition_id=IDENTITY,
        detector_config_id=IDENTITY,
        local_training_config_id=IDENTITY,
        development_run_ids=tuple(reversed(ids)),
        update_norms=(4.0, 3.0, 2.0, 1.0),
    )
    assert first.reference_id == second.reference_id
    assert first.lower_norm == 1.75
    assert first.upper_norm == 3.25


def test_artifact_rejects_any_evaluation_access() -> None:
    resolution = _resolution()
    with pytest.raises(AttackArtifactError, match="evidence boundary"):
        Phase6BResolution(
            **{
                name: getattr(resolution, name)
                for name in resolution.__dataclass_fields__
                if name != "test_partition_accessed"
            },
            test_partition_accessed=True,
        )
