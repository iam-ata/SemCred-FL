"""Tests for frozen post-audit dataset preparation policies."""

from __future__ import annotations

from dataclasses import replace

import pytest

from semcred_fl.data.contracts import DataContractError
from semcred_fl.data.preprocessing import (
    ABSENT_CATEGORY,
    CIC_PREPARATION_POLICY,
    EDGE_PREPARATION_POLICY,
    CategoryRule,
    ConflictEvidence,
    FeaturePreparationPolicy,
    FeatureValueError,
    canonical_numeric,
    normalize_feature_values,
)


def test_phase2c_evidence_freezes_raw_lower_bounds() -> None:
    cic = CIC_PREPARATION_POLICY.evidence
    edge = EDGE_PREPARATION_POLICY.evidence

    assert cic.raw_primary_consistent_unique_fingerprints == 20_594_360
    assert edge.raw_primary_consistent_unique_fingerprints == 15_199_406
    assert len(CIC_PREPARATION_POLICY.policy_id) == 64
    assert len(EDGE_PREPARATION_POLICY.policy_id) == 64


def test_edge_policy_covers_raw_features_and_drops_verified_constant() -> None:
    features = EDGE_PREPARATION_POLICY.features

    assert features.dropped_columns == ("icmp.unused",)
    assert "dns.qry.name.len" in features.continuous_columns
    assert set(features.categorical_columns) == {
        "http.request.method",
        "http.request.version",
        "mqtt.conack.flags",
        "mqtt.protoname",
    }
    assert "icmp.unused" not in features.prepared_feature_columns
    assert len(features.prepared_feature_columns) == 42


def test_edge_normalization_unifies_numeric_representations_and_absence() -> None:
    policy = EDGE_PREPARATION_POLICY.features
    values = dict.fromkeys(policy.source_feature_columns, "0.0")
    values.update(
        {
            "icmp.checksum": "0x10",
            "http.request.method": "0",
            "http.request.version": "HTTP/1.1",
            "mqtt.conack.flags": "0x00000000",
            "mqtt.protoname": "MQTT",
        }
    )

    normalized = dict(normalize_feature_values(values, policy))

    assert normalized["icmp.checksum"] == "16"
    assert normalized["http.request.method"] == ABSENT_CATEGORY
    assert normalized["http.request.version"] == "HTTP/1.1"
    assert normalized["mqtt.conack.flags"] == "0"
    assert normalized["mqtt.protoname"] == "MQTT"
    assert "icmp.unused" not in normalized


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("dns.qry.name.len", "raspberrypi.local", "Non-numeric"),
        ("http.request.version", "script>alert(1)", "Out-of-domain"),
        ("mqtt.conack.flags", "12", "Out-of-domain"),
        ("mqtt.conack.flags", "0.5", "Non-integer"),
        ("mqtt.protoname", "3", "Out-of-domain"),
    ],
)
def test_edge_normalization_rejects_observed_semantic_corruption(
    column: str, value: str, message: str
) -> None:
    policy = EDGE_PREPARATION_POLICY.features
    values = dict.fromkeys(policy.source_feature_columns, "0")
    values[column] = value

    with pytest.raises(FeatureValueError, match=message):
        normalize_feature_values(values, policy)


def test_numeric_normalization_rejects_text_and_nonfinite_values() -> None:
    assert canonical_numeric("1.0", "x") == "1"
    assert canonical_numeric("-0x10", "x") == "-16"
    with pytest.raises(FeatureValueError, match="Non-numeric"):
        canonical_numeric("payload", "x")
    with pytest.raises(FeatureValueError, match="Non-finite"):
        canonical_numeric("nan", "x")


def test_feature_policy_and_evidence_invariants_fail_closed() -> None:
    with pytest.raises(DataContractError, match="exactly one"):
        CategoryRule("x")
    with pytest.raises(DataContractError, match="exactly one"):
        CategoryRule("x", allowed_text_tokens=("a",), allowed_numeric_codes=(1,))
    with pytest.raises(DataContractError, match="cover"):
        FeaturePreparationPolicy(("x",), (), (), (), ())
    with pytest.raises(DataContractError, match="full lowercase"):
        replace(CIC_PREPARATION_POLICY.evidence, phase_2c_analysis_id="bad")
    with pytest.raises(DataContractError, match="cannot exceed valid"):
        ConflictEvidence("a" * 64, "b" * 64, 1, 2, 0, 0, 0, 0)


def test_normalization_rejects_missing_raw_columns() -> None:
    with pytest.raises(FeatureValueError, match="Missing raw model features"):
        normalize_feature_values({}, EDGE_PREPARATION_POLICY.features)
