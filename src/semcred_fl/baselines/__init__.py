"""Reference federated aggregation baselines."""

from semcred_fl.baselines.agat_fl import agat_trust_aggregate
from semcred_fl.baselines.coordinate_median import coordinate_median
from semcred_fl.baselines.fed_lsae import (
    centered_kernel_alignment,
    fed_lsae_aggregate,
    fed_lsae_partition,
    rbf_cka,
    rbf_kernel,
)
from semcred_fl.baselines.fedavg import AggregationResult, fedavg
from semcred_fl.baselines.fedkd_ids import (
    ABSTAIN_LABEL,
    FedKdIdsAggregationResult,
    aggregate_public_labels,
    apply_local_verifier_mask,
)
from semcred_fl.baselines.multi_krum import multi_krum
from semcred_fl.baselines.trimmed_mean import trimmed_mean

__all__ = [
    "ABSTAIN_LABEL",
    "AggregationResult",
    "FedKdIdsAggregationResult",
    "agat_trust_aggregate",
    "aggregate_public_labels",
    "apply_local_verifier_mask",
    "centered_kernel_alignment",
    "coordinate_median",
    "fed_lsae_aggregate",
    "fed_lsae_partition",
    "fedavg",
    "multi_krum",
    "rbf_cka",
    "rbf_kernel",
    "trimmed_mean",
]
