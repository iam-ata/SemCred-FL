"""Dataset ingestion interfaces for SemCred-FL."""

from semcred_fl.data.audit import AuditRequest, run_audit
from semcred_fl.data.ciciot2023 import CICIoT2023Adapter
from semcred_fl.data.contracts import (
    DataContractError,
    DatasetRecord,
    DatasetSchema,
    FileLoadResult,
    IngestionReport,
    InspectionSummary,
)
from semcred_fl.data.edge_iiotset import EdgeIIoTsetAdapter
from semcred_fl.data.preprocessing import (
    CIC_PREPARATION_POLICY,
    EDGE_PREPARATION_POLICY,
    FeatureValueError,
    normalize_feature_values,
)

__all__ = [
    "CICIoT2023Adapter",
    "CIC_PREPARATION_POLICY",
    "AuditRequest",
    "DataContractError",
    "DatasetRecord",
    "DatasetSchema",
    "EdgeIIoTsetAdapter",
    "EDGE_PREPARATION_POLICY",
    "FeatureValueError",
    "FileLoadResult",
    "IngestionReport",
    "InspectionSummary",
    "normalize_feature_values",
    "run_audit",
]
