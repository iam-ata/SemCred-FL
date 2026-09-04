"""Offline, immutable semantic-prior construction for SemCred-FL."""

from semcred_fl.semantic.attribution import GroupedAttribution, grouped_gradient_times_input
from semcred_fl.semantic.contracts import (
    LLMInvocationMetadata,
    LLMRelevanceResponse,
    PublicClass,
    PublicFeatureGroup,
    PublicSemanticMetadata,
    PublicSource,
    SemanticPriorError,
    load_invocation_metadata,
    load_public_metadata,
    parse_llm_response,
)
from semcred_fl.semantic.openai_provider import (
    OPENAI_API_KEY_ENVIRONMENT,
    OPENAI_MODEL_SNAPSHOT,
    OpenAIProviderError,
    build_openai_request,
    invoke_openai_semantic_prior,
    relevance_json_schema,
)
from semcred_fl.semantic.pipeline import (
    SemanticPriorBundle,
    build_semantic_prior_bundle,
    load_semantic_prior_bundle,
    prepare_semantic_prompt,
)
from semcred_fl.semantic.prior import (
    AnchorDataPrior,
    FeatureGroupLayout,
    compute_anchor_data_prior,
    fuse_priors,
    load_feature_group_layout,
    normalize_prior,
)

__all__ = [
    "AnchorDataPrior",
    "FeatureGroupLayout",
    "GroupedAttribution",
    "LLMInvocationMetadata",
    "LLMRelevanceResponse",
    "OPENAI_API_KEY_ENVIRONMENT",
    "OPENAI_MODEL_SNAPSHOT",
    "OpenAIProviderError",
    "PublicClass",
    "PublicFeatureGroup",
    "PublicSemanticMetadata",
    "PublicSource",
    "SemanticPriorBundle",
    "SemanticPriorError",
    "build_semantic_prior_bundle",
    "build_openai_request",
    "compute_anchor_data_prior",
    "fuse_priors",
    "grouped_gradient_times_input",
    "load_feature_group_layout",
    "load_invocation_metadata",
    "load_public_metadata",
    "load_semantic_prior_bundle",
    "normalize_prior",
    "parse_llm_response",
    "prepare_semantic_prompt",
    "invoke_openai_semantic_prior",
    "relevance_json_schema",
]
