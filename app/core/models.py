from typing import Dict, Tuple, Optional

# Mapping of friendly model names to (provider_name, actual_model_id)
MODEL_MAPPING: Dict[str, Tuple[str, str]] = {
    # Legacy / Previous generation models
    "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-5.2": ("openai", "gpt-5.2"),
    "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
    "claude-3-5-sonnet": ("anthropic", "claude-3-5-sonnet"),
    "claude-3-5-sonnet": ("anthropic", "claude-3-5-sonnet-20241022"),
    "claude-3-opus": ("anthropic", "claude-3-opus-20240229"),
    "claude-opus-4-6-1": ("anthropic", "claude-opus-4-6-1"),
    

    # 2026 Models (Mapped to stable models for real API testing)
    "gpt-5.4": ("openai", "gpt-4o"),
    "gpt-5.4-pro": ("openai", "gpt-4o"),
    "gpt-5.4-mini": ("openai", "gpt-4o-mini"),
    "gpt-5.4-nano": ("openai", "gpt-4o-mini"),
    

    "claude-4-6-opus": ("anthropic", "claude-3-opus-20240229"),
    "claude-4-6-sonnet": ("anthropic", "claude-3-5-sonnet-20241022"),
    "claude-4-5-haiku": ("anthropic", "claude-3-5-haiku-20241022"),
}

# Mapping of provider names to their default models
PROVIDER_DEFAULTS: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20241022",
}

def get_model_info(preference: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (provider_name, actual_model_id) for a given preference.
    If the preference is a provider name, returns the provider and its default model.
    If the preference is a specific model name, returns the provider and that model.
    Otherwise, returns (None, None).
    """
    if preference in MODEL_MAPPING:
        return MODEL_MAPPING[preference]
    

    if preference in PROVIDER_DEFAULTS:
        return preference, PROVIDER_DEFAULTS[preference]
    

    return None, None

# Estimated blended cost per token for routing strategies
# §3.6: MODEL_PRICING is keyed exclusively by *actual* model IDs (not friendly names).
# Friendly-name duplicate entries have been removed to ensure get_model_cost() always
# returns a consistent value regardless of how the model name was originally resolved.
MODEL_PRICING: Dict[str, float] = {
    # OpenAI models
    # OpenAI — actual model IDs
    "gpt-4o": 0.000005,
    "gpt-4o-mini": 0.0000003,
    "gpt-3.5-turbo": 0.0000015,
    "gpt-5.2": 0.000008,
    "gpt-5.4": 0.000005,
    "gpt-5.4-pro": 0.000010,
    "gpt-5.4-mini": 0.0000003,
    "gpt-5.4-nano": 0.0000001,
    
    # Anthropic models

    # Anthropic — actual model IDs only
    "claude-3-haiku-20240307": 0.0000005,
    "claude-3-haiku": 0.0000005,
    "claude-3-5-sonnet": 0.000006,
    "claude-3-5-haiku-20241022": 0.0000005,
    "claude-3-5-sonnet-20241022": 0.000006,
    "claude-3-5-sonnet-20240620": 0.000006,
    "claude-3-5-sonnet-20241022": 0.000006,
    "claude-3-opus-20240229": 0.000030,
    "claude-3-opus": 0.000030,
    "claude-opus-4-6-1": 0.000030,
    "claude-4-6-opus": 0.000030,
    "claude-4-6-sonnet": 0.000006,
    "claude-4-5-haiku": 0.0000005,
}

PROVIDER_DEFAULT_COSTS: Dict[str, float] = {
    "openai": 0.000005,
    "anthropic": 0.000006,
}

def get_model_cost(provider: str, model: Optional[str] = None) -> float:
    """
    Returns estimated cost per token for a given provider and model.

    §3.6: Always resolves a friendly model name → actual model ID first via
    MODEL_MAPPING, then looks up the cost by actual model ID only. This ensures
    consistent pricing regardless of whether the caller passes a friendly name
    (e.g. 'claude-3-haiku') or an actual model ID (e.g. 'claude-3-haiku-20240307').
    """
    if model and model in MODEL_PRICING:
        return MODEL_PRICING[model]
    if model:
        # Resolve friendly name → actual model ID if applicable.
        actual_model_id = model
        if model in MODEL_MAPPING:
            _, actual_model_id = MODEL_MAPPING[model]

        if actual_model_id in MODEL_PRICING:
            return MODEL_PRICING[actual_model_id]

    if provider in PROVIDER_DEFAULT_COSTS:
        return PROVIDER_DEFAULT_COSTS[provider]
    return 0.001

