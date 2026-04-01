from typing import Dict, Tuple, Optional

# Mapping of friendly model names to (provider_name, actual_model_id)
MODEL_MAPPING: Dict[str, Tuple[str, str]] = {
    # Legacy / Previous generation models
    "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-5.2": ("openai", "gpt-5.2"),
    "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
    "claude-3-5-sonnet": ("anthropic", "claude-3-5-sonnet"),
    "claude-3-opus": ("anthropic", "claude-3-opus-20240229"),
    "claude-opus-4-6-1": ("anthropic", "claude-opus-4-6-1"),
    
    # 2026 Models
    "gpt-5.4": ("openai", "gpt-5.4"),
    "gpt-5.4-pro": ("openai", "gpt-5.4-pro"),
    "gpt-5.4-mini": ("openai", "gpt-5.4-mini"),
    "gpt-5.4-nano": ("openai", "gpt-5.4-nano"),
    
    "claude-4-6-opus": ("anthropic", "claude-4-6-opus-latest"),
    "claude-4-6-sonnet": ("anthropic", "claude-4-6-sonnet-latest"),
    "claude-4-5-haiku": ("anthropic", "claude-4-5-haiku-latest"),
}

# Mapping of provider names to their default models
PROVIDER_DEFAULTS: Dict[str, str] = {
    "openai": "gpt-5.4",
    "anthropic": "claude-4-6-sonnet-latest",
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
