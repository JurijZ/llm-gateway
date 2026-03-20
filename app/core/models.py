from typing import Dict, Tuple

# Mapping of friendly model names to (provider_name, actual_model_id)
MODEL_MAPPING: Dict[str, Tuple[str, str]] = {
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-5.2": ("openai", "gpt-5.2"),
    "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "claude-3-5-sonnet": ("anthropic", "claude-3-5-sonnet"),
    "claude-opus-4-6-1": ("anthropic", "claude-opus-4-6-1"),
    "claude-3-haiku": ("anthropic", "claude-3-haiku-20240307"),
}

# Mapping of provider names to their default models
PROVIDER_DEFAULTS: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-5-sonnet-20240620",
}

def get_model_info(preference: str) -> Tuple[str, str]:
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
