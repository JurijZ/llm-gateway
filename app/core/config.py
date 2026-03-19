from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from dotenv import load_dotenv
import os

# Explicitly load .env file
load_dotenv()

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "LLM Gateway"
    DEBUG: bool = False
    
    # LLM Provider API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Default Routing Strategy
    DEFAULT_STRATEGY: str = "hardcoded"  # hardcoded, load_balance, latency
    
    # Timeout settings (seconds)
    # TTFC_TIMEOUT: max wait for the *first* chunk — exceeding this triggers fallback.
    # CHUNK_TIMEOUT: max idle time *between* subsequent chunks — exceeding this means
    #                the stream stalled mid-response; no fallback is possible because
    #                bytes have already been sent to the client.
    TTFC_TIMEOUT: int = 10
    CHUNK_TIMEOUT: int = 30
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
