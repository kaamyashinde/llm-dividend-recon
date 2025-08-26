"""Configuration management for the agentic system."""

import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class OpenAIConfig:
    """OpenAI API configuration."""
    api_key: str
    model: str = "gpt-4-turbo-preview"
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0


@dataclass
class AgentConfig:
    """Agent system configuration."""
    max_concurrent_requests: int = 5
    rate_limit_requests_per_minute: int = 50
    enable_logging: bool = True
    log_level: str = "INFO"
    response_timeout: int = 120


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self):
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables and defaults."""
        return {
            "openai": OpenAIConfig(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
                temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
                timeout=int(os.getenv("OPENAI_TIMEOUT", "60")),
                max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
                retry_delay=float(os.getenv("OPENAI_RETRY_DELAY", "1.0")),
                max_retry_delay=float(os.getenv("OPENAI_MAX_RETRY_DELAY", "60.0"))
            ),
            "agent": AgentConfig(
                max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5")),
                rate_limit_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "50")),
                enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                response_timeout=int(os.getenv("RESPONSE_TIMEOUT", "120"))
            )
        }
    
    @property
    def openai(self) -> OpenAIConfig:
        """Get OpenAI configuration."""
        return self._config["openai"]
    
    @property
    def agent(self) -> AgentConfig:
        """Get agent configuration."""
        return self._config["agent"]
    
    def validate_config(self) -> bool:
        """Validate that all required configuration is present."""
        if not self.openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True


# Global configuration instance
config = ConfigManager()