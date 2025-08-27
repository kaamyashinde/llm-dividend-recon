"""
Utility modules for the agentic system.

This package provides:
- OpenAI API client with rate limiting and error handling
- Agent execution and orchestration
- Response parsing and validation
- Configuration management
- Logging utilities
- Error handling with retry logic
"""

from .config import config, ConfigManager, OpenAIConfig, AgentConfig
from .logger import logger, AgentLogger
from .error_handler import (
    AgentError, 
    APIError, 
    ValidationError, 
    ConfigurationError,
    ErrorHandler,
    handle_errors
)
from .openai_client import openai_client, OpenAIClient, RateLimiter
from .response_parser import (
    ResponseParser, 
    ParsedResponse,
    BaseAgentResponse,
    ListResponse,
    AnalysisResponse,
    ValidationResponse,
    ReportResponse
)
from .agent_executor import (
    AgentExecutor,
    AgentTask,
    AgentResult,
    AgentStatus
)

__all__ = [
    # Configuration
    "config",
    "ConfigManager", 
    "OpenAIConfig",
    "AgentConfig",
    
    # Logging
    "logger",
    "AgentLogger",
    
    # Error handling
    "AgentError",
    "APIError", 
    "ValidationError",
    "ConfigurationError",
    "ErrorHandler",
    "handle_errors",
    
    # OpenAI client
    "openai_client",
    "OpenAIClient",
    "RateLimiter",
    
    # Response parsing
    "ResponseParser",
    "ParsedResponse",
    "BaseAgentResponse",
    "ListResponse", 
    "AnalysisResponse",
    "ValidationResponse",
    "ReportResponse",
    
    # Agent execution
    "AgentExecutor",
    "AgentTask", 
    "AgentResult",
    "AgentStatus"
]

# Version info
__version__ = "1.0.0"

# Package metadata
__author__ = "LLM Dividend Recon System"
__description__ = "Utilities for agentic AI system with OpenAI integration"