"""Logging utilities for the agentic system."""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime

from .config import config


class AgentLogger:
    """Centralized logging for agent operations."""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.agent.log_level))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_file)
    
    def _setup_handlers(self, log_file: Optional[str] = None):
        """Setup logging handlers."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file or config.agent.enable_logging:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            if not log_file:
                log_file = f"agent_{datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = logging.FileHandler(log_dir / log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def log_api_request(self, agent_name: str, prompt_length: int, model: str):
        """Log API request details."""
        self.info(
            f"API Request - Agent: {agent_name}, Model: {model}, Prompt Length: {prompt_length}"
        )
    
    def log_api_response(self, agent_name: str, response_length: int, tokens_used: Optional[int] = None):
        """Log API response details."""
        message = f"API Response - Agent: {agent_name}, Response Length: {response_length}"
        if tokens_used:
            message += f", Tokens Used: {tokens_used}"
        self.info(message)
    
    def log_error(self, agent_name: str, error: Exception, context: Optional[str] = None):
        """Log error with context."""
        message = f"Error in {agent_name}: {str(error)}"
        if context:
            message += f" | Context: {context}"
        self.error(message, exc_info=True)


# Global logger instance
logger = AgentLogger("agentic_system")