"""Error handling utilities for the agentic system."""

import asyncio
from typing import Optional, Any, Callable, TypeVar, Dict
from functools import wraps
import openai
from .logger import logger

T = TypeVar('T')


class AgentError(Exception):
    """Base exception for agent-related errors."""
    pass


class APIError(AgentError):
    """API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class ValidationError(AgentError):
    """Data validation errors."""
    pass


class ConfigurationError(AgentError):
    """Configuration-related errors."""
    pass


class ErrorHandler:
    """Centralized error handling for agent operations."""
    
    # Mapping of OpenAI error types to our custom errors
    OPENAI_ERROR_MAPPING = {
        openai.AuthenticationError: lambda e: APIError(f"Authentication failed: {e}", 401),
        openai.PermissionDeniedError: lambda e: APIError(f"Permission denied: {e}", 403),
        openai.RateLimitError: lambda e: APIError(f"Rate limit exceeded: {e}", 429, getattr(e, 'retry_after', None)),
        openai.BadRequestError: lambda e: APIError(f"Bad request: {e}", 400),
        openai.APIConnectionError: lambda e: APIError(f"API connection error: {e}"),
        openai.APITimeoutError: lambda e: APIError(f"API timeout: {e}"),
        openai.InternalServerError: lambda e: APIError(f"Internal server error: {e}", 500),
    }
    
    @classmethod
    def handle_openai_error(cls, error: Exception) -> AgentError:
        """Convert OpenAI errors to our custom error types."""
        error_type = type(error)
        
        if error_type in cls.OPENAI_ERROR_MAPPING:
            return cls.OPENAI_ERROR_MAPPING[error_type](error)
        
        # Fallback for unknown OpenAI errors
        return APIError(f"Unknown OpenAI error: {error}")
    
    @classmethod
    def is_retryable_error(cls, error: Exception) -> bool:
        """Determine if an error is retryable."""
        if isinstance(error, APIError):
            # Retry on server errors and rate limits
            return error.status_code in [429, 500, 502, 503, 504] or error.status_code is None
        
        if isinstance(error, (openai.RateLimitError, openai.APIConnectionError, 
                            openai.APITimeoutError, openai.InternalServerError)):
            return True
        
        return False
    
    @classmethod
    def get_retry_delay(cls, error: Exception, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate retry delay with exponential backoff."""
        if isinstance(error, APIError) and error.retry_after:
            return float(error.retry_after)
        
        # Exponential backoff: base_delay * (2 ^ attempt) with jitter
        import random
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter


def handle_errors(
    retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    handle_async: bool = True
):
    """Decorator for error handling with retry logic."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func) and handle_async:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                last_error = None
                
                for attempt in range(retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    
                    except Exception as e:
                        # Convert OpenAI errors to our custom errors
                        if "openai" in str(type(e)):
                            error = ErrorHandler.handle_openai_error(e)
                        else:
                            error = e
                        
                        last_error = error
                        
                        # Log the error
                        context = f"Attempt {attempt + 1}/{retries + 1}"
                        logger.log_error(func.__name__, error, context)
                        
                        # Don't retry on the last attempt or non-retryable errors
                        if attempt == retries or not ErrorHandler.is_retryable_error(error):
                            break
                        
                        # Calculate delay and wait
                        delay = min(
                            ErrorHandler.get_retry_delay(error, attempt, base_delay),
                            max_delay
                        )
                        logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds...")
                        await asyncio.sleep(delay)
                
                # Re-raise the last error if all retries failed
                raise last_error
            
            return async_wrapper
        
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                last_error = None
                
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    
                    except Exception as e:
                        # Convert OpenAI errors to our custom errors
                        if "openai" in str(type(e)):
                            error = ErrorHandler.handle_openai_error(e)
                        else:
                            error = e
                        
                        last_error = error
                        
                        # Log the error
                        context = f"Attempt {attempt + 1}/{retries + 1}"
                        logger.log_error(func.__name__, error, context)
                        
                        # Don't retry on the last attempt or non-retryable errors
                        if attempt == retries or not ErrorHandler.is_retryable_error(error):
                            break
                        
                        # Calculate delay and wait
                        delay = min(
                            ErrorHandler.get_retry_delay(error, attempt, base_delay),
                            max_delay
                        )
                        logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds...")
                        import time
                        time.sleep(delay)
                
                # Re-raise the last error if all retries failed
                raise last_error
            
            return sync_wrapper
    
    return decorator