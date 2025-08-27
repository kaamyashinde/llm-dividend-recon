"""OpenAI API client with advanced features."""

import asyncio
import time
from typing import Dict, Any, Optional, List, AsyncIterator
from dataclasses import asdict
import openai
from openai import AsyncOpenAI

from .config import config
from .logger import logger
from .error_handler import handle_errors, APIError, ConfigurationError


class RateLimiter:
    """Simple rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 50):
        self.requests_per_minute = requests_per_minute
        self.requests = []
    
    async def acquire(self):
        """Acquire permission to make a request."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests 
                        if current_time - req_time < 60]
        
        # Check if we're at the limit
        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            oldest_request = min(self.requests)
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.requests.append(current_time)


class OpenAIClient:
    """Enhanced OpenAI client with rate limiting, retries, and logging."""
    
    def __init__(self):
        # Validate configuration
        config.validate_config()
        
        self.client = AsyncOpenAI(
            api_key=config.openai.api_key,
            timeout=config.openai.timeout,
            max_retries=0  # We handle retries ourselves
        )
        
        self.rate_limiter = RateLimiter(config.agent.rate_limit_requests_per_minute)
        
        logger.info(f"OpenAI client initialized with model: {config.openai.model}")
    
    @handle_errors(
        retries=config.openai.max_retries,
        base_delay=config.openai.retry_delay,
        max_delay=config.openai.max_retry_delay
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        agent_name: str = "unknown",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a chat completion with rate limiting and error handling."""
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Use config defaults if not provided
        model = model or config.openai.model
        temperature = temperature if temperature is not None else config.openai.temperature
        max_tokens = max_tokens or config.openai.max_tokens
        
        # Calculate prompt length for logging
        prompt_length = sum(len(msg.get("content", "")) for msg in messages)
        
        logger.log_api_request(agent_name, prompt_length, model)
        
        try:
            # Prepare request parameters
            request_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }
            
            # Add response format if specified
            if response_format:
                request_params["response_format"] = response_format
            
            # Make the API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Log successful response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else None
            
            logger.log_api_response(
                agent_name, 
                len(content) if content else 0, 
                tokens_used
            )
            
            return {
                "content": content,
                "usage": response.usage.model_dump() if response.usage else None,
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.log_error(agent_name, e, f"Chat completion failed")
            raise
    
    @handle_errors(
        retries=config.openai.max_retries,
        base_delay=config.openai.retry_delay,
        max_delay=config.openai.max_retry_delay
    )
    async def streaming_chat_completion(
        self,
        messages: List[Dict[str, str]],
        agent_name: str = "unknown",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Create a streaming chat completion."""
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Use config defaults if not provided
        model = model or config.openai.model
        temperature = temperature if temperature is not None else config.openai.temperature
        max_tokens = max_tokens or config.openai.max_tokens
        
        # Calculate prompt length for logging
        prompt_length = sum(len(msg.get("content", "")) for msg in messages)
        
        logger.log_api_request(agent_name, prompt_length, model)
        
        try:
            # Make the streaming API call
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            full_content = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield content
            
            # Log the complete response
            logger.log_api_response(agent_name, len(full_content))
            
        except Exception as e:
            logger.log_error(agent_name, e, f"Streaming chat completion failed")
            raise
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()


# Global client instance
openai_client = OpenAIClient()