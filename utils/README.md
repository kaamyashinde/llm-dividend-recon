# Utils Package

A comprehensive utility package for building agentic AI systems with OpenAI integration.

## Features

- **OpenAI API Client**: Enhanced client with rate limiting, retry logic, and error handling
- **Agent Execution**: Orchestrate multiple agents with dependency management and concurrency control  
- **Response Parsing**: Parse and validate agent responses with Pydantic models
- **Error Handling**: Robust error handling with automatic retries and fallback strategies
- **Configuration**: Centralized configuration management with environment variable support
- **Logging**: Structured logging for debugging and monitoring

## Quick Start

### 1. Environment Setup

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.1
LOG_LEVEL=INFO
```

### 2. Basic Usage

```python
from utils import openai_client, AgentExecutor, logger

# Simple agent execution
async def run_single_agent():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze this data..."}
    ]
    
    result = await agent_executor.execute_single_agent(
        agent_name="data_analyzer",
        messages=messages,
        timeout=60.0
    )
    
    if result.status == AgentStatus.COMPLETED:
        print(f"Agent completed: {result.response.content}")
    else:
        print(f"Agent failed: {result.error}")

# Multiple agent orchestration
async def run_agent_pipeline():
    executor = AgentExecutor()
    
    tasks = [
        AgentTask(
            agent_name="validator",
            messages=[...],
            dependencies=[]
        ),
        AgentTask(
            agent_name="analyzer", 
            messages=[...],
            dependencies=["validator"]
        ),
        AgentTask(
            agent_name="reporter",
            messages=[...], 
            dependencies=["analyzer"]
        )
    ]
    
    results = await executor.execute_tasks(tasks)
    summary = executor.get_results_summary()
    print(f"Pipeline completed: {summary}")
```

### 3. Response Validation

```python
from utils import ResponseParser, ValidationResponse
from pydantic import BaseModel

class CustomResponse(BaseModel):
    analysis: str
    confidence: float
    recommendations: List[str]

# Parse and validate response
parsed = ResponseParser.parse_response(
    response_content=api_response,
    expected_format=CustomResponse,
    agent_name="analyzer"
)

if parsed.validation_errors:
    print(f"Validation errors: {parsed.validation_errors}")
else:
    print(f"Structured data: {parsed.structured_data}")
```

## Configuration

The package supports configuration via environment variables:

### OpenAI Configuration
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo-preview)  
- `OPENAI_TEMPERATURE`: Temperature setting (default: 0.1)
- `OPENAI_MAX_TOKENS`: Max tokens per request (default: 4000)
- `OPENAI_TIMEOUT`: Request timeout in seconds (default: 60)
- `OPENAI_MAX_RETRIES`: Number of retries on failure (default: 3)

### Agent Configuration  
- `MAX_CONCURRENT_REQUESTS`: Max concurrent API requests (default: 5)
- `RATE_LIMIT_RPM`: Rate limit in requests per minute (default: 50)
- `ENABLE_LOGGING`: Enable file logging (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)
- `RESPONSE_TIMEOUT`: Agent response timeout (default: 120)

## Architecture

### Core Components

1. **OpenAI Client** (`openai_client.py`)
   - Async OpenAI API wrapper
   - Built-in rate limiting
   - Automatic retry with exponential backoff
   - Request/response logging

2. **Agent Executor** (`agent_executor.py`)
   - Execute single or multiple agents
   - Dependency management between agents
   - Concurrency control with semaphores
   - Pipeline execution support

3. **Response Parser** (`response_parser.py`) 
   - Extract structured data from text responses
   - Pydantic model validation
   - Error handling and reporting

4. **Error Handler** (`error_handler.py`)
   - Custom exception hierarchy
   - Retry logic with configurable backoff
   - OpenAI error mapping

5. **Configuration** (`config.py`)
   - Environment-based configuration
   - Validation and defaults
   - Type-safe configuration classes

6. **Logger** (`logger.py`)
   - Structured logging
   - API request/response tracking
   - Error logging with context

## Error Handling

The package includes comprehensive error handling:

```python
from utils import handle_errors, APIError

@handle_errors(retries=3, base_delay=1.0)
async def my_agent_function():
    # Your code here
    pass

# Custom error handling
try:
    result = await openai_client.chat_completion(...)
except APIError as e:
    if e.status_code == 429:  # Rate limit
        print(f"Rate limited, retry after: {e.retry_after}")
    else:
        print(f"API error: {e}")
```

## Advanced Features

### Streaming Responses

```python
async for chunk in openai_client.streaming_chat_completion(messages):
    print(chunk, end="", flush=True)
```

### Agent Dependencies

```python
tasks = [
    AgentTask("step1", messages1, dependencies=[]),
    AgentTask("step2", messages2, dependencies=["step1"]), 
    AgentTask("step3", messages3, dependencies=["step1", "step2"])
]

results = await executor.execute_tasks(tasks)
```

### Custom Response Models

```python
from utils import BaseAgentResponse

class MyCustomResponse(BaseAgentResponse):
    data: Dict[str, Any]
    processed_items: List[str]
    metadata: Optional[Dict[str, Any]] = None
```

## Best Practices

1. **Always set appropriate timeouts** for your use case
2. **Use structured response models** for validation
3. **Handle errors gracefully** with proper fallbacks
4. **Monitor token usage** and costs
5. **Use logging** for debugging and monitoring
6. **Set up proper rate limiting** to avoid API limits
7. **Use dependencies** to create logical agent workflows

## Troubleshooting

### Common Issues

1. **API Key not set**: Ensure `OPENAI_API_KEY` environment variable is set
2. **Rate limiting**: Reduce `RATE_LIMIT_RPM` or increase delays
3. **Timeouts**: Increase timeout values for complex tasks
4. **Memory usage**: Monitor for large responses and limit `max_tokens`

### Debug Logging

Enable debug logging to see detailed API interactions:

```python
import logging
logging.getLogger("agentic_system").setLevel(logging.DEBUG)
```