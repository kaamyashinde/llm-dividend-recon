"""Agent execution and orchestration utilities."""

import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import time
from pydantic import BaseModel

from .openai_client import openai_client
from .response_parser import ResponseParser, ParsedResponse, BaseAgentResponse
from .logger import logger
from .error_handler import AgentError, ValidationError
from .config import config


class AgentStatus(Enum):
    """Agent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentResult:
    """Result of agent execution."""
    agent_name: str
    status: AgentStatus
    response: Optional[ParsedResponse] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Task configuration for agent execution."""
    agent_name: str
    messages: List[Dict[str, str]]
    expected_response_type: Optional[Type[BaseAgentResponse]] = None
    timeout: float = 120.0
    priority: int = 0  # Higher number = higher priority
    dependencies: List[str] = field(default_factory=list)  # List of agent names this task depends on
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentExecutor:
    """Execute and orchestrate multiple agents."""
    
    def __init__(self, max_concurrent_tasks: int = None):
        self.max_concurrent_tasks = max_concurrent_tasks or config.agent.max_concurrent_requests
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self.results: Dict[str, AgentResult] = {}
        
    async def execute_single_agent(
        self,
        agent_name: str,
        messages: List[Dict[str, str]],
        expected_response_type: Optional[Type[BaseAgentResponse]] = None,
        timeout: float = 120.0,
        **openai_kwargs
    ) -> AgentResult:
        """Execute a single agent."""
        
        start_time = time.time()
        result = AgentResult(agent_name=agent_name, status=AgentStatus.RUNNING)
        
        try:
            logger.info(f"Starting execution of agent: {agent_name}")
            
            # Execute with timeout
            api_response = await asyncio.wait_for(
                openai_client.chat_completion(
                    messages=messages,
                    agent_name=agent_name,
                    **openai_kwargs
                ),
                timeout=timeout
            )
            
            # Parse the response
            parsed_response = ResponseParser.parse_response(
                api_response["content"],
                expected_format=expected_response_type,
                agent_name=agent_name
            )
            
            result.response = parsed_response
            result.tokens_used = api_response.get("usage", {}).get("total_tokens")
            result.status = AgentStatus.COMPLETED
            
            # Check for validation errors
            if parsed_response.validation_errors:
                logger.warning(
                    f"Agent {agent_name} completed with validation errors: "
                    f"{'; '.join(parsed_response.validation_errors)}"
                )
            
            logger.info(f"Agent {agent_name} completed successfully")
            
        except asyncio.TimeoutError:
            error_msg = f"Agent {agent_name} timed out after {timeout} seconds"
            result.error = AgentError(error_msg)
            result.status = AgentStatus.FAILED
            logger.error(error_msg)
            
        except Exception as e:
            result.error = e
            result.status = AgentStatus.FAILED
            logger.log_error(agent_name, e, "Agent execution failed")
            
        finally:
            result.execution_time = time.time() - start_time
            result.metadata.update({
                "execution_time": result.execution_time,
                "timestamp": time.time()
            })
        
        return result
    
    async def _execute_task_with_semaphore(self, task: AgentTask) -> AgentResult:
        """Execute a task with concurrency control."""
        async with self.semaphore:
            return await self.execute_single_agent(
                agent_name=task.agent_name,
                messages=task.messages,
                expected_response_type=task.expected_response_type,
                timeout=task.timeout
            )
    
    def _check_dependencies(self, task: AgentTask) -> bool:
        """Check if all dependencies for a task are completed."""
        for dep_name in task.dependencies:
            if dep_name not in self.results:
                return False
            if self.results[dep_name].status != AgentStatus.COMPLETED:
                return False
        return True
    
    def _get_dependency_results(self, task: AgentTask) -> Dict[str, AgentResult]:
        """Get results from task dependencies."""
        return {dep_name: self.results[dep_name] for dep_name in task.dependencies}
    
    async def execute_tasks(
        self, 
        tasks: List[AgentTask],
        fail_fast: bool = False
    ) -> Dict[str, AgentResult]:
        """Execute multiple tasks with dependency management."""
        
        logger.info(f"Starting execution of {len(tasks)} tasks")
        
        # Sort tasks by priority (higher priority first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Track pending tasks
        pending_tasks = {task.agent_name: task for task in sorted_tasks}
        running_tasks: Dict[str, asyncio.Task] = {}
        
        while pending_tasks or running_tasks:
            # Start tasks whose dependencies are satisfied
            for task_name, task in list(pending_tasks.items()):
                if self._check_dependencies(task):
                    # Remove from pending and start execution
                    del pending_tasks[task_name]
                    
                    # Add dependency results to task metadata
                    dep_results = self._get_dependency_results(task)
                    task.metadata["dependency_results"] = dep_results
                    
                    # Start the task
                    running_task = asyncio.create_task(
                        self._execute_task_with_semaphore(task)
                    )
                    running_tasks[task_name] = running_task
                    
                    logger.info(f"Started task: {task_name}")
            
            # Wait for at least one task to complete
            if running_tasks:
                done, running_task_objects = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for completed_task in done:
                    result = await completed_task
                    
                    # Find the task name for this result
                    task_name = result.agent_name
                    
                    # Store result and remove from running tasks
                    self.results[task_name] = result
                    del running_tasks[task_name]
                    
                    logger.info(f"Task completed: {task_name} ({result.status.value})")
                    
                    # Check if we should fail fast
                    if fail_fast and result.status == AgentStatus.FAILED:
                        # Cancel all running and pending tasks
                        for running_task in running_tasks.values():
                            running_task.cancel()
                        
                        logger.error(f"Failing fast due to task failure: {task_name}")
                        return self.results
            
            # If no tasks are running and there are still pending tasks,
            # it means we have unresolved dependencies
            if not running_tasks and pending_tasks:
                unresolved = list(pending_tasks.keys())
                error_msg = f"Circular or unresolved dependencies detected: {unresolved}"
                logger.error(error_msg)
                
                # Mark remaining tasks as failed
                for task_name in unresolved:
                    self.results[task_name] = AgentResult(
                        agent_name=task_name,
                        status=AgentStatus.FAILED,
                        error=AgentError(error_msg)
                    )
                
                break
        
        # Log summary
        completed = sum(1 for r in self.results.values() if r.status == AgentStatus.COMPLETED)
        failed = sum(1 for r in self.results.values() if r.status == AgentStatus.FAILED)
        total_time = sum(r.execution_time for r in self.results.values())
        
        logger.info(
            f"Task execution completed. "
            f"Completed: {completed}, Failed: {failed}, Total time: {total_time:.2f}s"
        )
        
        return self.results
    
    async def execute_pipeline(
        self, 
        agents_config: List[Dict[str, Any]],
        shared_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, AgentResult]:
        """Execute a pipeline of agents with shared context."""
        
        tasks = []
        shared_context = shared_context or {}
        
        for i, agent_config in enumerate(agents_config):
            # Build messages with shared context
            messages = agent_config.get("messages", [])
            
            # Add shared context to the first message if it's a system message
            if messages and messages[0].get("role") == "system" and shared_context:
                context_str = "\n".join([f"{k}: {v}" for k, v in shared_context.items()])
                messages[0]["content"] += f"\n\nShared Context:\n{context_str}"
            
            # Create task with dependencies on previous agents (sequential pipeline)
            dependencies = []
            if i > 0:
                dependencies = [agents_config[i-1]["name"]]
            
            task = AgentTask(
                agent_name=agent_config["name"],
                messages=messages,
                expected_response_type=agent_config.get("expected_response_type"),
                timeout=agent_config.get("timeout", 120.0),
                priority=agent_config.get("priority", 0),
                dependencies=dependencies,
                metadata=agent_config.get("metadata", {})
            )
            
            tasks.append(task)
        
        return await self.execute_tasks(tasks)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of execution results."""
        if not self.results:
            return {"message": "No tasks executed"}
        
        total_tasks = len(self.results)
        completed = sum(1 for r in self.results.values() if r.status == AgentStatus.COMPLETED)
        failed = sum(1 for r in self.results.values() if r.status == AgentStatus.FAILED)
        total_time = sum(r.execution_time for r in self.results.values())
        total_tokens = sum(r.tokens_used or 0 for r in self.results.values())
        
        return {
            "total_tasks": total_tasks,
            "completed": completed,
            "failed": failed,
            "success_rate": completed / total_tasks if total_tasks > 0 else 0,
            "total_execution_time": total_time,
            "total_tokens_used": total_tokens,
            "average_time_per_task": total_time / total_tasks if total_tasks > 0 else 0,
        }