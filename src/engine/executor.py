import logging
import sys
from typing import Dict, Any, Type
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from prometheus_client import Counter, Histogram, Gauge
from engine.workflow import Workflow, WorkflowValidationError
from languages import registry

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
executor_executions = Counter('executor_executions_total', 'Total code executions', ['language'])
execution_errors = Counter('execution_errors_total', 'Total execution errors', ['language'])
execution_duration = Histogram('execution_duration_seconds', 'Code execution duration', ['language'])
resource_usage = Gauge('resource_usage', 'Resource usage during execution', ['language', 'resource_type'])

class ExecutorError(Exception):
    """Exception raised for errors in the code execution process."""
    pass

@asynccontextmanager
async def execution_context(language: str, description: str):
    """Async context manager for code execution."""
    logger.info(f"Starting execution: {description} for {language}")
    executor_executions.labels(language=language).inc()
    try:
        yield
    except Exception as e:
        execution_errors.labels(language=language).inc()
        logger.error(f"Execution failed: {description} - {str(e)}")
        raise ExecutorError(f"Execution failed: {str(e)}")
    finally:
        logger.info(f"Completed execution: {description}")
        resource_usage.labels(language=language, resource_type="memory").set(sys.getsizeof({}))

class Executor(ABC):
    """Abstract base class for code executors."""
    @abstractmethod
    async def execute(self, code: str, context: Dict[str, Any]) -> Any:
        """
        Execute the generated code with the given context.

        Args:
            code: The code to execute.
            context: Execution context including inputs and state.

        Returns:
            Any: Execution result.

        Raises:
            ExecutorError: If execution fails.
        """
        pass

async def execute_workflow(workflow: Workflow, inputs: Dict[str, Any], context: Dict[str, Any]) -> Any:
    """
    Execute a workflow by generating and executing code for the target language.

    Args:
        workflow: The Workflow object containing function and steps.
        inputs: Input data for the workflow.
        context: Context data (e.g., game state, blockchain data).

    Returns:
        Any: Execution result.

    Raises:
        ExecutorError: If code generation or execution fails.
    """
    language = workflow.metadata.get("target_language", "python")
    async with execution_context(language, f"Workflow {workflow.function}"):
        try:
            # Get generator and generate code
            generator = registry.get_generator(language)
            code = generator.generate(workflow)

            # Get executor and execute code
            executor = registry.get_executor(language)
            result = await executor.execute(code, context)

            # Validate outputs
            for key, spec in workflow.schema_data.get("outputs", {}).items():
                if key not in result and spec.get("constraints", {}).get("required"):
                    raise ExecutorError(f"Missing required output: {key}")

            return result

        except Exception as e:
            raise ExecutorError(f"Workflow execution failed: {str(e)}")
