# Copyright (c) 2024 James Chapman
#
# This software is dual-licensed:
#
# - For individuals and non-commercial use: Licensed under the MIT License.
# - For commercial or corporate use: A separate commercial license is required.
#
# To obtain a commercial license, please contact: iconoclastdao@gmail.com
#
# By using this software, you agree to these terms.
#
# MIT License (for individuals and non-commercial use):
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import sys
import json
import importlib
import pkgutil
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram
from engine.workflow import Workflow, WorkflowValidationError
from engine.validator import validate_workflow
from engine.generator import generate_workflow
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

class ExecutorPluginInterface(ABC):
    """Interface for executor plugins."""
    @abstractmethod
    async def execute_step(self, step_type: str, step_data: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a specific step type."""
        pass

    @abstractmethod
    def get_step_types(self) -> List[str]:
        """Return the step types supported by the plugin."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return plugin metadata."""
        pass

class ExecutorPluginManager:
    """Manages executor plugins."""
    def __init__(self):
        self.plugins: Dict[str, ExecutorPluginInterface] = {}

    def load_plugins(self, plugin_dir: str = "languages.plugins") -> None:
        """
        Dynamically load executor plugins.

        Args:
            plugin_dir: Package path for plugins.
        """
        try:
            package = importlib.import_module(plugin_dir)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{plugin_dir}.{module_name}")
                    if hasattr(module, "ExecutorPlugin"):
                        plugin = module.ExecutorPlugin()
                        metadata = plugin.get_metadata()
                        plugin_name = metadata.get("name", module_name)
                        self.plugins[plugin_name] = plugin
                        logger.info(f"Loaded executor plugin: {plugin_name} (version: {metadata.get('version', 'unknown')})")
                except Exception as e:
                    logger.error(f"Failed to load executor plugin {module_name}: {str(e)}", exc_info=True)
        except ImportError as e:
            logger.error(f"Failed to import executor plugin package {plugin_dir}: {str(e)}", exc_info=True)

    def get_supported_steps(self) -> Dict[str, ExecutorPluginInterface]:
        """
        Get all supported step types and their corresponding plugins.

        Returns:
            Dict[str, ExecutorPluginInterface]: Mapping of step types to plugins.
        """
        supported_steps = {}
        for plugin in self.plugins.values():
            for step_type in plugin.get_step_types():
                if step_type in supported_steps:
                    logger.warning(f"Duplicate step type {step_type} registered by plugin {plugin.get_metadata()['name']}")
                supported_steps[step_type] = plugin
        return supported_steps

@asynccontextmanager
async def execution_context(language: str, description: str):
    """
    Async context manager for code execution with metrics and logging.

    Args:
        language: Target language for execution.
        description: Description of the execution task.
    """
    logger.info(f"Starting execution: {description} for {language}")
    executor_executions.labels(language=language).inc()
    start_time = datetime.now()
    try:
        yield
    except Exception as e:
        execution_errors.labels(language=language).inc()
        logger.error(f"Execution failed: {description} - {str(e)}", exc_info=True)
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        execution_duration.labels(language=language).observe(duration)
        logger.info(f"Completed execution: {description} in {duration:.2f}s")
        resource_usage.labels(language=language, resource_type="memory").set(sys.getsizeof({}))

class Executor(ABC):
    """Abstract base class for code executors."""
    def __init__(self):
        self.plugin_manager = ExecutorPluginManager()
        self.plugin_manager.load_plugins()
        self.plugin_steps = self.plugin_manager.get_supported_steps()
        self.supported_steps = {
            "set", "if", "return", "call", "try", "while", "foreach", "parallel",
            "assert", "event", "require_role"
        }

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
            # Validate workflow
            validate_workflow(workflow)

            # Validate inputs against schema
            for key, spec in workflow.schema_data.get("inputs", {}).items():
                if key not in inputs and spec.get("constraints", {}).get("required", False):
                    raise ExecutorError(f"Missing required input: {key}")

            # Generate code
            code = generate_workflow(workflow)

            # Get executor and execute code
            executor = registry.get_executor(language)
            result = await executor.execute(code, {"inputs": inputs, **context})

            # Validate outputs
            for key, spec in workflow.schema_data.get("outputs", {}).items():
                if key not in result and spec.get("constraints", {}).get("required", False):
                    logger.error(f"Missing required output: {key}")
                    raise ExecutorError(f"Missing required output: {key}")

            logger.info(f"Workflow {workflow.function} executed successfully")
            return result

        except WorkflowValidationError as e:
            logger.error(f"Workflow validation failed: {str(e)}")
            raise ExecutorError(f"Invalid workflow: {str(e)}")
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}", exc_info=True)
            raise ExecutorError(f"Workflow execution failed: {str(e)}")
