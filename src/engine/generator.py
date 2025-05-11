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
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram
from engine.workflow import Workflow, WorkflowValidationError
from engine.validator import validate_workflow
from languages import registry

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
generator_executions = Counter('generator_executions_total', 'Total code generations', ['language'])
generation_errors = Counter('generation_errors_total', 'Total generation errors', ['language'])
generation_duration = Histogram('generation_duration_seconds', 'Code generation duration', ['language'])
resource_usage = Gauge('resource_usage', 'Resource usage during generation', ['language', 'resource_type'])

class GeneratorError(Exception):
    """Exception raised for errors in the code generation process."""
    pass

class PluginInterface(ABC):
    """Interface for generator plugins."""
    @abstractmethod
    def get_step_types(self) -> List[str]:
        """Return the step types supported by the plugin."""
        pass

    @abstractmethod
    def generate_step(self, step: Dict[str, Any], language: str) -> str:
        """Generate code for a specific step."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return plugin metadata (name, version, etc.)."""
        pass

class PluginManager:
    """Manages plugin loading and registration."""
    def __init__(self):
        self.plugins: Dict[str, PluginInterface] = {}

    def load_plugins(self, plugin_dir: str = "languages.plugins") -> None:
        """
        Dynamically load plugins from the specified package.

        Args:
            plugin_dir: Package path for plugins (e.g., 'languages.plugins').
        """
        try:
            package = importlib.import_module(plugin_dir)
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{plugin_dir}.{module_name}")
                    if hasattr(module, "Plugin"):
                        plugin = module.Plugin()
                        metadata = plugin.get_metadata()
                        plugin_name = metadata.get("name", module_name)
                        self.plugins[plugin_name] = plugin
                        logger.info(f"Loaded plugin: {plugin_name} (version: {metadata.get('version', 'unknown')})")
                except Exception as e:
                    logger.error(f"Failed to load plugin {module_name}: {str(e)}", exc_info=True)
        except ImportError as e:
            logger.error(f"Failed to import plugin package {plugin_dir}: {str(e)}", exc_info=True)

    def get_supported_steps(self) -> Dict[str, PluginInterface]:
        """
        Get all supported step types and their corresponding plugins.

        Returns:
            Dict[str, PluginInterface]: Mapping of step types to plugins.
        """
        supported_steps = {}
        for plugin in self.plugins.values():
            for step_type in plugin.get_step_types():
                if step_type in supported_steps:
                    logger.warning(f"Duplicate step type {step_type} registered by plugin {plugin.get_metadata()['name']}")
                supported_steps[step_type] = plugin
        return supported_steps

@contextmanager
def generation_context(language: str, description: str):
    """
    Context manager for code generation with metrics and logging.

    Args:
        language: Target language for generation.
        description: Description of the generation task.
    """
    logger.info(f"Starting generation: {description} for {language}")
    generator_executions.labels(language=language).inc()
    start_time = datetime.now()
    try:
        yield
    except Exception as e:
        generation_errors.labels(language=language).inc()
        logger.error(f"Generation failed: {description} - {str(e)}", exc_info=True)
        raise
    finally:
        duration = (datetime.now() - start_time).total_seconds()
        generation_duration.labels(language=language).observe(duration)
        logger.info(f"Completed generation: {description} in {duration:.2f}s")
        resource_usage.labels(language=language, resource_type="memory").set(sys.getsizeof({}))

class LanguageGenerator(ABC):
    """
    Base class for language-specific code generators.
    """
    def __init__(self):
        self.plugin_manager = PluginManager()
        self.plugin_manager.load_plugins()
        self.plugin_steps = self.plugin_manager.get_supported_steps()
        self.supported_steps = {
            "set", "if", "return", "call", "try", "while", "foreach", "parallel",
            "assert", "event", "require_role"
        }

    def generate(self, workflow: Workflow) -> str:
        """
        Generate code for a given workflow.

        Args:
            workflow: The Workflow object containing function and steps.

        Returns:
            str: Generated code as a string.

        Raises:
            GeneratorError: If code generation fails.
        """
        language = workflow.metadata.get("target_language", "unknown")
        with generation_context(language, f"Workflow {workflow.function}"):
            try:
                # Validate workflow
                validate_workflow(workflow)

                # Initialize code output
                code = [self._generate_header(workflow)]
                code.extend(self._generate_imports())
                code.append(self._generate_function_signature(workflow))

                # Generate code for each step
                for step in workflow.steps:
                    step_id = step.get("id", "unknown")
                    code.append(f"    # Step: {step_id}")
                    generated_step = self.generate_step(step)
                    code.append(f"    {generated_step}")

                code.append(self._generate_function_footer())
                generated_code = "\n".join(code)

                # Update resource usage metrics
                resource_usage.labels(language=language, resource_type="memory").set(sys.getsizeof(generated_code))

                logger.info(f"Generated code for workflow {workflow.function} ({len(code)} lines)")
                return generated_code

            except WorkflowValidationError as e:
                logger.error(f"Workflow validation failed: {str(e)}")
                raise GeneratorError(f"Invalid workflow: {str(e)}")
            except Exception as e:
                logger.error(f"Code generation failed: {str(e)}", exc_info=True)
                raise GeneratorError(f"Code generation failed: {str(e)}")

    def _generate_header(self, workflow: Workflow) -> str:
        """
        Generate header comments with metadata.

        Args:
            workflow: The Workflow object.

        Returns:
            str: Header code with metadata.
        """
        metadata = workflow.metadata
        return f"""
# Generated code for {workflow.function}
# Version: {metadata.get('version', '1.0.0')}
# Author: {metadata.get('author', 'Unknown')}
# Created: {metadata.get('created', datetime.now().isoformat())}
# Description: {metadata.get('description', 'Generated workflow code')}
# Language: {metadata.get('target_language', 'unknown')}
"""

    def _generate_imports(self) -> List[str]:
        """
        Generate import statements for the language.

        Returns:
            List[str]: List of import statements.
        """
        return []

    def _generate_function_signature(self, workflow: Workflow) -> str:
        """
        Generate the function signature for the workflow.

        Args:
            workflow: The Workflow object.

        Returns:
            str: Function signature code.
        """
        return f"# Workflow function: {workflow.function}"

    def _generate_function_footer(self) -> str:
        """
        Generate the function footer (e.g., return statement).

        Returns:
            str: Function footer code.
        """
        return ""

    def generate_step(self, step: Dict[str, Any]) -> str:
        """
        Generate code for a single step by dispatching to the appropriate method or plugin.

        Args:
            step: The step dictionary.

        Returns:
            str: Generated code for the step.

        Raises:
            GeneratorError: If step type is unsupported or generation fails.
        """
        step_type = step.get('type', '')
        step_id = step.get('id', 'unknown')

        if not step_type:
            logger.error(f"Step {step_id} missing type")
            raise GeneratorError(f"Step {step_id} missing type")

        if step_type in self.plugin_steps:
            plugin = self.plugin_steps[step_type]
            try:
                return plugin.generate_step(step, workflow.metadata.get("target_language", "unknown"))
            except Exception as e:
                logger.error(f"Plugin {plugin.get_metadata()['name']} failed for step {step_id}: {str(e)}")
                raise GeneratorError(f"Plugin failed for step {step_type}: {str(e)}")

        if step_type in self.supported_steps:
            method = getattr(self, f"generate_{step_type}", self.generate_default)
            try:
                return method(step)
            except Exception as e:
                logger.error(f"Failed to generate step {step_id} of type {step_type}: {str(e)}")
                raise GeneratorError(f"Failed to generate step {step_type}: {str(e)}")

        logger.warning(f"Unsupported step type {step_type} for step {step_id}")
        return self.generate_default(step)

    def generate_default(self, step: Dict[str, Any]) -> str:
        """
        Fallback method for unsupported step types.

        Args:
            step: The step dictionary.

        Returns:
            str: Placeholder code for unsupported steps.
        """
        step_type = step.get('type', 'unknown')
        step_id = step.get('id', 'unknown')
        return f"# Unsupported step type: {step_type} # ID: {step_id}"

    @abstractmethod
    def generate_set(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'set' step."""
        pass

    @abstractmethod
    def generate_if(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'if' step."""
        pass

    @abstractmethod
    def generate_return(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'return' step."""
        pass

    @abstractmethod
    def generate_call(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'call' step."""
        pass

    @abstractmethod
    def generate_try(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'try' step."""
        pass

    @abstractmethod
    def generate_while(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'while' step."""
        pass

    @abstractmethod
    def generate_foreach(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'foreach' step."""
        pass

    @abstractmethod
    def generate_parallel(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'parallel' step."""
        pass

    @abstractmethod
    def generate_assert(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'assert' step."""
        pass

    @abstractmethod
    def generate_event(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'event' step."""
        pass

    @abstractmethod
    def generate_require_role(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'require_role' step."""
        pass

def generate_workflow(workflow: Workflow) -> str:
    """
    Generate code for a workflow using the appropriate language generator.

    Args:
        workflow: The Workflow object.

    Returns:
        str: Generated code.

    Raises:
        GeneratorError: If generation fails.
    """
    language = workflow.metadata.get("target_language", "python")
    try:
        generator = registry.get_generator(language)
        return generator.generate(workflow)
    except ValueError as e:
        logger.error(f"Unsupported language: {language}")
        raise GeneratorError(f"Unsupported language: {language}")
    except Exception as e:
        logger.error(f"Workflow generation failed: {str(e)}", exc_info=True)
        raise GeneratorError(f"Workflow generation failed: {str(e)}")
