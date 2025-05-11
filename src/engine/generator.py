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
from typing import Dict, Any, Type, List, Optional
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from prometheus_client import Counter, Gauge
from engine.workflow import Workflow, WorkflowValidationError
from jinja2 import Environment, BaseLoader

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
resource_usage = Gauge('resource_usage', 'Resource usage during generation', ['language', 'resource_type'])

class GeneratorError(Exception):
    """Exception raised for errors in the code generation process."""
    pass

class TCCLogger:
    """Traceable logging for plugin operations."""
    def __init__(self):
        self.tcc_log: List[Dict[str, Any]] = []
        self.step_counter: int = 0

    def log(self, operation: str, input_data: bytes, output_data: bytes, metadata: Dict[str, Any] = None, log_level: str = "INFO", error_code: str = "NONE") -> None:
        entry = {
            "step": self.step_counter,
            "operation": operation,
            "input_data": base64.b64encode(input_data).decode('utf-8'),
            "output_data": base64.b64encode(output_data).decode('utf-8'),
            "metadata": metadata or {},
            "log_level": log_level,
            "error_code": error_code,
            "prev_hash": self._compute_prev_hash(),
            "operation_id": hashlib.sha256(f"{self.step_counter}:{operation}:{time.time_ns()}".encode()).hexdigest()[:32],
            "timestamp": time.time_ns(),
            "execution_time_ns": 0
        }
        self.tcc_log.append(entry)
        self.step_counter += 1
        logger.info(f"TCC Log: {operation} - Step {entry['step']} - ID {entry['operation_id']}")

    def _compute_prev_hash(self) -> str:
        if not self.tcc_log:
            return base64.b64encode(b'\x00' * 32).decode('utf-8')
        last_entry = self.tcc_log[-1]
        return base64.b64encode(hashlib.sha256(json.dumps(last_entry).encode()).digest()).decode('utf-8')

    def save_log(self, filename: str) -> None:
        with open(filename, 'w') as f:
            for entry in self.tcc_log:
                f.write(json.dumps(entry) + '\n')

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
        self.logger = TCCLogger()

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
                        self.logger.log(
                            "plugin_load",
                            module_name.encode('utf-8'),
                            plugin_name.encode('utf-8'),
                            {"metadata": metadata}
                        )
                        logger.info(f"Loaded plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Failed to load plugin {module_name}: {str(e)}")
                    self.logger.log(
                        "plugin_load_error",
                        module_name.encode('utf-8'),
                        str(e).encode('utf-8'),
                        {"error": str(e)},
                        "ERROR",
                        "PLUGIN_LOAD_FAILED"
                    )
        except ImportError as e:
            logger.error(f"Failed to import plugin package {plugin_dir}: {str(e)}")

    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """
        Retrieve a plugin by name.

        Args:
            plugin_name: Name of the plugin.

        Returns:
            Optional[PluginInterface]: The plugin instance or None if not found.
        """
        return self.plugins.get(plugin_name)

    def get_supported_steps(self) -> Dict[str, PluginInterface]:
        """
        Get all supported step types and their corresponding plugins.

        Returns:
            Dict[str, PluginInterface]: Mapping of step types to plugins.
        """
        supported_steps = {}
        for plugin in self.plugins.values():
            for step_type in plugin.get_step_types():
                supported_steps[step_type] = plugin
        return supported_steps

@contextmanager
def generation_context(language: str, description: str):
    """Context manager for code generation."""
    logger.info(f"Starting generation: {description} for {language}")
    generator_executions.labels(language=language).inc()
    try:
        yield
    except Exception as e:
        generation_errors.labels(language=language).inc()
        logger.error(f"Generation failed: {description} - {str(e)}")
        raise GeneratorError(f"Generation failed: {str(e)}")
    finally:
        logger.info(f"Completed generation: {description}")
        resource_usage.labels(language=language, resource_type="memory").set(sys.getsizeof({}))

class LanguageGenerator(ABC):
    """
    Base class for language-specific code generators.
    """
    def __init__(self):
        self.jinja_env = Environment(loader=BaseLoader())
        self.logger = TCCLogger()
        self.plugin_manager = PluginManager()
        self.plugin_manager.load_plugins()
        self.supported_steps = {
            "set", "if", "return", "call", "try", "while", "foreach", "parallel",
            "assert", "event", "require_role", "ai_infer", "ai_train", "ai_classify",
            "ai_embed", "ai_explain", "quantum_circuit", "quantum_measure",
            "quantum_algorithm", "blockchain_operation", "crypto_sign",
            "crypto_verify", "regex_match", "audit_log", "call_workflow",
            "game_render", "game_physics", "game_multiplayer_sync", "game_input",
            "game_animation", "script"
        }
        self.plugin_steps = self.plugin_manager.get_supported_steps()

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
        with generation_context(workflow.metadata.get("target_language", "unknown"), f"Workflow {workflow.function}"):
            try:
                # Validate workflow
                self._validate_workflow(workflow)

                code = [self._generate_header(workflow)]
                code.extend(self._generate_imports())
                code.append(self._generate_function_signature(workflow))

                for step in workflow.steps:
                    code.append(f"    # Step: {step['id']}")
                    generated_step = self.generate_step(step)
                    code.append(f"    {generated_step}")
                    self.logger.log(
                        "generate_step",
                        json.dumps(step).encode('utf-8'),
                        generated_step.encode('utf-8'),
                        {"step_id": step['id'], "step_type": step['type']}
                    )

                code.append(self._generate_function_footer())
                generated_code = "\n".join(code)

                # Update metrics
                resource_usage.labels(
                    language=workflow.metadata.get("target_language", "unknown"),
                    resource_type="memory"
                ).set(sys.getsizeof(generated_code))

                self.logger.log(
                    "generate_workflow",
                    json.dumps(workflow.metadata).encode('utf-8'),
                    generated_code.encode('utf-8'),
                    {"function": workflow.function}
                )

                return generated_code

            except Exception as e:
                self.logger.log(
                    "generate_error",
                    json.dumps(workflow.metadata).encode('utf-8'),
                    str(e).encode('utf-8'),
                    {"error": str(e)},
                    "ERROR",
                    "GENERATION_FAILED"
                )
                raise GeneratorError(f"Code generation failed: {str(e)}")

    def _validate_workflow(self, workflow: Workflow) -> None:
        """
        Validate the workflow structure and metadata.

        Args:
            workflow: The Workflow object to validate.

        Raises:
            WorkflowValidationError: If validation fails.
        """
        if not workflow.function:
            raise WorkflowValidationError("Workflow function name is required")
        if not workflow.steps:
            raise WorkflowValidationError("Workflow must contain at least one step")
        for step in workflow.steps:
            if "type" not in step or "id" not in step:
                raise WorkflowValidationError(f"Step must have 'type' and 'id': {step}")

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
            GeneratorError: If step type is unsupported.
        """
        step_type = step['type']
        if step_type.startswith("custom_"):
            return self.generate_custom_step(step)
        if step_type in self.plugin_steps:
            plugin = self.plugin_steps[step_type]
            return plugin.generate_step(step, workflow.metadata.get("target_language", "unknown"))
        method = getattr(self, f"generate_{step_type}", self.generate_default)
        return method(step)

    def generate_default(self, step: Dict[str, Any]) -> str:
        """
        Fallback method for unsupported step types.

        Args:
            step: The step dictionary.

        Returns:
            str: Placeholder code for unsupported steps.
        """
        return f"# Unsupported step type: {step['type']} # ID: {step['id']}"

    def generate_custom_step(self, step: Dict[str, Any]) -> str:
        """
        Generate code for custom steps.

        Args:
            step: The step dictionary.

        Returns:
            str: Code for custom step.
        """
        props = step.get("custom_properties", {})
        return f"# Custom step: {step['type']} with properties {json.dumps(props, indent=2)}"

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

    @abstractmethod
    def generate_ai_infer(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'ai_infer' step."""
        pass

    @abstractmethod
    def generate_ai_train(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'ai_train' step."""
        pass

    @abstractmethod
    def generate_ai_classify(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'ai_classify' step."""
        pass

    @abstractmethod
    def generate_ai_embed(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'ai_embed' step."""
        pass

    @abstractmethod
    def generate_ai_explain(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'ai_explain' step."""
        pass

    @abstractmethod
    def generate_quantum_circuit(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'quantum_circuit' step."""
        pass

    @abstractmethod
    def generate_quantum_measure(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'quantum_measure' step."""
        pass

    @abstractmethod
    def generate_quantum_algorithm(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'quantum_algorithm' step."""
        pass

    @abstractmethod
    def generate_blockchain_operation(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'blockchain_operation' step."""
        pass

    @abstractmethod
    def generate_crypto_sign(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'crypto_sign' step."""
        pass

    @abstractmethod
    def generate_crypto_verify(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'crypto_verify' step."""
        pass

    @abstractmethod
    def generate_regex_match(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'regex_match' step."""
        pass

    @abstractmethod
    def generate_audit_log(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'audit_log' step."""
        pass

    @abstractmethod
    def generate_call_workflow(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'call_workflow' step."""
        pass

    @abstractmethod
    def generate_game_render(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'game_render' step."""
        pass

    @abstractmethod
    def generate_game_physics(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'game_physics' step."""
        pass

    @abstractmethod
    def generate_game_multiplayer_sync(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'game_multiplayer_sync' step."""
        pass

    @abstractmethod
    def generate_game_input(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'game_input' step."""
        pass

    @abstractmethod
    def generate_game_animation(self, step: Dict[str, Any]) -> str:
        """Generate code for an 'game_animation' step."""
        pass

    @abstractmethod
    def generate_script(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'script' step."""
        pass
