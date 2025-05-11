import logging
import sys
import json
from typing import Dict, Any, Type, List
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
        self.supported_steps = {
            "set", "if", "return", "call", "try", "while", "foreach", "parallel",
            "assert", "event", "require_role", "ai_infer", "ai_train", "ai_classify",
            "ai_embed", "ai_explain", "quantum_circuit", "quantum_measure",
            "quantum_algorithm", "blockchain_operation", "crypto_sign",
            "crypto_verify", "regex_match", "audit_log", "call_workflow",
            "game_render", "game_physics", "game_multiplayer_sync", "game_input",
            "game_animation", "script"
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
        with generation_context(workflow.metadata.get("target_language", "unknown"), f"Workflow {workflow.function}"):
            try:
                # Validate workflow
                self._validate_workflow(workflow)

                code = [self._generate_header(workflow)]
                code.extend(self._generate_imports())
                code.append(self._generate_function_signature(workflow))

                for step in workflow.steps:
                    code.append(f"    # Step: {step['id']}")
                    code.append(f"    {self.generate_step(step)}")

                code.append(self._generate_function_footer())
                generated_code = "\n".join(code)

                # Update metrics
                resource_usage.labels(
                    language=workflow.metadata.get("target_language", "unknown"),
                    resource_type="memory"
                ).set(sys.getsizeof(generated_code))

                return generated_code

            except Exception as e:
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
        Generate code for a single step by dispatching to the appropriate method.

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
        """Generate code for a 'game_animation' step."""
        pass

    @abstractmethod
    def generate_script(self, step: Dict[str, Any]) -> str:
        """Generate code for a 'script' step."""
        pass
