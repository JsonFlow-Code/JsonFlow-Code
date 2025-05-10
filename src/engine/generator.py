from typing import Dict, Any
from engine.workflow import Workflow  # Assuming Workflow is defined here

# Global registry for generators
generators: Dict[str, Any] = {}

def register_generator(language: str, generator_class: type) -> None:
    """
    Register a generator class for a specific language.

    Args:
        language: The language identifier (e.g., 'react', 'python').
        generator_class: The generator class to register.
    """
    generators[language] = generator_class

def get_generator(language: str) -> 'LanguageGenerator':
    """
    Retrieve a generator instance for the specified language.

    Args:
        language: The language identifier.

    Returns:
        LanguageGenerator: An instance of the registered generator.

    Raises:
        ValueError: If the language is not supported.
    """
    if language not in generators:
        raise ValueError(f"Unsupported language: {language}")
    return generators[language]()

class GeneratorError(Exception):
    """Exception raised for errors in the code generation process."""
    pass

class LanguageGenerator:
    """
    Base class for language-specific code generators.
    """
    def generate(self, workflow: Workflow) -> str:
        """
        Generate code for a given workflow.

        Args:
            workflow: The Workflow object containing function and steps.

        Returns:
            str: Generated code as a string.
        """
        code = [f"# Generated code for {workflow.function}"]
        for step in workflow.steps:
            method = getattr(self, f"generate_{step['type']}", self.generate_default)
            code.append(method(step))
        return "\n".join(code)
    
    def generate_default(self, step: Dict[str, Any]) -> str:
        """
        Fallback method for unsupported step types.

        Args:
            step: The step dictionary.

        Returns:
            str: Placeholder code for unsupported steps.
        """
        return f"# Unsupported step type: {step['type']}"
    
    def generate_set(self, step: Dict[str, Any]) -> str:
        """
        Generate code for a 'set' step.

        Args:
            step: The step dictionary with target and value.

        Returns:
            str: Code to set a variable.
        """
        return f"{step['target']} = {step['value']}"
    
    def generate_if(self, step: Dict[str, Any]) -> str:
        """
        Generate code for an 'if' step.

        Args:
            step: The step dictionary with condition and then branches.

        Returns:
            str: Code for the if statement.
        """
        condition = step['condition']
        then_code = "\n    ".join(self.generate_step(s) for s in step['then'])
        return f"if {condition}:\n    {then_code}"
    
    def generate_step(self, step: Dict[str, Any]) -> str:
        """
        Generate code for a single step by dispatching to the appropriate method.

        Args:
            step: The step dictionary.

        Returns:
            str: Generated code for the step.
        """
        method = getattr(self, f"generate_{step['type']}", self.generate_default)
        return method(step)