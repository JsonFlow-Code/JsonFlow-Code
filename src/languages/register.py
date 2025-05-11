from typing import Dict, Type, Any
from .python import PythonGenerator, PythonExecutor
from .cpp import CppGenerator, CppExecutor
from .go import GoGenerator, GoExecutor
from .javascript import JavaScriptGenerator, JavaScriptExecutor
from .julia import JuliaGenerator, JuliaExecutor
from .perl import PerlGenerator, PerlExecutor
from .react import ReactGenerator, ReactExecutor
from .rust import RustGenerator, RustExecutor
from .solidity import SolidityGenerator, SolidityExecutor
from .typescript import TypeScriptGenerator, TypeScriptExecutor
from .qiskitGenerator import QiskitGenerator
from .mermaid import MermaidGenerator
from .natural import NaturalGenerator

generators: Dict[str, Type[Any]] = {}
executors: Dict[str, Type[Any]] = {}

def register_generator(language: str, generator_class: Type[Any]) -> None:
    """
    Register a generator class for a specific language.

    Args:
        language: The language identifier.
        generator_class: The generator class to register.

    Raises:
        ValueError: If the language is already registered.
    """
    if language in generators:
        raise ValueError(f"Generator for {language} already registered")
    generators[language] = generator_class

def register_executor(language: str, executor_class: Type[Any]) -> None:
    """
    Register an executor class for a specific language.

    Args:
        language: The language identifier.
        executor_class: The executor class to register.

    Raises:
        ValueError: If the language is already registered.
    """
    if language in executors:
        raise ValueError(f"Executor for {language} already registered")
    executors[language] = executor_class

def get_generator(language: str) -> Any:
    """
    Retrieve a generator instance for the specified language.

    Args:
        language: The language identifier.

    Returns:
        Any: An instance of the registered generator.

    Raises:
        ValueError: If the language is not supported.
    """
    if language not in generators:
        raise ValueError(f"Unsupported language: {language}")
    return generators[language]()

def get_executor(language: str) -> Any:
    """
    Retrieve an executor instance for the specified language.

    Args:
        language: The language identifier.

    Returns:
        Any: An instance of the registered executor.

    Raises:
        ValueError: If the language is not supported.
    """
    if language not in executors:
        raise ValueError(f"Unsupported language: {language}")
    return executors[language]()

# Register generators and executors
register_generator("python", PythonGenerator)
register_executor("python", PythonExecutor)
register_generator("cpp", CppGenerator)
register_executor("cpp", CppExecutor)
register_generator("go", GoGenerator)
register_executor("go", GoExecutor)
register_generator("javascript", JavaScriptGenerator)
register_executor("javascript", JavaScriptExecutor)
register_generator("julia", JuliaGenerator)
register_executor("julia", JuliaExecutor)
register_generator("perl", PerlGenerator)
register_executor("perl", PerlExecutor)
register_generator("react", ReactGenerator)
register_executor("react", ReactExecutor)
register_generator("rust", RustGenerator)
register_executor("rust", RustExecutor)
register_generator("solidity", SolidityGenerator)
register_executor("solidity", SolidityExecutor)
register_generator("typescript", TypeScriptGenerator)
register_executor("typescript", TypeScriptExecutor)
register_generator("qiskit", QiskitGenerator)
register_generator("mermaid", MermaidGenerator)
register_generator("natural", NaturalGenerator)
