from typing import Any, Dict
from abc import ABC, abstractmethod

class Executor(ABC):
    """Abstract base class for code executors."""
    @abstractmethod
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        pass

class PythonExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        local_context = context.copy()
        exec(code, local_context)
        return local_context.get('result', local_context)

class CppExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "cpp_executed"}

class GoExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "go_executed"}

class JavaScriptExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "javascript_executed"}

class SolidityExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "solidity_executed"}

class RustExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "rust_executed"}

class TypeScriptExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "typescript_executed"}

class ReactExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "react_executed"}

class PerlExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "perl_executed"}

class JuliaExecutor(Executor):
    def execute(self, code: str, context: Dict[str, Any]) -> Any:
        return {"status": "julia_executed"}

def get_executor(language: str) -> Executor:
    """Get the appropriate executor for the language."""
    executors = {
        "python": PythonExecutor,
        "cpp": CppExecutor,
        "go": GoExecutor,
        "javascript": JavaScriptExecutor,
        "solidity": SolidityExecutor,
        "rust": RustExecutor,
        "typescript": TypeScriptExecutor,
        "react": ReactExecutor,
        "perl": PerlExecutor,
        "julia": JuliaExecutor
    }
    executor_cls = executors.get(language)
    if not executor_cls:
        raise ValueError(f"Unsupported language: {language}")
    return executor_cls()
