import json
import logging
import time
from typing import Any, Dict, List, Optional
from jsonschema import validate, ValidationError
from restrictedpython import compile_restricted, safe_globals, utility_builtins
import asyncio
import platform
import resource
import signal
import sys
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds: int):
    """Context manager to enforce execution time limit."""
    def signal_handler(signum, frame):
        raise TimeoutException(f"Execution timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

@contextmanager
def memory_limit(kb: int):
    """Context manager to enforce memory limit in KB."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (kb * 1024, kb * 1024))
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

class JSONFlowParser:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.context: Dict[str, Any] = {}
        self.logger = logger

    def validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """Validate workflow against the schema."""
        try:
            validate(instance=workflow, schema=self.schema)
            self.logger.info("Workflow validation successful")
        except ValidationError as e:
            self.logger.error(f"Workflow validation failed: {str(e)}")
            raise

    def evaluate_expr(self, expr: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Evaluate an expression in the given context."""
        if "get" in expr:
            keys = expr["get"].split(".")
            value = context
            for key in keys:
                value = value.get(key)
                if value is None:
                    raise ValueError(f"Invalid context key: {expr['get']}")
            return value
        elif "value" in expr:
            return expr["value"]
        elif "add" in expr:
            return sum(self.evaluate_expr(item, context) for item in expr["add"])
        elif "subtract" in expr:
            result = self.evaluate_expr(expr["subtract"][0], context)
            for item in expr["subtract"][1:]:
                result -= self.evaluate_expr(item, context)
            return result
        elif "multiply" in expr:
            result = 1
            for item in expr["multiply"]:
                result *= self.evaluate_expr(item, context)
            return result
        elif "divide" in expr:
            result = self.evaluate_expr(expr["divide"][0], context)
            for item in expr["divide"][1:]:
                value = self.evaluate_expr(item, context)
                if value == 0:
                    raise ValueError("Division by zero")
                result /= value
            return result
        elif "compare" in expr:
            left = self.evaluate_expr(expr["compare"]["left"], context)
            right = self.evaluate_expr(expr["compare"]["right"], context)
            op = expr["compare"]["op"]
            if op == "===":
                return left == right
            elif op == "<":
                return left < right
            elif op == ">":
                return left > right
            elif op == "<=":
                return left <= right
            elif op == ">=":
                return left >= right
            elif op == "!==":
                return left != right
        elif "not" in expr:
            return not self.evaluate_expr(expr["not"], context)
        elif "and" in expr:
            return all(self.evaluate_expr(item, context) for item in expr["and"])
        elif "or" in expr:
            return any(self.evaluate_expr(item, context) for item in expr["or"])
        elif "concat" in expr:
            return "".join(str(self.evaluate_expr(item, context)) for item in expr["concat"])
        else:
            raise ValueError(f"Unsupported expression: {expr}")

    def execute_script(self, script: str, inputs: Dict[str, Any], sandbox: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a Python script in a sandboxed environment."""
        try:
            # Prepare sandbox globals
            sandbox_globals = safe_globals.copy()
            sandbox_globals.update(utility_builtins)
            
            # Restrict allowed modules
            allowed_modules = sandbox.get("allowed_modules", [])
            sandbox_globals["__builtins__"] = {
                k: v for k, v in __builtins__.__dict__.items()
                if k in ["print", "len", "range", "int", "str", "float", "bool", "list", "dict", "set"]
            }
            
            # Import allowed modules
            for module in allowed_modules:
                if module == "math":
                    import math
                    sandbox_globals["math"] = math
                elif module == "random":
                    import random
                    sandbox_globals["random"] = random
                elif module == "datetime":
                    import datetime
                    sandbox_globals["datetime"] = datetime
                elif module == "json":
                    import json
                    sandbox_globals["json"] = json
                elif module == "pygame" and platform.system() == "Emscripten":
                    import pygame
                    sandbox_globals["pygame"] = pygame
                elif module == "numpy":
                    import numpy
                    sandbox_globals["numpy"] = numpy
                else:
                    raise ValueError(f"Unsupported module: {module}")

            # Prepare script locals with evaluated inputs
            script_locals = {}
            for key, expr in inputs.items():
                script_locals[key] = self.evaluate_expr(expr, context)

            # Compile and execute script with restrictions
            max_execution_time = sandbox.get("max_execution_time", 5)
            max_memory = sandbox.get("max_memory", 10240)  # Default 10MB
            code = compile_restricted(script, "<inline>", "exec")
            
            with time_limit(max_execution_time), memory_limit(max_memory):
                exec(code, sandbox_globals, script_locals)
            
            # Return the result (assuming script sets a 'result' variable)
            return script_locals.get("result")
        
        except TimeoutException as e:
            self.logger.error(f"Script execution timed out: {str(e)}")
            raise
        except MemoryError:
            self.logger.error("Script exceeded memory limit")
            raise
        except Exception as e:
            self.logger.error(f"Script execution failed: {str(e)}")
            raise

    async def evaluate_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single step in the workflow."""
        try:
            step_type = step["type"]
            self.logger.info(f"Evaluating step: {step_type} (id: {step.get('id', 'unknown')})")

            # Handle timeout
            timeout = step.get("timeout", {})
            duration = timeout.get("duration", "30s")
            max_retries = timeout.get("max_retries", 0)
            action = timeout.get("action", "fail")
            retries = 0

            while retries <= max_retries:
                try:
                    if step_type == "set":
                        value = self.evaluate_expr(step["value"], context)
                        context[step["target"]] = value
                        return None
                    elif step_type == "if":
                        condition = self.evaluate_expr(step["condition"], context)
                        steps = step["then"] if condition else step.get("else", [])
                        for sub_step in steps:
                            result = await self.evaluate_step(sub_step, context)
                            if result is not None:
                                return result
                        return None
                    elif step_type == "return":
                        return {"value": self.evaluate_expr(step["value"], context)}
                    elif step_type == "call":
                        # Placeholder for function call
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "try":
                        try:
                            for sub_step in step["body"]:
                                result = await self.evaluate_step(sub_step, context)
                                if result is not None:
                                    return result
                        except Exception as e:
                            if "catch" in step:
                                catch = step["catch"]
                                context[catch.get("error_var", "error")] = str(e)
                                for sub_step in catch["body"]:
                                    result = await self.evaluate_step(sub_step, context)
                                    if result is not None:
                                        return result
                        finally:
                            if "finally" in step:
                                for sub_step in step["finally"]:
                                    result = await self.evaluate_step(sub_step, context)
                                    if result is not None:
                                        return result
                        return None
                    elif step_type == "while":
                        while self.evaluate_expr(step["condition"], context):
                            for sub_step in step["body"]:
                                result = await self.evaluate_step(sub_step, context)
                                if result is not None:
                                    return result
                        return None
                    elif step_type == "foreach":
                        collection = self.evaluate_expr(step["collection"], context)
                        for item in collection:
                            context[step["iterator"]] = item
                            for sub_step in step["body"]:
                                result = await self.evaluate_step(sub_step, context)
                                if result is not None:
                                    return result
                        return None
                    elif step_type == "parallel":
                        # Placeholder for parallel execution
                        for branch in step["branches"]:
                            for sub_step in branch:
                                await self.evaluate_step(sub_step, context)
                        return None
                    elif step_type == "assert":
                        if not self.evaluate_expr(step["condition"], context):
                            raise AssertionError(step.get("message", "Assertion failed"))
                        return None
                    elif step_type == "event":
                        self.logger.info(f"Emitting event: {step['name']} with params: {step['params']}")
                        return None
                    elif step_type == "require_role":
                        # Placeholder for role check
                        return None
                    elif step_type in ["ai_infer", "ai_train", "ai_classify", "ai_embed", "ai_explain"]:
                        # Placeholder for AI operations
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "quantum_circuit":
                        # Placeholder for quantum circuit
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "quantum_measure":
                        # Placeholder for quantum measurement
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "quantum_algorithm":
                        # Placeholder for quantum algorithm
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "blockchain_operation":
                        # Placeholder for blockchain operation
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "crypto_sign":
                        # Placeholder for crypto sign
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "crypto_verify":
                        # Placeholder for crypto verify
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "regex_match":
                        # Placeholder for regex match
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type == "audit_log":
                        self.logger.info(f"Audit log: {step['message']}")
                        return None
                    elif step_type == "call_workflow":
                        # Placeholder for subworkflow
                        context[step["target"]] = {}  # Mock result
                        return None
                    elif step_type.startswith("custom_"):
                        # Placeholder for custom step
                        return None
                    elif step_type == "game_render":
                        # Placeholder for game render
                        context[step["render_target"]] = {}  # Mock render output
                        return None
                    elif step_type == "game_physics":
                        # Placeholder for game physics
                        context[step["target"]] = {}  # Mock physics state
                        return None
                    elif step_type == "game_multiplayer_sync":
                        # Placeholder for multiplayer sync
                        context[step["target"]] = {}  # Mock sync result
                        return None
                    elif step_type == "game_input":
                        # Placeholder for game input
                        context[step["target"]] = {}  # Mock input data
                        return None
                    elif step_type == "game_animation":
                        # Placeholder for game animation
                        context[step["target"]] = {}  # Mock animation result
                        return None
                    elif step_type == "script":
                        inputs = step.get("inputs", {})
                        sandbox = step.get("sandbox", {})
                        result = self.execute_script(step["script"], inputs, sandbox, context)
                        context[step["target"]] = result
                        return None
                    else:
                        raise ValueError(f"Unknown step type: {step_type}")

                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        if action == "fail":
                            raise
                        elif action == "skip":
                            self.logger.warning(f"Skipping step {step_type} after {retries} retries: {str(e)}")
                            return None
                        elif action == "retry":
                            self.logger.warning(f"Retrying step {step_type} ({retries}/{max_retries}): {str(e)}")
                            continue
                    time.sleep(1)  # Backoff before retry

            # Handle on_error
            if "on_error" in step:
                error_handler = step["on_error"]
                if error_handler.get("step_id"):
                    # Placeholder for jumping to error step
                    pass
                elif error_handler.get("body"):
                    for sub_step in error_handler["body"]:
                        result = await self.evaluate_step(sub_step, context)
                        if result is not None:
                            return result
                return None

        except Exception as e:
            self.logger.error(f"Error evaluating step {step.get('id', 'unknown')}: {str(e)}")
            raise

    async def execute_workflow(self, workflow: Dict[str, Any], initial_context: Dict[str, Any] = None) -> Any:
        """Execute the entire workflow."""
        try:
            self.validate_workflow(workflow)
            self.context = initial_context or {}
            
            # Apply execution policy
            execution_policy = workflow.get("execution_policy", {})
            max_runs_per_minute = execution_policy.get("max_runs_per_minute", float("inf"))
            max_concurrent_runs = execution_policy.get("max_concurrent_runs", 1)
            
            # Placeholder for rate limiting and concurrency control
            self.logger.info(f"Executing workflow: {workflow['function']}")

            # Initialize context with schema context
            for key, value in workflow["schema"].get("context", {}).items():
                if value.get("source") in ["game_state", "player_input"]:
                    self.context[key] = {}  # Mock game-specific context
                else:
                    self.context[key] = None  # Placeholder for other sources

            # Execute steps
            for step in workflow["steps"]:
                result = await self.evaluate_step(step, self.context)
                if result is not None:
                    return result["value"]

            # Return outputs
            outputs = {}
            for key, spec in workflow["schema"]["outputs"].items():
                outputs[key] = self.context.get(key)
            return outputs

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            raise

# Example usage for Pyodide-compatible game script
async def main():
    schema = {...}  # Extended schema with script step
    parser = JSONFlowParser(schema)
    
    workflow = {
        "function": "gameWorkflow",
        "schema": {
            "inputs": {"player_input": {"type": "object", "source": "player_input"}},
            "context": {"game_state": {"type": "game_state", "source": "game_state"}},
            "outputs": {"updated_state": {"type": "game_state"}}
        },
        "metadata": {"schema_version": "1.1.0"},
        "steps": [
            {
                "type": "script",
                "id": "step1",
                "script": "result = {'position': player_input['position'], 'score': game_state['score'] + 1}",
                "target": "updated_state",
                "sandbox": {
                    "allowed_modules": ["math", "pygame"],
                    "max_execution_time": 5,
                    "max_memory": 10240
                },
                "inputs": {
                    "player_input": {"get": "player_input"},
                    "game_state": {"get": "game_state"}
                }
            }
        ]
    }
    
    context = {
        "player_input": {"position": [0, 0, 0]},
        "game_state": {"score": 100}
    }
    
    result = await parser.execute_workflow(workflow, context)
    print(f"Workflow result: {result}")

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
