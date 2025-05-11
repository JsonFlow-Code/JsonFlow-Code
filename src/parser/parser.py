import json
import logging
import time
import asyncio
import platform
import resource
import signal
import sys
import re
import hashlib
import aiohttp
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from jsonschema import validate, ValidationError
from restrictedpython import compile_restricted, safe_globals, utility_builtins
from contextlib import contextmanager, AsyncExitStack
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import pygame  # For game-related scripts
import qiskit  # For quantum operations
from eth_account import Account  # For blockchain operations
from web3 import Web3

# Setup logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

class MemoryLimitException(Exception):
    pass

class RateLimitExceeded(Exception):
    pass

@contextmanager
def time_limit(seconds: int):
    """Context manager for execution time limit."""
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
    """Context manager for memory limit in KB."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (kb * 1024, kb * 1024))
        yield
    except resource.error:
        raise MemoryLimitException(f"Failed to set memory limit to {kb} KB")
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

@dataclass
class ExecutionStats:
    """Track execution metrics for performance monitoring."""
    start_time: float
    end_time: float = None
    memory_used: int = 0
    cpu_time: float = 0
    steps_executed: int = 0
    errors: List[str] = None

    def __post_init__(self):
        self.errors = []

    def record_end(self):
        self.end_time = time.time()

    def duration(self) -> float:
        return (self.end_time or time.time()) - self.start_time

class JSONFlowParser:
    def __init__(self, schema: Dict[str, Any], max_concurrent: int = 10):
        self.schema = schema
        self.context: Dict[str, Any] = {}
        self.logger = logger
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.rate_limiter: Dict[str, List[float]] = {}
        self.session = None  # For aiohttp
        self.execution_stats = ExecutionStats(start_time=time.time())
        self.web3_providers: Dict[str, Web3] = {}  # Blockchain providers
        self._initialize_web3_providers()

    def _initialize_web3_providers(self):
        """Initialize Web3 providers for supported blockchains."""
        chains = ["ethereum", "solana", "starknet"]
        for chain in chains:
            # Placeholder for actual provider URLs (e.g., Infura, Alchemy)
            if chain == "ethereum":
                self.web3_providers[chain] = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"))
            # Add other chain providers as needed

    async def _init_session(self):
        """Initialize aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def __aenter__(self):
        await self._init_session()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)

    def _enforce_rate_limit(self, workflow: Dict[str, Any]):
        """Enforce execution rate limits."""
        policy = workflow.get("execution_policy", {})
        max_runs_per_minute = policy.get("max_runs_per_minute", 60)
        function = workflow["function"]
        
        now = time.time()
        self.rate_limiter.setdefault(function, [])
        self.rate_limiter[function] = [t for t in self.rate_limiter[function] if now - t < 60]
        
        if len(self.rate_limiter[function]) >= max_runs_per_minute:
            raise RateLimitExceeded(f"Rate limit exceeded for {function}: {max_runs_per_minute}/min")
        
        self.rate_limiter[function].append(now)

    def validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """Validate workflow against schema with detailed error reporting."""
        try:
            validate(instance=workflow, schema=self.schema)
            self.logger.info(f"Workflow {workflow['function']} validated successfully")
        except ValidationError as e:
            error_msg = f"Validation failed for {workflow['function']}: {e.message} at {e.json_path}"
            self.logger.error(error_msg)
            self.execution_stats.errors.append(error_msg)
            raise ValueError(error_msg)

    def evaluate_expr(self, expr: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Evaluate expressions securely with support for advanced operations."""
        try:
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
                if op == "===": return left == right
                elif op == "<": return left < right
                elif op == ">": return left > right
                elif op == "<=": return left <= right
                elif op == ">=": return left >= right
                elif op == "!==": return left != right
            elif "not" in expr:
                return not self.evaluate_expr(expr["not"], context)
            elif "and" in expr:
                return all(self.evaluate_expr(item, context) for item in expr["and"])
            elif "or" in expr:
                return any(self.evaluate_expr(item, context) for item in expr["or"])
            elif "concat" in expr:
                return "".join(str(self.evaluate_expr(item, context)) for item in expr["concat"])
            elif "hash" in expr:
                algorithm = expr["hash"]["algorithm"]
                input_data = str(self.evaluate_expr(expr["hash"]["input"], context)).encode()
                if algorithm == "sha256":
                    return hashlib.sha256(input_data).hexdigest()
                elif algorithm == "sha3":
                    return hashlib.sha3_256(input_data).hexdigest()
                elif algorithm == "keccak256":
                    return Web3.keccak(input_data).hex()
                elif algorithm == "blake2b":
                    return hashlib.blake2b(input_data).hexdigest()
            elif "regex" in expr:
                pattern = expr["regex"]["pattern"]
                input_data = str(self.evaluate_expr(expr["regex"]["input"], context))
                return bool(re.match(pattern, input_data))
            elif "map" in expr:
                collection = self.evaluate_expr(expr["map"]["collection"], context)
                operation = expr["map"]["operation"]
                return [self.evaluate_expr(operation, {**context, "item": item}) for item in collection]
            elif "filter" in expr:
                collection = self.evaluate_expr(expr["filter"]["collection"], context)
                condition = expr["filter"]["condition"]
                return [item for item in collection if self.evaluate_expr(condition, {**context, "item": item})]
            else:
                raise ValueError(f"Unsupported expression: {expr}")
        except Exception as e:
            self.logger.error(f"Expression evaluation failed: {str(e)}")
            self.execution_stats.errors.append(str(e))
            raise

    def execute_script(self, script: str, language: str, inputs: Dict[str, Any], sandbox: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute scripts in a sandboxed environment with multi-language support."""
        try:
            if language == "python":
                sandbox_globals = safe_globals.copy()
                sandbox_globals.update(utility_builtins)
                sandbox_globals["__builtins__"] = {
                    k: v for k, v in __builtins__.__dict__.items()
                    if k in ["print", "len", "range", "int", "str", "float", "bool", "list", "dict", "set", "tuple"]
                }

                allowed_modules = sandbox.get("allowed_modules", [])
                for module in allowed_modules:
                    if module == "math": sandbox_globals["math"] = __import__("math")
                    elif module == "random": sandbox_globals["random"] = __import__("random")
                    elif module == "datetime": sandbox_globals["datetime"] = __import__("datetime")
                    elif module == "json": sandbox_globals["json"] = __import__("json")
                    elif module == "pygame" and platform.system() == "Emscripten": sandbox_globals["pygame"] = pygame
                    elif module == "numpy": sandbox_globals["numpy"] = np
                    else: raise ValueError(f"Unsupported module: {module}")

                script_locals = {key: self.evaluate_expr(expr, context) for key, expr in inputs.items()}
                max_execution_time = sandbox.get("max_execution_time", 5)
                max_memory = sandbox.get("max_memory", 10240)

                code = compile_restricted(script, "<inline>", "exec")
                with time_limit(max_execution_time), memory_limit(max_memory):
                    exec(code, sandbox_globals, script_locals)
                return script_locals.get("result")
            
            elif language == "javascript":
                # Placeholder for JavaScript execution (e.g., via PyMiniRacer or similar)
                raise NotImplementedError("JavaScript execution not yet implemented")
            elif language == "lua":
                # Placeholder for Lua execution (e.g., via lupa)
                raise NotImplementedError("Lua execution not yet implemented")
            else:
                raise ValueError(f"Unsupported scripting language: {language}")
        except Exception as e:
            self.logger.error(f"Script execution failed: {str(e)}")
            self.execution_stats.errors.append(str(e))
            raise

    async def _execute_blockchain_operation(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute blockchain operations with gas optimization."""
        chain = step["chain"]
        action = step["action"]
        params = step["params"]
        web3 = self.web3_providers.get(chain)
        if not web3:
            raise ValueError(f"No Web3 provider for chain: {chain}")

        if action == "transfer":
            # Example: Ethereum transfer
            tx = {
                "from": params["from"],
                "to": params["to"],
                "value": Web3.to_wei(params["amount"], "ether"),
                "gas": step.get("gas", {}).get("limit", 21000),
                "maxFeePerGas": step.get("gas", {}).get("max_fee_per_gas"),
                "maxPriorityFeePerGas": step.get("gas", {}).get("priority_fee_per_gas"),
            }
            signed_tx = web3.eth.account.sign_transaction(tx, params["private_key"])
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            return tx_hash.hex()
        # Add other actions (mint, burn, etc.)
        raise NotImplementedError(f"Blockchain action {action} not implemented")

    async def _execute_quantum_operation(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute quantum circuit operations using Qiskit."""
        if step["type"] == "quantum_circuit":
            from qiskit import QuantumCircuit
            circuit = QuantumCircuit(step["qubits"])
            for gate in step["gates"]:
                gate_type = gate["gate"]
                target = gate["target"]
                if gate_type == "H": circuit.h(target)
                elif gate_type == "X": circuit.x(target)
                elif gate_type == "CNOT": circuit.cx(gate["control"], target)
                # Add other gates
            return circuit
        elif step["type"] == "quantum_measure":
            circuit = self.evaluate_expr(step["circuit"], context)
            circuit.measure_all()
            from qiskit_aer import AerSimulator
            simulator = AerSimulator()
            result = simulator.run(circuit).result()
            return result.get_counts()
        raise NotImplementedError(f"Quantum operation {step['type']} not implemented")

    async def _execute_ai_operation(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute AI operations (placeholder for integration with AI frameworks)."""
        model = step["model"]
        input_data = self.evaluate_expr(step["input"], context)
        # Placeholder: Integrate with AI framework (e.g., TensorFlow, PyTorch, or API)
        return {"prediction": "mock_result"}

    async def _execute_game_operation(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute game-related operations with pygame support."""
        if step["type"] == "game_render":
            scene = self.evaluate_expr(step["scene"], context)
            # Placeholder: Render scene using pygame
            return {"render_output": "mock_frame"}
        elif step["type"] == "game_physics":
            objects = [self.evaluate_expr(obj, context) for obj in step["objects"]]
            # Placeholder: Simulate physics
            return {"physics_state": "mock_state"}
        elif step["type"] == "game_multiplayer_sync":
            state = self.evaluate_expr(step["state"], context)
            # Placeholder: Sync state with peers
            return {"sync_result": "mock_sync"}
        elif step["type"] == "game_input":
            # Placeholder: Capture input
            return {"input_data": "mock_input"}
        elif step["type"] == "game_animation":
            target = self.evaluate_expr(step["target_object"], context)
            # Placeholder: Apply animation
            return {"animation_result": "mock_animation"}
        raise NotImplementedError(f"Game operation {step['type']} not implemented")

    async def evaluate_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a workflow step with retry and error handling."""
        self.execution_stats.steps_executed += 1
        step_type = step["type"]
        step_id = step.get("id", "unknown")
        self.logger.info(f"Executing step {step_type} (id: {step_id})")

        async with AsyncExitStack() as stack:
            timeout = step.get("timeout", {})
            duration = timeout.get("duration", "30s")
            max_retries = timeout.get("max_retries", 0)
            action = timeout.get("action", "fail")
            duration_seconds = int(re.match(r"(\d+)[smh]", duration).group(1)) * {"s": 1, "m": 60, "h": 3600}[duration[-1]]
            retries = 0

            while retries <= max_retries:
                try:
                    stack.enter_context(time_limit(duration_seconds))
                    if step_type == "set":
                        context[step["target"]] = self.evaluate_expr(step["value"], context)
                        return None
                    elif step_type == "if":
                        condition = self.evaluate_expr(step["condition"], context)
                        steps = step["then"] if condition else step.get("else", [])
                        for sub_step in steps:
                            result = await self.evaluate_step(sub_step, context)
                            if result: return result
                        return None
                    elif step_type == "return":
                        return {"value": self.evaluate_expr(step["value"], context)}
                    elif step_type == "call":
                        # Placeholder: Call external function
                        context[step["target"]] = {}
                        return None
                    elif step_type == "try":
                        try:
                            for sub_step in step["body"]:
                                result = await self.evaluate_step(sub_step, context)
                                if result: return result
                        except Exception as e:
                            if "catch" in step:
                                catch = step["catch"]
                                context[catch["error_var"]] = str(e)
                                for sub_step in catch["body"]:
                                    result = await self.evaluate_step(sub_step, context)
                                    if result: return result
                        finally:
                            if "finally" in step:
                                for sub_step in step["finally"]:
                                    result = await self.evaluate_step(sub_step, context)
                                    if result: return result
                        return None
                    elif step_type == "while":
                        max_iterations = step.get("max_iterations", 1000)
                        iteration = 0
                        while self.evaluate_expr(step["condition"], context) and iteration < max_iterations:
                            for sub_step in step["body"]:
                                result = await self.evaluate_step(sub_step, context)
                                if result: return result
                            iteration += 1
                        if iteration >= max_iterations:
                            self.logger.warning(f"While loop exceeded max iterations: {max_iterations}")
                        return None
                    elif step_type == "foreach":
                        collection = self.evaluate_expr(step["collection"], context)
                        for item in collection:
                            context[step["iterator"]] = item
                            for sub_step in step["body"]:
                                result = await self.evaluate_step(sub_step, context)
                                if result: return result
                        return None
                    elif step_type == "parallel":
                        merge_strategy = step.get("merge_strategy", "all")
                        tasks = [asyncio.create_task(self._execute_branch(branch, context.copy())) for branch in step["branches"]]
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        if merge_strategy == "all":
                            context[step.get("target", "parallel_result")] = results
                        elif merge_strategy == "first":
                            context[step.get("target", "parallel_result")] = next((r for r in results if not isinstance(r, Exception)), None)
                        elif merge_strategy == "last":
                            context[step.get("target", "parallel_result")] = results[-1] if not isinstance(results[-1], Exception) else None
                        return None
                    elif step_type == "assert":
                        if not self.evaluate_expr(step["condition"], context):
                            raise AssertionError(step.get("message", "Assertion failed"))
                        return None
                    elif step_type == "event":
                        self.logger.info(f"Emitting event: {step['name']} with params: {step['params']}")
                        # Placeholder: Emit event to external system
                        return None
                    elif step_type == "require_role":
                        # Placeholder: Role-based access control
                        return None
                    elif step_type in ["ai_infer", "ai_train", "ai_classify", "ai_embed", "ai_explain"]:
                        result = await self._execute_ai_operation(step, context)
                        context[step["target"]] = result
                        return None
                    elif step_type in ["quantum_circuit", "quantum_measure", "quantum_algorithm"]:
                        result = await self._execute_quantum_operation(step, context)
                        context[step["target"]] = result
                        return None
                    elif step_type == "blockchain_operation":
                        result = await self._execute_blockchain_operation(step, context)
                        context[step["target"]] = result
                        return None
                    elif step_type == "crypto_sign":
                        algorithm = step["algorithm"]
                        data = self.evaluate_expr(step["data"], context)
                        key = self.evaluate_expr(step["key"], context)
                        if algorithm == "ecdsa":
                            signed = Account.sign_message(data, key)
                            context[step["target"]] = signed.signature.hex()
                        # Add other algorithms
                        return None
                    elif step_type == "crypto_verify":
                        # Placeholder: Verify signature
                        context[step["target"]] = True
                        return None
                    elif step_type == "regex_match":
                        pattern = step["pattern"]
                        input_data = self.evaluate_expr(step["input"], context)
                        context[step["target"]] = bool(re.match(pattern, str(input_data)))
                        return None
                    elif step_type == "audit_log":
                        self.logger.info(f"Audit log: {step['message']} - Metadata: {step.get('metadata', {})}")
                        return None
                    elif step_type == "call_workflow":
                        # Placeholder: Execute subworkflow
                        context[step["target"]] = {}
                        return None
                    elif step_type.startswith("custom_"):
                        # Placeholder: Custom step execution
                        context[step.get("target", "custom_result")] = step.get("custom_properties", {})
                        return None
                    elif step_type in ["game_render", "game_physics", "game_multiplayer_sync", "game_input", "game_animation"]:
                        result = await self._execute_game_operation(step, context)
                        context[step.get("target", step.get("render_target", "game_result"))] = result
                        return None
                    elif step_type == "script":
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.executor,
                            self.execute_script,
                            step["script"],
                            step.get("language", "python"),
                            step.get("inputs", {}),
                            step.get("sandbox", {}),
                            context
                        )
                        context[step["target"]] = result
                        return None
                    else:
                        raise ValueError(f"Unknown step type: {step_type}")
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        if action == "fail":
                            self.execution_stats.errors.append(str(e))
                            raise
                        elif action == "skip":
                            self.logger.warning(f"Skipping step {step_type} after {retries} retries: {str(e)}")
                            return None
                        elif action == "retry":
                            self.logger.warning(f"Retrying step {step_type} ({retries}/{max_retries}): {str(e)}")
                            await asyncio.sleep(2 ** retries)  # Exponential backoff
                            continue
                    await asyncio.sleep(2 ** retries)

            if "on_error" in step:
                error_handler = step["on_error"]
                if error_handler.get("step_id"):
                    # Placeholder: Jump to error step
                    pass
                elif error_handler.get("body"):
                    for sub_step in error_handler["body"]:
                        result = await self.evaluate_step(sub_step, context)
                        if result: return result
            return None

    async def _execute_branch(self, branch: List[Dict[str, Any]], context: Dict[str, Any]) -> Any:
        """Execute a branch of steps in parallel execution."""
        for step in branch:
            result = await self.evaluate_step(step, context)
            if result: return result
        return None

    async def execute_workflow(self, workflow: Dict[str, Any], initial_context: Dict[str, Any] = None) -> Any:
        """Execute the workflow with full production-ready features."""
        async with self:
            try:
                self.validate_workflow(workflow)
                self._enforce_rate_limit(workflow)
                self.context = initial_context or {}
                self.execution_stats = ExecutionStats(start_time=time.time())

                # Initialize context from schema
                for key, value in workflow["schema"].get("context", {}).items():
                    source = value.get("source")
                    if source == "game_state":
                        self.context[key] = {"score": 0, "position": [0, 0, 0]}  # Mock
                    elif source == "player_input":
                        self.context[key] = {}  # Mock
                    elif source == "blockchain":
                        # Placeholder: Fetch blockchain state
                        self.context[key] = {}
                    else:
                        self.context[key] = None

                # Verify attestation if present
                if "attestation" in workflow:
                    attestation = workflow["attestation"]
                    expected_hash = attestation["hash"]
                    computed_hash = hashlib.sha256(json.dumps(workflow, sort_keys=True).encode()).hexdigest()
                    if computed_hash != expected_hash:
                        raise ValueError("Workflow attestation verification failed")

                # Execute steps
                for step in workflow["steps"]:
                    result = await self.evaluate_step(step, self.context)
                    if result: return result["value"]

                # Collect outputs
                outputs = {}
                for key, spec in workflow["schema"]["outputs"].items():
                    outputs[key] = self.context.get(key)
                
                self.execution_stats.record_end()
                self.logger.info(f"Workflow {workflow['function']} completed in {self.execution_stats.duration():.2f}s")
                return outputs

            except Exception as e:
                self.execution_stats.errors.append(str(e))
                self.execution_stats.record_end()
                self.logger.error(f"Workflow {workflow.get('function', 'unknown')} failed: {str(e)}")
                raise
            finally:
                # Log execution stats
                self.logger.info(f"Execution stats: {self.execution_stats.__dict__}")

# Example usage
async def main():
    schema = json.load(open("jsonflow_schema.json"))  # Load updated schema
    parser = JSONFlowParser(schema)
    
    workflow = {
        "function": "gameWorkflow",
        "metadata": {"schema_version": "1.2.0", "version": "1.0.0", "author": "xAI", "description": "Game workflow"},
        "schema": {
            "inputs": {"player_input": {"type": "object", "source": "player_input"}},
            "context": {"game_state": {"type": "object", "source": "game_state"}},
            "outputs": {"updated_state": {"type": "object"}}
        },
        "execution_policy": {"max_runs_per_minute": 10, "max_concurrent_runs": 5},
        "steps": [
            {
                "type": "script",
                "id": "step1",
                "language": "python",
                "script": """
import pygame
result = {'position': player_input['position'], 'score': game_state['score'] + 10}
""",
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
