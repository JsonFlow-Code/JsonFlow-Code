import json
import logging
import re
import ast
import asyncio
from typing import Dict, Any, List, Set, Optional
from jsonschema import validate, ValidationError
from contextlib import contextmanager
from datetime import datetime
import hashlib
import esprima  # For JavaScript/TypeScript/React validation
import lupa  # For Lua validation
from qiskit import QuantumCircuit  # For quantum circuit validation
from solcx import compile_source  # For Solidity validation
from jinja2 import Environment, BaseLoader  # For template validation
from web3 import Web3  # For blockchain validation
from eth_account import Account  # For crypto operations
import numpy as np  # For numerical validation
import aiohttp  # For async HTTP requests
from mermaid2python import parse_mermaid  # Hypothetical Mermaid parser

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, path: Optional[str] = None):
        super().__init__(message)
        self.path = path

@contextmanager
def validation_context(description: str):
    """Context manager for logging validation steps."""
    logger.info(f"Starting validation: {description}")
    try:
        yield
    except Exception as e:
        logger.error(f"Validation failed: {description} - {str(e)}")
        raise
    finally:
        logger.info(f"Completed validation: {description}")

class JSONFlowValidator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.supported_languages = {
            "cpp", "go", "javascript", "julia", "mermaid", "perl", "python",
            "qiskit", "react", "rust", "solidity", "typescript", "cairo",
            "java", "kotlin", "lua"
        }
        self.supported_game_engines = {"unity", "unreal", "godot", "bevy", "custom"}
        self.reserved_keywords = {
            "cpp": {"class", "namespace", "void", "int", "return", "template", "static", "const"},
            "go": {"func", "package", "import", "var", "return", "defer", "go", "select"},
            "javascript": {"function", "var", "let", "const", "return", "async", "await", "class"},
            "typescript": {"function", "let", "const", "interface", "return", "type", "async", "await"},
            "react": {"function", "let", "const", "return", "useState", "useEffect", "component"},
            "julia": {"function", "end", "return", "module", "struct", "macro"},
            "perl": {"sub", "my", "return", "package", "use", "our"},
            "python": {"def", "return", "class", "import", "async", "await", "with", "try"},
            "qiskit": {"def", "return", "class", "import", "QuantumCircuit", "measure"},
            "rust": {"fn", "let", "return", "struct", "impl", "async", "await", "mut"},
            "solidity": {"function", "contract", "return", "public", "external", "view", "pure"},
            "cairo": {"func", "return", "let", "namespace", "struct", "felt"},
            "java": {"class", "public", "private", "return", "void", "static", "final"},
            "kotlin": {"fun", "val", "var", "return", "class", "override", "lateinit"},
            "lua": {"function", "local", "return", "end", "if", "then"},
            "mermaid": set()
        }
        self.valid_blockchains = {"ethereum", "solana", "starknet", "cosmos", "polkadot", "binance", "avalanche"}
        self.valid_asset_formats = {"fbx", "gltf", "png", "jpg", "wav", "mp3", "ogg", "blend", "dae"}
        self.valid_platforms = {"pc", "console", "mobile", "web", "vr", "ar"}
        self.jinja_env = Environment(loader=BaseLoader())
        self.lua_runtime = lupa.LuaRuntime()
        self.web3_providers = {
            "ethereum": Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/YOUR_PROJECT_ID")),
            "solana": None,  # Requires solana-py or similar
            "starknet": None  # Requires starknet.py or similar
        }
        self.session = None  # For aiohttp

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

    async def validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """Validate the workflow exhaustively against schema and custom rules."""
        async with self:
            with validation_context(f"Workflow {workflow.get('function', 'unknown')}"):
                try:
                    # JSON Schema validation
                    validate(instance=workflow, schema=self.schema)
                    logger.info("JSON Schema validation passed")

                    # Comprehensive custom validation
                    await asyncio.gather(
                        self._validate_function(workflow),
                        self._validate_metadata(workflow),
                        self._validate_schema(workflow),
                        self._validate_game(workflow),
                        self._validate_access_policy(workflow),
                        self._validate_execution_policy(workflow),
                        self._validate_secrets(workflow),
                        self._validate_invariants(workflow),
                        self._validate_tests(workflow),
                        self._validate_attestation(workflow),
                        self._validate_history(workflow),
                        self._validate_resource_estimates(workflow),
                        self._validate_ui(workflow),
                        self._validate_subworkflows(workflow),
                        self._validate_verification_results(workflow),
                        self._validate_steps(workflow["steps"], set()),
                    )
                    logger.info("All validations passed successfully")
                except ValidationError as e:
                    raise ValidationError(f"Schema validation failed: {e.message} at {e.json_path}", e.json_path)
                except Exception as e:
                    raise ValidationError(f"Validation failed: {str(e)}")

    async def _validate_function(self, workflow: Dict[str, Any]) -> None:
        """Validate function name across all target languages."""
        with validation_context("Function name"):
            function = workflow.get("function")
            if not function:
                raise ValidationError("Missing function name")
            
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{1,63}$", function):
                raise ValidationError(f"Invalid function name: {function}. Must match ^[a-zA-Z][a-zA-Z0-9_]{1,63}$")
            
            target_languages = workflow.get("metadata", {}).get("target_languages", self.supported_languages)
            for lang in target_languages:
                if lang not in self.supported_languages:
                    raise ValidationError(f"Unsupported target language: {lang}")
                if function.lower() in self.reserved_keywords.get(lang, set()):
                    raise ValidationError(f"Function name '{function}' is reserved in {lang}")

    async def _validate_metadata(self, workflow: Dict[str, Any]) -> None:
        """Validate metadata with version and dependency checks."""
        with validation_context("Metadata"):
            metadata = workflow.get("metadata", {})
            if not metadata.get("schema_version") or not re.match(r"^\d+\.\d+\.\d+$", metadata["schema_version"]):
                raise ValidationError("Invalid or missing schema_version")
            
            if not metadata.get("version") or not re.match(r"^\d+\.\d+\.\d+$", metadata["version"]):
                raise ValidationError("Invalid or missing version")
            
            if not metadata.get("author") or len(metadata["author"]) > 100:
                raise ValidationError("Invalid or missing author")
            
            if not metadata.get("description") or len(metadata["description"]) > 1000:
                raise ValidationError("Invalid or missing description")
            
            if metadata.get("created"):
                try:
                    datetime.fromisoformat(metadata["created"].replace("Z", "+00:00"))
                except ValueError:
                    raise ValidationError("Invalid created date format")
            
            if metadata.get("mermaid"):
                try:
                    parse_mermaid(metadata["mermaid"])
                except Exception as e:
                    raise ValidationError(f"Invalid Mermaid diagram: {str(e)}")
            
            for lang in metadata.get("target_languages", []):
                if lang not in self.supported_languages:
                    raise ValidationError(f"Invalid target language: {lang}")
            
            for engine in metadata.get("game_engines", []):
                if engine not in self.supported_game_engines:
                    raise ValidationError(f"Invalid game engine: {engine}")
            
            for dep_name, dep in metadata.get("dependencies", {}).items():
                if not re.match(r"^[a-zA-Z0-9_.-]+$", dep_name):
                    raise ValidationError(f"Invalid dependency name: {dep_name}")
                if not re.match(r"^\d+\.\d+\.\d+$", dep.get("version", "")):
                    raise ValidationError(f"Invalid version for dependency {dep_name}")
                if not re.match(r"^[a-fA-F0-9]{64}$", dep.get("hash", "")):
                    raise ValidationError(f"Invalid hash for dependency {dep_name}")
                if dep.get("type") not in ["library", "game_library", "asset", "contract"]:
                    raise ValidationError(f"Invalid type for dependency {dep_name}")
                if dep.get("source") and not re.match(r"^(http|https|file)://", dep["source"]):
                    raise ValidationError(f"Invalid source for dependency {dep_name}")

    async def _validate_schema(self, workflow: Dict[str, Any]) -> None:
        """Validate inputs, context, and outputs in schema."""
        with validation_context("Schema"):
            schema = workflow.get("schema", {})
            for section in ["inputs", "context", "outputs"]:
                for key, spec in schema.get(section, {}).items():
                    if not re.match(r"^[a-zA-Z0-9_]+$", key):
                        raise ValidationError(f"Invalid key for {section}.{key}")
                    if not spec.get("type"):
                        raise ValidationError(f"Missing type for {section}.{key}")
                    
                    valid_types = ["string", "integer", "number", "boolean", "object", "array", "null"]
                    if section == "outputs":
                        valid_types.extend(["game_state", "render_output", "physics_state", "quantum_state"])
                    
                    if spec["type"] not in valid_types:
                        raise ValidationError(f"Invalid type for {section}.{key}: {spec['type']}")
                    
                    if section == "context" and spec.get("source") not in [
                        "env", "config", "blockchain", "external_api", "game_state", "player_input", "database", "quantum_simulator"
                    ]:
                        raise ValidationError(f"Invalid source for context.{key}: {spec['source']}")
                    
                    if "constraints" in spec:
                        constraints = spec["constraints"]
                        if "minLength" in constraints and constraints["minLength"] < 0:
                            raise ValidationError(f"Invalid minLength for {section}.{key}")
                        if "maxLength" in constraints and constraints["maxLength"] < constraints.get("minLength", 0):
                            raise ValidationError(f"Invalid maxLength for {section}.{key}")
                        if "pattern" in constraints and not self._is_valid_regex(constraints["pattern"]):
                            raise ValidationError(f"Invalid regex pattern for {section}.{key}")
                        if "enum" in constraints and not constraints["enum"]:
                            raise ValidationError(f"Empty enum for {section}.{key}")
                        if "minimum" in constraints and "maximum" in constraints and constraints["minimum"] > constraints["maximum"]:
                            raise ValidationError(f"Invalid range for {section}.{key}")
                    
                    if "ui" in spec:
                        await self._validate_ui_definition(spec["ui"], f"{section}.{key}.ui")

    async def _validate_steps(self, steps: List[Dict[str, Any]], seen_ids: Set[str]) -> None:
        """Recursively validate all steps."""
        with validation_context("Steps"):
            for step in steps:
                step_id = step.get("id")
                if not step_id or not re.match(r"^[a-zA-Z0-9_]+$", step_id):
                    raise ValidationError("Missing or invalid step ID")
                if step_id in seen_ids:
                    raise ValidationError(f"Duplicate step ID: {step_id}")
                seen_ids.add(step_id)
                
                step_type = step.get("type")
                if not step_type:
                    raise ValidationError(f"Missing type for step {step_id}")
                
                await self._validate_common_step_properties(step, step_id)
                
                step_handlers = {
                    "set": self._validate_set_step,
                    "if": self._validate_if_step,
                    "return": self._validate_return_step,
                    "call": self._validate_call_step,
                    "try": self._validate_try_step,
                    "while": self._validate_while_step,
                    "foreach": self._validate_foreach_step,
                    "parallel": self._validate_parallel_step,
                    "assert": self._validate_assert_step,
                    "event": self._validate_event_step,
                    "require_role": self._validate_require_role_step,
                    "ai_infer": self._validate_ai_step,
                    "ai_train": self._validate_ai_step,
                    "ai_classify": self._validate_ai_step,
                    "ai_embed": self._validate_ai_step,
                    "ai_explain": self._validate_ai_step,
                    "quantum_circuit": self._validate_quantum_step,
                    "quantum_measure": self._validate_quantum_step,
                    "quantum_algorithm": self._validate_quantum_step,
                    "blockchain_operation": self._validate_blockchain_step,
                    "crypto_sign": self._validate_crypto_sign_step,
                    "crypto_verify": self._validate_crypto_verify_step,
                    "regex_match": self._validate_regex_match_step,
                    "audit_log": self._validate_audit_log_step,
                    "call_workflow": self._validate_call_workflow_step,
                    "game_render": self._validate_game_step,
                    "game_physics": self._validate_game_step,
                    "game_multiplayer_sync": self._validate_game_step,
                    "game_input": self._validate_game_step,
                    "game_animation": self._validate_game_step,
                    "script": self._validate_script_step
                }
                
                if step_type.startswith("custom_"):
                    await self._validate_custom_step(step, step_id)
                elif step_type in step_handlers:
                    await step_handlers[step_type](step, step_id, seen_ids if step_type in ["if", "try", "while", "foreach", "parallel"] else None)
                else:
                    raise ValidationError(f"Unknown step type for step {step_id}: {step_type}")

    async def _validate_common_step_properties(self, step: Dict[str, Any], step_id: str) -> None:
        """Validate common properties across all steps."""
        with validation_context(f"Common properties for step {step_id}"):
            if "timeout" in step:
                timeout = step["timeout"]
                if not re.match(r"^\d+[smh]$", timeout.get("duration", "")):
                    raise ValidationError(f"Invalid timeout duration for step {step_id}: {timeout.get('duration')}")
                if timeout.get("action") not in ["skip", "retry", "fail"]:
                    raise ValidationError(f"Invalid timeout action for step {step_id}: {timeout.get('action')}")
                if timeout.get("max_retries", 0) < 0 or timeout.get("max_retries", 0) > 10:
                    raise ValidationError(f"Invalid max_retries for step {step_id}")
            
            if "access_control" in step:
                for role in step["access_control"].get("roles", []):
                    if not re.match(r"^[a-zA-Z0-9_]+$", role):
                        raise ValidationError(f"Invalid role for step {step_id}: {role}")
                for perm in step["access_control"].get("permissions", []):
                    if not re.match(r"^[a-zA-Z0-9_]+$", perm):
                        raise ValidationError(f"Invalid permission for step {step_id}: {perm}")
            
            if "ui" in step:
                await self._validate_ui_definition(step["ui"], f"step.{step_id}.ui")
            
            if "resource_estimates" in step:
                for key in ["cpu", "memory", "storage", "network"]:
                    if step["resource_estimates"].get(key, 0) < 0:
                        raise ValidationError(f"Invalid {key} estimate for step {step_id}")
            
            if "on_error" in step:
                on_error = step["on_error"]
                if on_error.get("step_id") == step_id:
                    raise ValidationError(f"Self-referential on_error step_id in step {step_id}")
                if "body" in on_error:
                    await self._validate_steps(on_error["body"], seen_ids=set())

    async def _validate_set_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate set step."""
        with validation_context(f"Set step {step_id}"):
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")
            await self._validate_expression(step.get("value", {}), step_id, "value")

    async def _validate_if_step(self, step: Dict[str, Any], step_id: str, seen_ids: Set[str]) -> None:
        """Validate if step."""
        with validation_context(f"If step {step_id}"):
            await self._validate_expression(step.get("condition", {}), step_id, "condition")
            await self._validate_steps(step.get("then", []), seen_ids.copy())
            if "else" in step:
                await self._validate_steps(step.get("else", []), seen_ids.copy())

    async def _validate_return_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate return step."""
        with validation_context(f"Return step {step_id}"):
            await self._validate_expression(step.get("value", {}), step_id, "value")

    async def _validate_call_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate call step."""
        with validation_context(f"Call step {step_id}"):
            if not step.get("function") or not re.match(r"^[a-zA-Z0-9_]+$", step["function"]):
                raise ValidationError(f"Missing or invalid function for step {step_id}")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")
            for arg_name, expr in step.get("args", {}).items():
                if not re.match(r"^[a-zA-Z0-9_]+$", arg_name):
                    raise ValidationError(f"Invalid argument name for step {step_id}: {arg_name}")
                await self._validate_expression(expr, step_id, f"args.{arg_name}")

    async def _validate_try_step(self, step: Dict[str, Any], step_id: str, seen_ids: Set[str]) -> None:
        """Validate try step."""
        with validation_context(f"Try step {step_id}"):
            await self._validate_steps(step.get("body", []), seen_ids.copy())
            if "catch" in step:
                if not step["catch"].get("error_var") or not re.match(r"^[a-zA-Z0-9_]+$", step["catch"]["error_var"]):
                    raise ValidationError(f"Missing or invalid error_var for catch in step {step_id}")
                await self._validate_steps(step["catch"].get("body", []), seen_ids.copy())
            if "finally" in step:
                await self._validate_steps(step["finally"], seen_ids.copy())

    async def _validate_while_step(self, step: Dict[str, Any], step_id: str, seen_ids: Set[str]) -> None:
        """Validate while step."""
        with validation_context(f"While step {step_id}"):
            await self._validate_expression(step.get("condition", {}), step_id, "condition")
            if step.get("max_iterations", 1000) < 1 or step.get("max_iterations", 1000) > 10000:
                raise ValidationError(f"Invalid max_iterations for step {step_id}")
            await self._validate_steps(step.get("body", []), seen_ids.copy())

    async def _validate_foreach_step(self, step: Dict[str, Any], step_id: str, seen_ids: Set[str]) -> None:
        """Validate foreach step."""
        with validation_context(f"Foreach step {step_id}"):
            await self._validate_expression(step.get("collection", {}), step_id, "collection")
            if not step.get("iterator") or not re.match(r"^[a-zA-Z0-9_]+$", step["iterator"]):
                raise ValidationError(f"Missing or invalid iterator for step {step_id}")
            await self._validate_steps(step.get("body", []), seen_ids.copy())

    async def _validate_parallel_step(self, step: Dict[str, Any], step_id: str, seen_ids: Set[str]) -> None:
        """Validate parallel step."""
        with validation_context(f"Parallel step {step_id}"):
            if not step.get("branches"):
                raise ValidationError(f"Missing branches for step {step_id}")
            if step.get("merge_strategy") not in ["all", "first", "last", "merge"]:
                raise ValidationError(f"Invalid merge_strategy for step {step_id}")
            for i, branch in enumerate(step.get("branches", [])):
                if not branch:
                    raise ValidationError(f"Empty branch[{i}] for step {step_id}")
                await self._validate_steps(branch, seen_ids.copy())
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")

    async def _validate_assert_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate assert step."""
        with validation_context(f"Assert step {step_id}"):
            await self._validate_expression(step.get("condition", {}), step_id, "condition")
            if not step.get("message") or len(step["message"]) > 500:
                raise ValidationError(f"Missing or too long message for step {step_id}")

    async def _validate_event_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate event step."""
        with validation_context(f"Event step {step_id}"):
            if not step.get("name") or not re.match(r"^[a-zA-Z0-9_]+$", step["name"]):
                raise ValidationError(f"Missing or invalid name for step {step_id}")
            for param_name, expr in step.get("params", {}).items():
                if not re.match(r"^[a-zA-Z0-9_]+$", param_name):
                    raise ValidationError(f"Invalid parameter name for step {step_id}: {param_name}")
                await self._validate_expression(expr, step_id, f"params.{param_name}")

    async def _validate_require_role_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate require_role step."""
        with validation_context(f"Require_role step {step_id}"):
            if not step.get("role") or not re.match(r"^[a-zA-Z0-9_]+$", step["role"]):
                raise ValidationError(f"Missing or invalid role for step {step_id}")
            if step.get("policy") and step["policy"] not in ["rbac", "abac"]:
                raise ValidationError(f"Invalid policy for step {step_id}: {step.get('policy')}")

    async def _validate_ai_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate AI steps."""
        with validation_context(f"AI step {step_id}"):
            if not step.get("model") or not re.match(r"^[a-zA-Z0-9_.-]+$", step["model"]):
                raise ValidationError(f"Missing or invalid model for step {step_id}")
            await self._validate_expression(step.get("input", {}), step_id, "input")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")
            if step.get("framework") and step["framework"] not in ["tensorflow", "pytorch", "sklearn", "huggingface"]:
                raise ValidationError(f"Invalid framework for step {step_id}: {step.get('framework')}")

    async def _validate_quantum_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate quantum steps."""
        with validation_context(f"Quantum step {step_id}"):
            if step["type"] == "quantum_circuit":
                if not step.get("qubits") or step["qubits"] < 1 or step["qubits"] > 100:
                    raise ValidationError(f"Invalid qubits for step {step_id}: {step.get('qubits')}")
                valid_gates = {"H", "X", "Y", "Z", "CNOT", "T", "S", "RX", "RY", "RZ", "SWAP", "TOFFOLI", "CZ", "CPHASE"}
                for gate in step.get("gates", []):
                    if gate.get("gate") not in valid_gates:
                        raise ValidationError(f"Invalid gate for step {step_id}: {gate.get('gate')}")
                    if not gate.get("target") or not isinstance(gate["target"], (int, list)):
                        raise ValidationError(f"Missing or invalid target for gate in step {step_id}")
                    if gate.get("gate") in ["RX", "RY", "RZ", "CPHASE"] and not isinstance(gate.get("angle"), (int, float)):
                        raise ValidationError(f"Missing or invalid angle for gate {gate.get('gate')} in step {step_id}")
                    if gate.get("gate") in ["CNOT", "CZ", "CPHASE", "SWAP"] and not gate.get("control"):
                        raise ValidationError(f"Missing control qubit for gate {gate.get('gate')} in step {step_id}")
                # Validate circuit
                try:
                    circuit = QuantumCircuit(step["qubits"])
                    for gate in step.get("gates", []):
                        if gate["gate"] == "H":
                            circuit.h(gate["target"])
                        elif gate["gate"] == "CNOT":
                            circuit.cx(gate["control"], gate["target"])
                        # Add other gates as needed
                except Exception as e:
                    raise ValidationError(f"Invalid quantum circuit for step {step_id}: {str(e)}")
            elif step["type"] == "quantum_measure":
                await self._validate_expression(step.get("circuit", {}), step_id, "circuit")
                if not step.get("qubits") or not isinstance(step["qubits"], list):
                    raise ValidationError(f"Missing or invalid qubits for step {step_id}")
                if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                    raise ValidationError(f"Missing or invalid target for step {step_id}")
            elif step["type"] == "quantum_algorithm":
                valid_algorithms = {"grover", "shor", "qft", "vqe", "qaoa", "deutsch_jozsa"}
                if step.get("algorithm") not in valid_algorithms:
                    raise ValidationError(f"Invalid algorithm for step {step_id}: {step.get('algorithm')}")
                if not step.get("parameters") or not isinstance(step["parameters"], dict):
                    raise ValidationError(f"Missing or invalid parameters for step {step_id}")
                if step.get("simulator") not in ["qasm_simulator", "statevector_simulator", "unitary_simulator"]:
                    raise ValidationError(f"Invalid simulator for step {step_id}: {step.get('simulator')}")

    async def _validate_blockchain_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate blockchain operation step."""
        with validation_context(f"Blockchain step {step_id}"):
            if step.get("chain") not in self.valid_blockchains:
                raise ValidationError(f"Invalid chain for step {step_id}: {step.get('chain')}")
            valid_actions = {"transfer", "mint", "burn", "governance", "bridge", "flash_loan", "swap", "liquidate", "deploy_contract"}
            if step.get("action") not in valid_actions:
                raise ValidationError(f"Invalid action for step {step_id}: {step.get('action')}")
            if not step.get("params") or not isinstance(step["params"], dict):
                raise ValidationError(f"Missing or invalid params for step {step_id}")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")
            if "gas" in step:
                gas = step["gas"]
                if gas.get("limit", 21000) < 21000 or gas.get("limit", 21000) > 30000000:
                    raise ValidationError(f"Invalid gas limit for step {step_id}: {gas.get('limit')}")
                if gas.get("max_fee_per_gas", 0) < 0:
                    raise ValidationError(f"Invalid max_fee_per_gas for step {step_id}")
                if gas.get("max_priority_fee_per_gas", 0) < 0:
                    raise ValidationError(f"Invalid max_priority_fee_per_gas for step {step_id}")
            if step["chain"] == "ethereum" and step.get("action") == "deploy_contract":
                if not step["params"].get("source"):
                    raise ValidationError(f"Missing contract source for step {step_id}")
                try:
                    compile_source(step["params"]["source"], output_values=["abi", "bin"])
                except Exception as e:
                    raise ValidationError(f"Invalid Solidity contract for step {step_id}: {str(e)}")

    async def _validate_crypto_sign_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate crypto_sign step."""
        with validation_context(f"Crypto_sign step {step_id}"):
            valid_algorithms = {"ecdsa", "ed25519", "rsa", "schnorr"}
            if step.get("algorithm") not in valid_algorithms:
                raise ValidationError(f"Invalid algorithm for step {step_id}: {step.get('algorithm')}")
            await self._validate_expression(step.get("data", {}), step_id, "data")
            await self._validate_expression(step.get("key", {}), step_id, "key")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")
            if step["algorithm"] == "ecdsa" and step.get("chain") == "ethereum":
                key = step.get("key", {}).get("value", "")
                if isinstance(key, str) and not re.match(r"^0x[a-fA-F0-9]{64}$", key):
                    raise ValidationError(f"Invalid Ethereum private key for step {step_id}")

    async def _validate_crypto_verify_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate crypto_verify step."""
        with validation_context(f"Crypto_verify step {step_id}"):
            valid_algorithms = {"ecdsa", "ed25519", "rsa", "schnorr"}
            if step.get("algorithm") not in valid_algorithms:
                raise ValidationError(f"Invalid algorithm for step {step_id}: {step.get('algorithm')}")
            await self._validate_expression(step.get("data", {}), step_id, "data")
            await self._validate_expression(step.get("signature", {}), step_id, "signature")
            await self._validate_expression(step.get("key", {}), step_id, "key")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")
            if step["algorithm"] == "ecdsa" and step.get("chain") == "ethereum":
                signature = step.get("signature", {}).get("value", "")
                if isinstance(signature, str) and not re.match(r"^0x[a-fA-F0-9]+$", signature):
                    raise ValidationError(f"Invalid Ethereum signature for step {step_id}")

    async def _validate_regex_match_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate regex_match step."""
        with validation_context(f"Regex_match step {step_id}"):
            if not step.get("pattern"):
                raise ValidationError(f"Missing pattern for step {step_id}")
            if not self._is_valid_regex(step["pattern"]):
                raise ValidationError(f"Invalid regex pattern for step {step_id}: {step['pattern']}")
            await self._validate_expression(step.get("input", {}), step_id, "input")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")

    async def _validate_audit_log_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate audit_log step."""
        with validation_context(f"Audit_log step {step_id}"):
            if not step.get("message") or len(step["message"]) > 1000:
                raise ValidationError(f"Missing or too long message for step {step_id}")
            if step.get("metadata"):
                if not isinstance(step["metadata"], dict):
                    raise ValidationError(f"Invalid metadata for step {step_id}")
                if len(json.dumps(step["metadata"])) > 5000:
                    raise ValidationError(f"Metadata too large for step {step_id}")

    async def _validate_call_workflow_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate call_workflow step."""
        with validation_context(f"Call_workflow step {step_id}"):
            if not step.get("workflow"):
                raise ValidationError(f"Missing workflow URI for step {step_id}")
            if not re.match(r"^(http|https|file)://[a-zA-Z0-9_./-]+$", step["workflow"]):
                raise ValidationError(f"Invalid workflow URI for step {step_id}: {step['workflow']}")
            for arg_name, expr in step.get("args", {}).items():
                if not re.match(r"^[a-zA-Z0-9_]+$", arg_name):
                    raise ValidationError(f"Invalid argument name for step {step_id}: {arg_name}")
                await self._validate_expression(expr, step_id, f"args.{arg_name}")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")

    async def _validate_custom_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate custom step."""
        with validation_context(f"Custom step {step_id}"):
            if not step.get("custom_properties") or not isinstance(step["custom_properties"], dict):
                raise ValidationError(f"Missing or invalid custom_properties for step {step_id}")
            if not re.match(r"^custom_[a-zA-Z0-9_]+$", step["type"]):
                raise ValidationError(f"Invalid custom step type for step {step_id}: {step['type']}")
            if len(json.dumps(step["custom_properties"])) > 10000:
                raise ValidationError(f"Custom properties too large for step {step_id}")

    async def _validate_game_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate game-specific steps."""
        with validation_context(f"Game step {step_id}"):
            if step["type"] == "game_render":
                await self._validate_expression(step.get("scene", {}), step_id, "scene")
                if not step.get("render_target") or not re.match(r"^[a-zA-Z0-9_]+$", step["render_target"]):
                    raise ValidationError(f"Missing or invalid render_target for step {step_id}")
                if "camera" in step:
                    camera = step["camera"]
                    if len(camera.get("position", [])) != 3 or not all(isinstance(x, (int, float)) for x in camera.get("position", [])):
                        raise ValidationError(f"Invalid camera position for step {step_id}")
                    if "rotation" in camera and len(camera["rotation"]) != 3:
                        raise ValidationError(f"Invalid camera rotation for step {step_id}")
                    if "fov" in camera and (camera["fov"] < 1 or camera["fov"] > 180):
                        raise ValidationError(f"Invalid camera FOV for step {step_id}")
            elif step["type"] == "game_physics":
                if not step.get("objects"):
                    raise ValidationError(f"Missing objects for step {step_id}")
                for i, obj in enumerate(step.get("objects", [])):
                    await self._validate_expression(obj, step_id, f"objects[{i}]")
                if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                    raise ValidationError(f"Missing or invalid target for step {step_id}")
                simulation = step.get("simulation", {})
                valid_sim_types = {"rigid_body", "soft_body", "fluid", "particle"}
                if simulation.get("type") not in valid_sim_types:
                    raise ValidationError(f"Invalid simulation type for step {step_id}: {simulation.get('type')}")
                if "gravity" in simulation and (len(simulation["gravity"]) != 3 or not all(isinstance(x, (int, float)) for x in simulation["gravity"])):
                    raise ValidationError(f"Invalid gravity vector for step {step_id}")
                if "timestep" in simulation and simulation["timestep"] <= 0:
                    raise ValidationError(f"Invalid timestep for step {step_id}")
            elif step["type"] == "game_multiplayer_sync":
                await self._validate_expression(step.get("state", {}), step_id, "state")
                valid_sync_types = {"state", "event", "delta"}
                if step.get("sync_type") not in valid_sync_types:
                    raise ValidationError(f"Invalid sync_type for step {step_id}: {step.get('sync_type')}")
                if not step.get("peers") or not isinstance(step["peers"], list):
                    raise ValidationError(f"Missing or invalid peers for step {step_id}")
                if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                    raise ValidationError(f"Missing or invalid target for step {step_id}")
            elif step["type"] == "game_input":
                valid_input_types = {"keyboard", "mouse", "controller", "touch", "vr", "ar"}
                if step.get("input_type") not in valid_input_types:
                    raise ValidationError(f"Invalid input_type for step {step_id}: {step.get('input_type')}")
                if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                    raise ValidationError(f"Missing or invalid target for step {step_id}")
                if not step.get("bindings") or not isinstance(step["bindings"], dict):
                    raise ValidationError(f"Missing or invalid bindings for step {step_id}")
                for binding in step["bindings"]:
                    if not re.match(r"^[a-zA-Z0-9_]+$", binding):
                        raise ValidationError(f"Invalid binding key for step {step_id}: {binding}")
            elif step["type"] == "game_animation":
                await self._validate_expression(step.get("target_object", {}), step_id, "target_object")
                animation = step.get("animation", {})
                valid_anim_types = {"skeletal", "keyframe", "procedural"}
                if animation.get("type") not in valid_anim_types:
                    raise ValidationError(f"Invalid animation type for step {step_id}: {animation.get('type')}")
                if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                    raise ValidationError(f"Missing or invalid target for step {step_id}")
                if "duration" in animation and animation["duration"] <= 0:
                    raise ValidationError(f"Invalid animation duration for step {step_id}")
                if "keyframes" in animation and not isinstance(animation["keyframes"], list):
                    raise ValidationError(f"Invalid keyframes for step {step_id}")

    async def _validate_script_step(self, step: Dict[str, Any], step_id: str, _: Optional[Set[str]] = None) -> None:
        """Validate script step with language-specific checks."""
        with validation_context(f"Script step {step_id}"):
            if not step.get("script") or not isinstance(step["script"], str):
                raise ValidationError(f"Missing or invalid script for step {step_id}")
            if not step.get("target") or not re.match(r"^[a-zA-Z0-9_]+$", step["target"]):
                raise ValidationError(f"Missing or invalid target for step {step_id}")
            language = step.get("language", "python")
            valid_languages = {"python", "javascript", "typescript", "lua"}
            if language not in valid_languages:
                raise ValidationError(f"Unsupported script language for step {step_id}: {language}")
            
            # Validate script syntax
            if language == "python":
                try:
                    ast.parse(step["script"])
                except SyntaxError as e:
                    raise ValidationError(f"Invalid Python syntax in step {step_id}: {str(e)}")
            elif language in ["javascript", "typescript"]:
                try:
                    esprima.parseScript(step["script"])
                except esprima.Error as e:
                    raise ValidationError(f"Invalid JavaScript/TypeScript syntax in step {step_id}: {str(e)}")
            elif language == "lua":
                try:
                    self.lua_runtime.eval(step["script"])
                except lupa.LuaError as e:
                    raise ValidationError(f"Invalid Lua syntax in step {step_id}: {str(e)}")
            
            # Validate sandbox
            sandbox = step.get("sandbox", {})
            valid_modules = {
                "python": {"math", "random", "datetime", "json", "pygame", "numpy"},
                "javascript": {"mathjs", "lodash"},
                "typescript": {"mathjs", "lodash"},
                "lua": set()
            }
            for module in sandbox.get("allowed_modules", []):
                if module not in valid_modules[language]:
                    raise ValidationError(f"Invalid module for {language} in step {step_id}: {module}")
            
            if sandbox.get("max_execution_time", 5) < 1 or sandbox.get("max_execution_time", 5) > 60:
                raise ValidationError(f"Invalid max_execution_time for step {step_id}: {sandbox.get('max_execution_time')}")
            if sandbox.get("max_memory", 10240) < 1024 or sandbox.get("max_memory", 10240) > 524288:
                raise ValidationError(f"Invalid max_memory for step {step_id}: {sandbox.get('max_memory')}")
            
            # Validate inputs
            for key, expr in step.get("inputs", {}).items():
                if not re.match(r"^[a-zA-Z0-9_]+$", key):
                    raise ValidationError(f"Invalid input key for step {step_id}: {key}")
                await self._validate_expression(expr, step_id, f"inputs.{key}")

    async def _validate_expression(self, expr: Dict[str, Any], step_id: str, expr_path: str) -> None:
        """Validate expressions recursively."""
        with validation_context(f"Expression {expr_path} in step {step_id}"):
            if not isinstance(expr, dict):
                raise ValidationError(f"Invalid expression at {expr_path}: must be an object", expr_path)
            
            valid_ops = {
                "get", "value", "add", "subtract", "multiply", "divide", "compare",
                "not", "and", "or", "concat", "hash", "regex", "map", "filter",
                "modulo", "pow", "sqrt", "abs"
            }
            if not any(op in expr for op in valid_ops):
                raise ValidationError(f"Invalid expression at {expr_path}: no valid operation", expr_path)
            
            if "get" in expr:
                if not re.match(r"^[a-zA-Z0-9_\.]+$", expr["get"]):
                    raise ValidationError(f"Invalid get path at {expr_path}: {expr['get']}", expr_path)
                if len(expr["get"].split(".")) > 10:
                    raise ValidationError(f"Get path too deep at {expr_path}: {expr['get']}", expr_path)
            
            if "value" in expr:
                if expr["value"] is None and "null" not in expr:
                    raise ValidationError(f"Invalid null value at {expr_path}", expr_path)
                if isinstance(expr["value"], str) and len(expr["value"]) > 10000:
                    raise ValidationError(f"Value too long at {expr_path}", expr_path)
            
            for op in ["add", "subtract", "multiply", "divide", "and", "or", "concat", "modulo", "pow"]:
                if op in expr:
                    if len(expr[op]) < 2:
                        raise ValidationError(f"Invalid {op} operation at {expr_path}: requires at least 2 operands", expr_path)
                    for i, operand in enumerate(expr[op]):
                        await self._validate_expression(operand, step_id, f"{expr_path}.{op}[{i}]")
            
            if "compare" in expr:
                compare = expr["compare"]
                valid_compare_ops = {"<", ">", "===", "<=", ">=", "!==", "in", "not in"}
                if compare.get("op") not in valid_compare_ops:
                    raise ValidationError(f"Invalid compare operator at {expr_path}: {compare.get('op')}", expr_path)
                await self._validate_expression(compare.get("left", {}), step_id, f"{expr_path}.compare.left")
                await self._validate_expression(compare.get("right", {}), step_id, f"{expr_path}.compare.right")
            
            if "not" in expr:
                await self._validate_expression(expr["not"], step_id, f"{expr_path}.not")
            
            if "hash" in expr:
                valid_hash_algorithms = {"sha256", "sha3", "keccak256", "blake2b", "sha1", "md5"}
                if expr["hash"].get("algorithm") not in valid_hash_algorithms:
                    raise ValidationError(f"Invalid hash algorithm at {expr_path}: {expr['hash'].get('algorithm')}", expr_path)
                await self._validate_expression(expr["hash"].get("input", {}), step_id, f"{expr_path}.hash.input")
            
            if "regex" in expr:
                if not self._is_valid_regex(expr["regex"].get("pattern", "")):
                    raise ValidationError(f"Invalid regex pattern at {expr_path}: {expr['regex'].get('pattern')}", expr_path)
                await self._validate_expression(expr["regex"].get("input", {}), step_id, f"{expr_path}.regex.input")
            
            if "map" in expr:
                await self._validate_expression(expr["map"].get("collection", {}), step_id, f"{expr_path}.map.collection")
                await self._validate_expression(expr["map"].get("operation", {}), step_id, f"{expr_path}.map.operation")
            
            if "filter" in expr:
                await self._validate_expression(expr["filter"].get("collection", {}), step_id, f"{expr_path}.filter.collection")
                await self._validate_expression(expr["filter"].get("condition", {}), step_id, f"{expr_path}.filter.condition")
            
            for op in ["sqrt", "abs"]:
                if op in expr:
                    await self._validate_expression(expr[op], step_id, f"{expr_path}.{op}")

    async def _validate_game(self, workflow: Dict[str, Any]) -> None:
        """Validate game configuration."""
        with validation_context("Game configuration"):
            game = workflow.get("game", {})
            if game.get("engine") and game["engine"] not in self.supported_game_engines:
                raise ValidationError(f"Invalid game engine: {game.get('engine')}")
            
            for platform in game.get("platforms", []):
                if platform not in self.valid_platforms:
                    raise ValidationError(f"Invalid platform: {platform}")
            
            asset_pipeline = game.get("asset_pipeline", {})
            for fmt in asset_pipeline.get("formats", []):
                if fmt not in self.valid_asset_formats:
                    raise ValidationError(f"Invalid asset format: {fmt}")
            if "max_size_mb" in asset_pipeline and asset_pipeline["max_size_mb"] <= 0:
                raise ValidationError(f"Invalid max_size_mb: {asset_pipeline.get('max_size_mb')}")
            
            if "multiplayer" in game:
                multiplayer = game["multiplayer"]
                valid_mp_types = {"p2p", "client-server", "hybrid"}
                if multiplayer.get("type") not in valid_mp_types:
                    raise ValidationError(f"Invalid multiplayer type: {multiplayer.get('type')}")
                if multiplayer.get("max_players", 1) < 1 or multiplayer.get("max_players", 1) > 1000:
                    raise ValidationError(f"Invalid max_players: {multiplayer.get('max_players')}")
                valid_protocols = {"udp", "tcp", "websocket", "webrtc"}
                if multiplayer.get("network_protocol") not in valid_protocols:
                    raise ValidationError(f"Invalid network_protocol: {multiplayer.get('network_protocol')}")
                if "latency_ms" in multiplayer and multiplayer["latency_ms"] < 0:
                    raise ValidationError(f"Invalid latency_ms: {multiplayer.get('latency_ms')}")

    async def _validate_access_policy(self, workflow: Dict[str, Any]) -> None:
        """Validate access policy."""
        with validation_context("Access policy"):
            access_policy = workflow.get("access_policy", {})
            for role in access_policy.get("roles", []):
                if not re.match(r"^[a-zA-Z0-9_]+$", role):
                    raise ValidationError(f"Invalid role: {role}")
            for perm in access_policy.get("permissions", []):
                if not re.match(r"^[a-zA-Z0-9_]+$", perm):
                    raise ValidationError(f"Invalid permission: {perm}")
            valid_policy_engines = {"opa", "casbin", "custom"}
            if access_policy.get("policy_engine") not in valid_policy_engines:
                raise ValidationError(f"Invalid policy_engine: {access_policy.get('policy_engine')}")
            if access_policy.get("policies") and not isinstance(access_policy["policies"], list):
                raise ValidationError("Invalid policies format")

    async def _validate_execution_policy(self, workflow: Dict[str, Any]) -> None:
        """Validate execution policy."""
        with validation_context("Execution policy"):
            execution_policy = workflow.get("execution_policy", {})
            if execution_policy.get("max_runs_per_minute", 1) < 1 or execution_policy.get("max_runs_per_minute", 1) > 1000:
                raise ValidationError(f"Invalid max_runs_per_minute: {execution_policy.get('max_runs_per_minute')}")
            if execution_policy.get("max_concurrent_runs", 1) < 1 or execution_policy.get("max_concurrent_runs", 1) > 100:
                raise ValidationError(f"Invalid max_concurrent_runs: {execution_policy.get('max_concurrent_runs')}")
            valid_priorities = {"low", "medium", "high"}
            if execution_policy.get("priority") and execution_policy["priority"] not in valid_priorities:
                raise ValidationError(f"Invalid priority: {execution_policy.get('priority')}")
            if "timeout_ms" in execution_policy and execution_policy["timeout_ms"] <= 0:
                raise ValidationError(f"Invalid timeout_ms: {execution_policy.get('timeout_ms')}")

    async def _validate_secrets(self, workflow: Dict[str, Any]) -> None:
        """Validate secrets configuration."""
        with validation_context("Secrets"):
            secrets = workflow.get("secrets", [])
            seen_names = set()
            for secret in secrets:
                if not secret.get("name") or not re.match(r"^[a-zA-Z0-9_]+$", secret["name"]):
                    raise ValidationError(f"Missing or invalid name for secret: {secret.get('name')}")
                if secret["name"] in seen_names:
                    raise ValidationError(f"Duplicate secret name: {secret['name']}")
                seen_names.add(secret["name"])
                valid_sources = {"env", "vault", "kms", "secret_manager"}
                if secret.get("source") not in valid_sources:
                    raise ValidationError(f"Invalid source for secret {secret['name']}: {secret.get('source')}")
                if secret.get("encryption") and secret["encryption"] not in ["aes", "rsa"]:
                    raise ValidationError(f"Invalid encryption for secret {secret['name']}: {secret.get('encryption')}")

    async def _validate_invariants(self, workflow: Dict[str, Any]) -> None:
        """Validate invariants for formal verification."""
        with validation_context("Invariants"):
            for invariant in workflow.get("invariants", []):
                await self._validate_expression(invariant.get("condition", {}), "invariant", "condition")
                if not invariant.get("message") or len(invariant["message"]) > 500:
                    raise ValidationError(f"Missing or too long message for invariant")
                valid_severities = {"error", "warning", "info"}
                if invariant.get("severity") not in valid_severities:
                    raise ValidationError(f"Invalid severity for invariant: {invariant.get('severity')}")
                valid_verification_tools = {"certora", "scribble", "mythril", "slither"}
                if invariant.get("verification_tool") and invariant["verification_tool"] not in valid_verification_tools:
                    raise ValidationError(f"Invalid verification_tool: {invariant.get('verification_tool')}")

    async def _validate_tests(self, workflow: Dict[str, Any]) -> None:
        """Validate test cases."""
        with validation_context("Tests"):
            for test in workflow.get("tests", []):
                if not test.get("name") or not re.match(r"^[a-zA-Z0-9_]+$", test["name"]):
                    raise ValidationError(f"Missing or invalid name for test")
                valid_test_types = {"example", "property", "fuzz", "unit", "integration"}
                if test.get("type") not in valid_test_types:
                    raise ValidationError(f"Invalid test type: {test.get('type')}")
                if not test.get("inputs") or not isinstance(test["inputs"], dict):
                    raise ValidationError(f"Missing or invalid inputs for test {test['name']}")
                if not test.get("expected") or not isinstance(test["expected"], dict):
                    raise ValidationError(f"Missing or invalid expected for test {test['name']}")
                if test.get("type") in ["property", "fuzz"]:
                    for i, prop in enumerate(test.get("properties", [])):
                        await self._validate_expression(prop, f"test_{test['name']}", f"properties[{i}]")
                if test.get("timeout_ms", 1000) <= 0:
                    raise ValidationError(f"Invalid timeout_ms for test {test['name']}")

    async def _validate_attestation(self, workflow: Dict[str, Any]) -> None:
        """Validate cryptographic attestation."""
        with validation_context("Attestation"):
            attestation = workflow.get("attestation", {})
            if attestation:
                for signer in attestation.get("signers", []):
                    if not re.match(r"^0x[a-fA-F0-9]{40}$", signer):
                        raise ValidationError(f"Invalid signer address: {signer}")
                if not re.match(r"^0x[a-fA-F0-9]+$", attestation.get("signature", "")):
                    raise ValidationError(f"Invalid signature: {attestation.get('signature')}")
                if not re.match(r"^0x[a-fA-F0-9]{64}$", attestation.get("hash", "")):
                    raise ValidationError(f"Invalid hash: {attestation.get('hash')}")
                # Verify signature
                try:
                    message_hash = Web3.keccak(text=json.dumps(workflow, sort_keys=True))
                    recovered = Account.recover_message(message_hash, signature=attestation["signature"])
                    if recovered not in attestation["signers"]:
                        raise ValidationError("Signature verification failed")
                except Exception as e:
                    raise ValidationError(f"Attestation verification failed: {str(e)}")

    async def _validate_history(self, workflow: Dict[str, Any]) -> None:
        """Validate change history."""
        with validation_context("History"):
            for entry in workflow.get("history", []):
                if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", entry.get("timestamp", "")):
                    raise ValidationError(f"Invalid timestamp for history entry: {entry.get('timestamp')}")
                if not entry.get("author") or len(entry["author"]) > 100:
                    raise ValidationError(f"Missing or invalid author for history entry")
                if not entry.get("change_summary") or len(entry["change_summary"]) > 1000:
                    raise ValidationError(f"Missing or too long change_summary for history entry")
                if entry.get("version") and not re.match(r"^\d+\.\d+\.\d+$", entry["version"]):
                    raise ValidationError(f"Invalid version for history entry: {entry.get('version')}")

    async def _validate_resource_estimates(self, workflow: Dict[str, Any]) -> None:
        """Validate resource estimates."""
        with validation_context("Resource estimates"):
            estimates = workflow.get("resource_estimates", {})
            for key in ["cpu", "memory", "storage", "network"]:
                if estimates.get(key, 0) < 0:
                    raise ValidationError(f"Invalid {key} estimate: {estimates.get(key)}")
                if key == "memory" and estimates.get(key, 0) > 1024 * 1024:  # 1TB
                    raise ValidationError(f"Memory estimate too large: {estimates.get(key)}")
            if estimates.get("execution_time_ms", 1000) <= 0:
                raise ValidationError(f"Invalid execution_time_ms: {estimates.get('execution_time_ms')}")

    async def _validate_ui(self, workflow: Dict[str, Any]) -> None:
        """Validate global UI configuration."""
        with validation_context("Global UI"):
            if "ui" in workflow:
                await self._validate_ui_definition(workflow["ui"], "global.ui")

    async def _validate_ui_definition(self, ui: Dict[str, Any], path: str) -> None:
        """Validate UI configuration."""
        with validation_context(f"UI at {path}"):
            valid_frameworks = {"react", "vue", "svelte", "game", "flutter"}
            if ui.get("framework") not in valid_frameworks:
                raise ValidationError(f"Invalid UI framework at {path}: {ui.get('framework')}")
            if not ui.get("component") or not re.match(r"^[a-zA-Z0-9_]+$", ui["component"]):
                raise ValidationError(f"Missing or invalid component at {path}")
            if "game_ui" in ui:
                game_ui = ui["game_ui"]
                valid_ui_types = {"hud", "menu", "3d", "vr", "ar"}
                if game_ui.get("type") not in valid_ui_types:
                    raise ValidationError(f"Invalid game_ui type at {path}: {game_ui.get('type')}")
                if "position" in game_ui and (len(game_ui["position"]) != 3 or not all(isinstance(x, (int, float)) for x in game_ui["position"])):
                    raise ValidationError(f"Invalid position at {path}")
                if "rotation" in game_ui and (len(game_ui["rotation"]) != 3 or not all(isinstance(x, (int, float)) for x in game_ui["rotation"])):
                    raise ValidationError(f"Invalid rotation at {path}")
                if "scale" in game_ui and (not isinstance(game_ui["scale"], (int, float)) or game_ui["scale"] <= 0):
                    raise ValidationError(f"Invalid scale at {path}")
            if ui.get("template"):
                try:
                    self.jinja_env.from_string(ui["template"]).render()
                except Exception as e:
                    raise ValidationError(f"Invalid Jinja2 template at {path}: {str(e)}")

    async def _validate_subworkflows(self, workflow: Dict[str, Any]) -> None:
        """Validate subworkflow references."""
        with validation_context("Subworkflows"):
            for uri in workflow.get("subworkflows", []):
                if not re.match(r"^(http|https|file)://[a-zA-Z0-9_./-]+$", uri):
                    raise ValidationError(f"Invalid subworkflow URI: {uri}")
                if uri.startswith("http"):
                    try:
                        async with self.session.get(uri, timeout=5) as response:
                            if response.status != 200:
                                raise ValidationError(f"Failed to fetch subworkflow: {uri}")
                            subworkflow = await response.json()
                            await self.validate_workflow(subworkflow)
                    except Exception as e:
                        raise ValidationError(f"Subworkflow validation failed for {uri}: {str(e)}")

    async def _validate_verification_results(self, workflow: Dict[str, Any]) -> None:
        """Validate verification results."""
        with validation_context("Verification results"):
            for result in workflow.get("verification_results", []):
                if not result.get("tool") or not re.match(r"^[a-zA-Z0-9_]+$", result["tool"]):
                    raise ValidationError(f"Missing or invalid tool for verification result")
                if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result.get("date", "")):
                    raise ValidationError(f"Invalid date for verification result: {result.get('date')}")
                valid_results = {"success", "failure", "partial"}
                if result.get("result") not in valid_results:
                    raise ValidationError(f"Invalid result: {result.get('result')}")
                if result.get("issues") and not isinstance(result["issues"], list):
                    raise ValidationError(f"Invalid issues format for verification result")

    def _is_valid_regex(self, pattern: str) -> bool:
        """Check if a regex pattern is valid."""
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

# Example usage
async def main():
    try:
        with open("jsonflow_schema.json", "r") as f:
            schema = json.load(f)
        
        validator = JSONFlowValidator(schema)
        
        workflow = {
            "function": "gameWorkflow",
            "metadata": {
                "schema_version": "1.2.0",
                "version": "1.0.0",
                "author": "xAI",
                "description": "Game workflow with multiplayer and quantum components",
                "target_languages": ["python", "javascript", "lua"],
                "game_engines": ["unity"],
                "created": "2025-05-11T10:00:00Z",
                "dependencies": {
                    "pygame": {"version": "2.5.2", "hash": "a" * 64, "type": "library", "source": "https://pypi.org"},
                    "three.js": {"version": "0.150.0", "hash": "b" * 64, "type": "library", "source": "https://npmjs.com"}
                }
            },
            "schema": {
                "inputs": {
                    "player_input": {
                        "type": "object",
                        "source": "player_input",
                        "constraints": {"required": ["position"]}
                    }
                },
                "context": {
                    "game_state": {
                        "type": "object",
                        "source": "game_state"
                    }
                },
                "outputs": {
                    "updated_state": {
                        "type": "object",
                        "constraints": {"required": ["score"]}
                    }
                }
            },
            "game": {
                "engine": "unity",
                "platforms": ["pc", "mobile"],
                "asset_pipeline": {"formats": ["fbx", "png"], "max_size_mb": 500},
                "multiplayer": {
                    "type": "client-server",
                    "max_players": 16,
                    "network_protocol": "websocket",
                    "latency_ms": 100
                }
            },
            "access_policy": {
                "roles": ["admin", "player"],
                "permissions": ["execute", "view"],
                "policy_engine": "opa"
            },
            "execution_policy": {
                "max_runs_per_minute": 10,
                "max_concurrent_runs": 5,
                "priority": "medium",
                "timeout_ms": 30000
            },
            "secrets": [
                {"name": "api_key", "source": "vault", "encryption": "aes"}
            ],
            "invariants": [
                {
                    "condition": {"value": true},
                    "message": "State consistency maintained",
                    "severity": "info",
                    "verification_tool": "slither"
                }
            ],
            "tests": [
                {
                    "name": "test_player_move",
                    "type": "unit",
                    "inputs": {"player_input": {"position": [0, 0, 0]}},
                    "expected": {"updated_state": {"score": 110}},
                    "timeout_ms": 1000
                }
            ],
            "attestation": {
                "signers": ["0x" + "1" * 40],
                "signature": "0x" + "a" * 130,
                "hash": "0x" + "b" * 64
            },
            "history": [
                {
                    "timestamp": "2025-05-11T10:00:00Z",
                    "author": "xAI",
                    "change_summary": "Initial version",
                    "version": "1.0.0"
                }
            ],
            "resource_estimates": {
                "cpu": 2.5,
                "memory": 4096,
                "storage": 1024,
                "network": 100,
                "execution_time_ms": 5000
            },
            "ui": {
                "framework": "react",
                "component": "GameUI",
                "game_ui": {
                    "type": "hud",
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0],
                    "scale": 1.0
                },
                "template": "{{ score }} points"
            },
            "subworkflows": [
                "https://example.com/workflows/subworkflow.json"
            ],
            "verification_results": [
                {
                    "tool": "slither",
                    "date": "2025-05-11T10:00:00Z",
                    "result": "success",
                    "issues": []
                }
            ],
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
                    },
                    "timeout": {"duration": "10s", "action": "retry", "max_retries": 2},
                    "access_control": {"roles": ["player"], "permissions": ["execute"]},
                    "resource_estimates": {"cpu": 0.5, "memory": 512}
                },
                {
                    "type": "quantum_circuit",
                    "id": "step2",
                    "qubits": 2,
                    "gates": [
                        {"gate": "H", "target": 0},
                        {"gate": "CNOT", "control": 0, "target": 1}
                    ],
                    "target": "quantum_state",
                    "timeout": {"duration": "5s", "action": "fail"}
                },
                {
                    "type": "blockchain_operation",
                    "id": "step3",
                    "chain": "ethereum",
                    "action": "transfer",
                    "params": {
                        "from": "0x" + "1" * 40,
                        "to": "0x" + "2" * 40,
                        "amount": 1.0
                    },
                    "target": "tx_hash",
                    "gas": {"limit": 21000, "max_fee_per_gas": 1000000000},
                    "timeout": {"duration": "30s", "action": "retry", "max_retries": 3}
                }
            ]
        }
        
        await validator.validate_workflow(workflow)
        logger.info("Workflow is valid!")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
