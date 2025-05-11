
import json
import logging
import re
import ast
from typing import Dict, Any, List, Set
from jsonschema import validate, ValidationError
from contextlib import contextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONFlowValidator:
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.supported_languages = {
            "cpp", "go", "javascript", "julia", "mermaid", "perl", "python",
            "qiskit", "react", "rust", "solidity", "typescript"
        }
        self.supported_game_engines = {"unity", "unreal", "godot", "custom"}
        self.reserved_keywords = {
            "cpp": {"class", "namespace", "void", "int", "return"},
            "go": {"func", "package", "import", "var", "return"},
            "javascript": {"function", "var", "let", "const", "return"},
            "typescript": {"function", "let", "const", "interface", "return"},
            "react": {"function", "let", "const", "return", "component"},
            "julia": {"function", "end", "return", "module"},
            "perl": {"sub", "my", "return", "package"},
            "python": {"def", "return", "class", "import"},
            "qiskit": {"def", "return", "class", "import"},  # Inherits Python keywords
            "rust": {"fn", "let", "return", "struct"},
            "solidity": {"function", "contract", "return", "public"},
            "mermaid": set()  # No reserved keywords for diagrams
        }

    def validate_workflow(self, workflow: Dict[str, Any]) -> None:
        """Validate the workflow against the schema and custom rules."""
        try:
            # Schema validation
            logger.info("Starting schema validation")
            validate(instance=workflow, schema=self.schema)
            logger.info("Schema validation successful")

            # Custom validation
            self._validate_function(workflow)
            self._validate_metadata(workflow)
            self._validate_schema(workflow)
            self._validate_steps(workflow["steps"], set())
            self._validate_game(workflow)
            self._validate_access_policy(workflow)
            self._validate_execution_policy(workflow)
            self._validate_secrets(workflow)
            logger.info("Custom validation successful")

        except ValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            raise ValueError(f"Schema validation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise

    def _validate_function(self, workflow: Dict[str, Any]) -> None:
        """Validate the function name for all target languages."""
        function = workflow.get("function")
        if not function:
            raise ValueError("Missing function name")
        
        # Check naming conventions
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", function):
            raise ValueError(f"Invalid function name: {function}. Must match ^[a-zA-Z][a-zA-Z0-9_]*$")
        
        # Check reserved keywords for each language
        target_languages = workflow.get("metadata", {}).get("target_languages", self.supported_languages)
        for lang in target_languages:
            if lang not in self.supported_languages:
                raise ValueError(f"Unsupported target language: {lang}")
            if function in self.reserved_keywords.get(lang, set()):
                raise ValueError(f"Function name '{function}' is a reserved keyword in {lang}")

    def _validate_metadata(self, workflow: Dict[str, Any]) -> None:
        """Validate metadata properties."""
        metadata = workflow.get("metadata", {})
        schema_version = metadata.get("schema_version")
        if not schema_version:
            raise ValueError("Missing schema_version in metadata")
        
        target_languages = metadata.get("target_languages", [])
        for lang in target_languages:
            if lang not in self.supported_languages:
                raise ValueError(f"Unsupported target language in metadata: {lang}")
        
        game_engines = metadata.get("game_engines", [])
        for engine in game_engines:
            if engine not in self.supported_game_engines:
                raise ValueError(f"Unsupported game engine: {engine}")
        
        dependencies = metadata.get("dependencies", {})
        for dep_name, dep in dependencies.items():
            if not dep.get("version"):
                raise ValueError(f"Dependency {dep_name} missing version")
            if dep.get("type") not in ["library", "game_library", "asset"]:
                raise ValueError(f"Invalid dependency type for {dep_name}: {dep['type']}")

    def _validate_schema(self, workflow: Dict[str, Any]) -> None:
        """Validate schema (inputs, context, outputs)."""
        schema = workflow.get("schema", {})
        for section in ["inputs", "context", "outputs"]:
            for key, spec in schema.get(section, {}).items():
                if not spec.get("type"):
                    raise ValueError(f"Missing type for {section}.{key}")
                if section == "context" and spec.get("source") not in [
                    "env", "config", "blockchain", "external_api", "game_state", "player_input"
                ]:
                    raise ValueError(f"Invalid source for context.{key}: {spec['source']}")
                if section == "outputs" and spec["type"] not in [
                    "string", "integer", "number", "boolean", "object", "array", "null",
                    "game_state", "render_output", "physics_state"
                ]:
                    raise ValueError(f"Invalid output type for {section}.{key}: {spec['type']}")

    def _validate_steps(self, steps: List[Dict[str, Any]], seen_ids: Set[str]) -> None:
        """Recursively validate steps."""
        for step in steps:
            step_id = step.get("id", "unknown")
            if step_id in seen_ids:
                raise ValueError(f"Duplicate step ID: {step_id}")
            seen_ids.add(step_id)

            step_type = step.get("type")
            if not step_type:
                raise ValueError(f"Missing type for step {step_id}")

            # Validate common step properties
            self._validate_common_step_properties(step, step_id)

            # Step-specific validation
            if step_type == "script":
                self._validate_script_step(step, step_id)
            elif step_type in ["if", "try", "while", "foreach", "parallel"]:
                self._validate_control_flow_step(step, step_id, seen_ids)
            elif step_type in ["game_render", "game_physics", "game_multiplayer_sync", "game_input", "game_animation"]:
                self._validate_game_step(step, step_id)
            elif step_type in ["quantum_circuit", "quantum_measure", "quantum_algorithm"]:
                self._validate_quantum_step(step, step_id)
            elif step_type == "blockchain_operation":
                self._validate_blockchain_step(step, step_id)
            elif step_type.startswith("custom_"):
                self._validate_custom_step(step, step_id)

    def _validate_common_step_properties(self, step: Dict[str, Any], step_id: str) -> None:
        """Validate common step properties."""
        if "timeout" in step:
            timeout = step["timeout"]
            if not re.match(r"^\d+[smh]$", timeout.get("duration", "")):
                raise ValueError(f"Invalid timeout duration for step {step_id}: {timeout.get('duration')}")
            if timeout.get("action") not in ["skip", "retry", "fail"]:
                raise ValueError(f"Invalid timeout action for step {step_id}: {timeout.get('action')}")
        
        if "on_error" in step:
            on_error = step["on_error"]
            if on_error.get("step_id") and on_error.get("step_id") == step_id:
                raise ValueError(f"Self-referential on_error step_id in step {step_id}")
            if "body" in on_error:
                self._validate_steps(on_error["body"], set())

    def _validate_script_step(self, step: Dict[str, Any], step_id: str) -> None:
        """Validate script step."""
        if not step.get("script"):
            raise ValueError(f"Missing script for step {step_id}")
        if not step.get("target"):
            raise ValueError(f"Missing target for step {step_id}")
        
        # Validate script syntax
        try:
            ast.parse(step["script"])
        except SyntaxError as e:
            raise ValueError(f"Invalid Python script syntax in step {step_id}: {str(e)}")
        
        # Validate sandbox
        sandbox = step.get("sandbox", {})
        allowed_modules = sandbox.get("allowed_modules", [])
        valid_modules = {"math", "random", "datetime", "json", "pygame", "numpy"}
        for module in allowed_modules:
            if module not in valid_modules:
                raise ValueError(f"Invalid module in step {step_id}: {module}")
        
        max_execution_time = sandbox.get("max_execution_time", 5)
        if not isinstance(max_execution_time, int) or max_execution_time < 1:
            raise ValueError(f"Invalid max_execution_time in step {step_id}: {max_execution_time}")
        
        max_memory = sandbox.get("max_memory", 10240)
        if not isinstance(max_memory, int) or max_memory < 1024:
            raise ValueError(f"Invalid max_memory in step {step_id}: {max_memory}")
        
        # Validate inputs
        for key, expr in step.get("inputs", {}).items():
            self._validate_expression(expr, step_id, f"script.inputs.{key}")

    def _validate_control_flow_step(self, step: Dict[str, Any], step_id: str, seen_ids: Set[str]) -> None:
        """Validate control flow steps (if, try, while, foreach, parallel)."""
        if step["type"] == "if":
            self._validate_expression(step.get("condition"), step_id, "condition")
            self._validate_steps(step.get("then", []), seen_ids.copy())
            if "else" in step:
                self._validate_steps(step["else"], seen_ids.copy())
        elif step["type"] == "try":
            self._validate_steps(step.get("body", []), seen_ids.copy())
            if "catch" in step:
                self._validate_steps(step["catch"].get("body", []), seen_ids.copy())
            if "finally" in step:
                self._validate_steps(step["finally"], seen_ids.copy())
        elif step["type"] == "while":
            self._validate_expression(step.get("condition"), step_id, "condition")
            self._validate_steps(step.get("body", []), seen_ids.copy())
        elif step["type"] == "foreach":
            self._validate_expression(step.get("collection"), step_id, "collection")
            if not step.get("iterator"):
                raise ValueError(f"Missing iterator for step {step_id}")
            self._validate_steps(step.get("body", []), seen_ids.copy())
        elif step["type"] == "parallel":
            for branch in step.get("branches", []):
                self._validate_steps(branch, seen_ids.copy())

    def _validate_game_step(self, step: Dict[str, Any], step_id: str) -> None:
        """Validate game-specific steps."""
        if step["type"] == "game_render":
            self._validate_expression(step.get("scene"), step_id, "scene")
            if not step.get("render_target"):
                raise ValueError(f"Missing render_target for step {step_id}")
            if "camera" in step:
                camera = step["camera"]
                if len(camera.get("position", [])) != 3:
                    raise ValueError(f"Invalid camera position for step {step_id}")
        elif step["type"] == "game_physics":
            for obj in step.get("objects", []):
                self._validate_expression(obj, step_id, "objects")
            if not step.get("target"):
                raise ValueError(f"Missing target for step {step_id}")
            simulation = step.get("simulation", {})
            if simulation.get("type") not in ["rigid_body", "soft_body", "fluid"]:
                raise ValueError(f"Invalid simulation type for step {step_id}")
        elif step["type"] == "game_multiplayer_sync":
            self._validate_expression(step.get("state"), step_id, "state")
            if step.get("sync_type") not in ["state", "event", "delta"]:
                raise ValueError(f"Invalid sync_type for step {step_id}")
        elif step["type"] == "game_input":
            if step.get("input_type") not in ["keyboard", "mouse", "controller", "touch", "vr"]:
                raise ValueError(f"Invalid input_type for step {step_id}")
            if not step.get("target"):
                raise ValueError(f"Missing target for step {step_id}")
        elif step["type"] == "game_animation":
            self._validate_expression(step.get("target_object"), step_id, "target_object")
            if step.get("animation", {}).get("type") not in ["skeletal", "keyframe", "procedural"]:
                raise ValueError(f"Invalid animation type for step {step_id}")

    def _validate_quantum_step(self, step: Dict[str, Any], step_id: str) -> None:
        """Validate quantum steps."""
        if step["type"] == "quantum_circuit":
            if not step.get("qubits") or step["qubits"] < 1:
                raise ValueError(f"Invalid qubits for step {step_id}")
            for gate in step.get("gates", []):
                if gate.get("gate") not in ["H", "X", "Y", "Z", "CNOT", "T", "S", "RX", "RY", "RZ"]:
                    raise ValueError(f"Invalid gate for step {step_id}: {gate.get('gate')}")
        elif step["type"] == "quantum_measure":
            self._validate_expression(step.get("circuit"), step_id, "circuit")
        elif step["type"] == "quantum_algorithm":
            if step.get("algorithm") not in ["grover", "shor", "qft"]:
                raise ValueError(f"Invalid algorithm for step {step_id}")

    def _validate_blockchain_step(self, step: Dict[str, Any], step_id: str) -> None:
        """Validate blockchain operation step."""
        if step.get("chain") not in ["ethereum", "solana", "starknet", "cosmos", "polkadot"] and not re.match(r"^[a-zA-Z0-9_]+$", step.get("chain", "")):
            raise ValueError(f"Invalid chain for step {step_id}: {step.get('chain')}")
        if step.get("action") not in ["transfer", "mint", "burn", "governance", "bridge", "flash_loan", "swap", "liquidate"] and not re.match(r"^[a-zA-Z0-9_]+$", step.get("action", "")):
            raise ValueError(f"Invalid action for step {step_id}: {step.get('action')}")

    def _validate_custom_step(self, step: Dict[str, Any], step_id: str) -> None:
        """Validate custom steps."""
        if not step.get("custom_properties"):
            raise ValueError(f"Missing custom_properties for step {step_id}")

    def _validate_expression(self, expr: Dict[str, Any], step_id: str, expr_path: str) -> None:
        """Validate an expression."""
        if not isinstance(expr, dict):
            raise ValueError(f"Invalid expression at {expr_path} in step {step_id}: must be an object")
        
        valid_keys = {"get", "value", "add", "subtract", "multiply", "divide", "compare", "not", "and", "or", "concat", "hash", "regex"}
        if not any(key in expr for key in valid_keys):
            raise ValueError(f"Invalid expression at {expr_path} in step {step_id}: no valid operation")
        
        if "get" in expr and not re.match(r"^[a-zA-Z0-9_\.]+$", expr["get"]):
            raise ValueError(f"Invalid get expression at {expr_path} in step {step_id}: {expr['get']}")
        
        if "compare" in expr:
            compare = expr["compare"]
            if compare.get("op") not in ["<", ">", "===", "<=", ">=", "!=="]:
                raise ValueError(f"Invalid compare operator at {expr_path} in step {step_id}: {compare.get('op')}")
            self._validate_expression(compare.get("left", {}), step_id, f"{expr_path}.compare.left")
            self._validate_expression(compare.get("right", {}), step_id, f"{expr_path}.compare.right")

    def _validate_game(self, workflow: Dict[str, Any]) -> None:
        """Validate game configuration."""
        game = workflow.get("game", {})
        if game.get("engine") and game["engine"] not in self.supported_game_engines:
            raise ValueError(f"Invalid game engine: {game['engine']}")
        
        platforms = game.get("platforms", [])
        valid_platforms = {"pc", "console", "mobile", "web", "vr"}
        for platform in platforms:
            if platform not in valid_platforms:
                raise ValueError(f"Invalid platform: {platform}")
        
        asset_pipeline = game.get("asset_pipeline", {})
        valid_formats = {"fbx", "gltf", "png", "wav", "mp3"}
        for fmt in asset_pipeline.get("formats", []):
            if fmt not in valid_formats:
                raise ValueError(f"Invalid asset format: {fmt}")

    def _validate_access_policy(self, workflow: Dict[str, Any]) -> None:
        """Validate access policy."""
        access_policy = workflow.get("access_policy", {})
        for role in access_policy.get("roles", []):
            if not re.match(r"^[a-zA-Z0-9_]+$", role):
                raise ValueError(f"Invalid role: {role}")
        for perm in access_policy.get("permissions", []):
            if not re.match(r"^[a-zA-Z0-9_]+$", perm):
                raise ValueError(f"Invalid permission: {perm}")

    def _validate_execution_policy(self, workflow: Dict[str, Any]) -> None:
        """Validate execution policy."""
        execution_policy = workflow.get("execution_policy", {})
        if "max_runs_per_minute" in execution_policy and execution_policy["max_runs_per_minute"] < 1:
            raise ValueError(f"Invalid max_runs_per_minute: {execution_policy['max_runs_per_minute']}")
        if "max_concurrent_runs" in execution_policy and execution_policy["max_concurrent_runs"] < 1:
            raise ValueError(f"Invalid max_concurrent_runs: {execution_policy['max_concurrent_runs']}")

    def _validate_secrets(self, workflow: Dict[str, Any]) -> None:
        """Validate secrets."""
        secrets = workflow.get("secrets", [])
        for secret in secrets:
            if not secret.get("name"):
                raise ValueError("Missing name for secret")
            if secret.get("source") not in ["env", "vault", "other"]:
                raise ValueError(f"Invalid source for secret {secret['name']}: {secret['source']}")

# Example usage
if __name__ == "__main__":
    with open("extended_schema.json", "r") as f:
        schema = json.load(f)
    
    validator = JSONFlowValidator(schema)
    
    # Sample workflow
    workflow = {
        "function": "gameWorkflow",
        "schema": {
            "inputs": {"player_input": {"type": "object", "source": "player_input"}},
            "context": {"game_state": {"type": "game_state", "source": "game_state"}},
            "outputs": {"updated_state": {"type": "game_state"}}
        },
        "metadata": {
            "schema_version": "1.1.0",
            "target_languages": ["python", "javascript", "react"],
            "game_engines": ["unity"]
        },
        "game": {
            "engine": "unity",
            "platforms": ["pc", "mobile"],
            "asset_pipeline": {"formats": ["fbx", "png"]}
        },
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
    
    try:
        validator.validate_workflow(workflow)
        print("Workflow is valid!")
    except Exception as e:
        print(f"Validation failed: {str(e)}")
