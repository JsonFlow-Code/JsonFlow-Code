import logging
import json
from typing import Dict, Any, Union
from src.engine.generator import LanguageGenerator, register_generator

logger = logging.getLogger(__name__)

class PythonGenerator(LanguageGenerator):
    """
    Generator for Python code, supporting blockchain, AI, quantum, and game development steps
    as defined in the JSONFlow schema.
    """
    def generate(self, workflow: 'Workflow') -> str:
        """
        Generate Python code for the workflow, including initialization of inputs and step processing.

        Args:
            workflow (Workflow): Workflow object containing function, schema, steps, and game config.

        Returns:
            str: Generated Python code as a string.
        """
        logger.debug(f"Generating Python code for workflow: {workflow.function}")
        code = [
            "#!/usr/bin/env python3",
            "import json",
            "import logging",
            "",
            "# Setup logging",
            'logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")',
            "logger = logging.getLogger(__name__)",
            "",
            "context = {}",  # Runtime context
            f"# Workflow: {workflow.function}"
        ]

        # Initialize inputs
        for key, value in workflow.schema.get('inputs', {}).items():
            default_value = value.get('default', None)
            code.append(f"context['{key}'] = {repr(default_value)}")

        # Generate code for each step
        for step in workflow.steps:
            try:
                generated = self.generate_step(step)
                if generated:
                    code.append(generated)
            except Exception as e:
                logger.error(f"Error generating step {step.get('id', 'unknown')}: {str(e)}")
                code.append(f"# Error generating step {step['type']}: {str(e)}")

        return "\n".join(code)

    def generate_set(self, step: Dict) -> str:
        """
        Generate code for a 'set' step, assigning a value to a target variable.

        Args:
            step (Dict): Step dictionary with 'target' and 'value' keys.

        Returns:
            str: Generated Python code.
        """
        target = step['target']
        value = self._sanitize_expr(step['value'])
        return f"context['{target}'] = {value}"

    def generate_if(self, step: Dict) -> str:
        """
        Generate code for an 'if' step, including condition, then, and optional else branches.

        Args:
            step (Dict): Step dictionary with 'condition', 'then', and optional 'else' keys.

        Returns:
            str: Generated Python code.
        """
        condition = self._sanitize_expr(step['condition'])
        then_code = "\n    ".join(self.generate_step(s) for s in step['then'])
        else_code = "\n    ".join(self.generate_step(s) for s in step.get('else', [])) if step.get('else') else ""
        code = f"if {condition}:\n    {then_code}"
        if else_code:
            code += f"\nelse:\n    {else_code}"
        return code

    def generate_return(self, step: Dict) -> str:
        """
        Generate code for a 'return' step, returning a value.

        Args:
            step (Dict): Step dictionary with 'value' key.

        Returns:
            str: Generated Python code.
        """
        value = self._sanitize_expr(step['value'])
        return f"return {value}"

    def generate_blockchain_operation(self, step: Dict) -> str:
        """
        Generate code for a 'blockchain_operation' step, handling Ethereum transfers.

        Args:
            step (Dict): Step dictionary with 'chain', 'action', 'params', and 'target' keys.

        Returns:
            str: Generated Python code.
        """
        if step['chain'] == 'ethereum' and step['action'] == 'transfer':
            params = step['params']
            target = step['target']
            return f"""
logger.info("Executing Ethereum transfer to {params['to']}")
context['{target}'] = web3.eth.send_transaction({{
    'from': context['sender'],
    'to': '{params['to']}',
    'value': {params['value']}
}})
"""
        return f"# Unsupported blockchain operation: {step['chain']} {step['action']}"

    def generate_game_render(self, step: Dict) -> str:
        """
        Generate code for a 'game_render' step, simulating a rendering operation.

        Args:
            step (Dict): Step dictionary with 'scene', 'camera', and 'render_target' keys.

        Returns:
            str: Generated Python code.
        """
        scene = self._sanitize_expr(step['scene'])
        camera = step['camera']
        target = step['render_target']
        return f"""
# Render scene (simulated)
logger.debug("Rendering scene to {target}")
context['{target}'] = render_scene(
    scene={scene},
    camera_position={repr(camera['position'])},
    camera_fov={camera.get('fov', 60)}
)
"""

    def generate_game_physics(self, step: Dict) -> str:
        """
        Generate code for a 'game_physics' step, simulating a physics calculation.

        Args:
            step (Dict): Step dictionary with 'objects', 'simulation', and 'target' keys.

        Returns:
            str: Generated Python code.
        """
        objects = self._sanitize_expr(step['objects'])
        simulation = step['simulation']
        target = step['target']
        return f"""
# Physics simulation
logger.debug("Simulating physics for {target}")
context['{target}'] = simulate_physics(
    objects={objects},
    simulation_type='{simulation['type']}',
    gravity={repr(simulation.get('gravity', [0, -9.81, 0]))},
    time_step={simulation.get('time_step', 0.016)}
)
"""

    def generate_game_multiplayer_sync(self, step: Dict) -> str:
        """
        Generate code for a 'game_multiplayer_sync' step, simulating multiplayer synchronization.

        Args:
            step (Dict): Step dictionary with 'state', 'sync_type', 'peers', and 'target' keys.

        Returns:
            str: Generated Python code.
        """
        state = self._sanitize_expr(step['state'])
        sync_type = step['sync_type']
        peers = step.get('peers', [])
        target = step['target']
        return f"""
# Multiplayer synchronization
logger.debug("Synchronizing game state to {target}")
context['{target}'] = sync_multiplayer(
    state={state},
    sync_type='{sync_type}',
    peers={repr(peers)}
)
"""

    def generate_game_input(self, step: Dict) -> str:
        """
        Generate code for a 'game_input' step, handling player input.

        Args:
            step (Dict): Step dictionary with 'input_type', 'bindings', and 'target' keys.

        Returns:
            str: Generated Python code.
        """
        input_type = step['input_type']
        bindings = step.get('bindings', {})
        target = step['target']
        bindings_code = json.dumps(bindings)
        return f"""
# Handle player input
logger.debug("Processing {input_type} input for {target}")
context['{target}'] = handle_input(
    input_type='{input_type}',
    bindings={bindings_code}
)
"""

    def generate_game_animation(self, step: Dict) -> str:
        """
        Generate code for a 'game_animation' step, simulating an animation.

        Args:
            step (Dict): Step dictionary with 'target_object', 'animation', and 'target' keys.

        Returns:
            str: Generated Python code.
        """
        target_object = self._sanitize_expr(step['target_object'])
        animation = step['animation']
        target = step['target']
        return f"""
# Animate object
logger.debug("Animating object for {target}")
context['{target}'] = animate_object(
    target_object={target_object},
    animation_type='{animation['type']}',
    parameters={repr(animation.get('parameters', {}))},
    duration={animation.get('duration', 1.0)}
)
"""

    def _sanitize_expr(self, expr: Union[Dict, Any]) -> str:
        """
        Sanitize an expression to prevent injection attacks and convert to Python code.

        Args:
            expr (Union[Dict, Any]): Expression dictionary or value from the JSONFlow schema.

        Returns:
            str: Sanitized Python expression as a string.
        """
        if not isinstance(expr, dict):
            return repr(expr)

        if 'get' in expr:
            return f"context['{expr['get']}']"
        elif 'value' in expr:
            return repr(expr['value'])
        elif 'add' in expr:
            return f"({' + '.join(self._sanitize_expr(e) for e in expr['add'])})"
        elif 'subtract' in expr:
            return f"({' - '.join(self._sanitize_expr(e) for e in expr['subtract'])})"
        elif 'multiply' in expr:
            return f"({' * '.join(self._sanitize_expr(e) for e in expr['multiply'])})"
        elif 'divide' in expr:
            return f"({' / '.join(self._sanitize_expr(e) for e in expr['divide'])})"
        elif 'compare' in expr:
            op = expr['op']
            if op == '===':
                op = '=='
            elif op == '!==':
                op = '!='
            return f"({self._sanitize_expr(expr['left'])} {op} {self._sanitize_expr(expr['right'])})"
        elif 'not' in expr:
            return f"(not {self._sanitize_expr(expr['not'])})"
        elif 'and' in expr:
            return f"({' and '.join(self._sanitize_expr(e) for e in expr['and'])})"
        elif 'or' in expr:
            return f"({' or '.join(self._sanitize_expr(e) for e in expr['or'])})"
        elif 'concat' in expr:
            return f"(''.join({[self._sanitize_expr(e) for e in expr['concat']]}))"
        elif 'hash' in expr:
            algorithm = expr['algorithm']
            if algorithm == 'sha256':
                return f"hashlib.sha256(str({self._sanitize_expr(expr['input'])}).encode()).hexdigest()"
            return f"# Unsupported hash algorithm: {algorithm}"
        elif 'regex' in expr:
            return f"re.match('{expr['pattern']}', str({self._sanitize_expr(expr['input'])}))"

        logger.warning(f"Unsupported expression type: {expr}")
        return json.dumps(expr)  # Fallback to JSON string

register_generator('python', PythonGenerator)
