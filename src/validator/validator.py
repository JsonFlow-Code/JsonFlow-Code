import ast
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, line: Optional[int] = None, step_id: Optional[str] = None):
        self.message = message
        self.line = line
        self.step_id = step_id
        msg = f"ValidationError: {message}"
        if line:
            msg += f" at line {line}"
        if step_id:
            msg += f" in step {step_id}"
        super().__init__(msg)

class CodeValidator(ABC):
    """Base class for code validators."""
    @abstractmethod
    def validate(self, code: str, workflow: 'Workflow') -> List[ValidationError]:
        """Validate the generated code and return a list of validation errors."""
        pass

class PythonValidator(CodeValidator):
    """Validator for Python code, including game development steps."""
    def validate(self, code: str, workflow: 'Workflow') -> List[ValidationError]:
        errors = []
        logger.info("Validating Python code")

        # 1. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(ValidationError(f"Syntax error: {str(e)}", line=e.lineno))
            logger.error(f"Syntax error at line {e.lineno}: {str(e)}")
            return errors

        # 2. Check for undefined variables
        try:
            tree = ast.parse(code)
            defined_vars = set()
            used_vars = set()

            class VariableVisitor(ast.NodeVisitor):
                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_vars.add(target.id)
                    self.generic_visit(node)

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        used_vars.add(node.id)
                    self.generic_visit(node)

            VariableVisitor().visit(tree)
            allowed_globals = {'context', 'web3', 'THREE'}  # Allow game-related globals
            undefined_vars = used_vars - defined_vars - set(workflow.schema.get('inputs', {}).keys()) - allowed_globals
            for var in undefined_vars:
                errors.append(ValidationError(f"Undefined variable: {var}"))
                logger.warning(f"Undefined variable detected: {var}")
        except Exception as e:
            errors.append(ValidationError(f"Error during variable analysis: {str(e)}"))
            logger.error(f"Variable analysis failed: {str(e)}")

        # 3. Validate blockchain operations
        for step in workflow.steps:
            step_id = step.get('id', 'unknown')
            if step.get('type') == 'blockchain_operation' and step.get('chain') == 'ethereum':
                if step.get('action') == 'transfer':
                    params = step.get('params', {})
                    if not params.get('to') or not re.match(r'^0x[a-fA-F0-9]{40}$', params.get('to')):
                        errors.append(ValidationError("Invalid Ethereum address in transfer operation", step_id=step_id))
                        logger.error(f"Invalid Ethereum address in step {step_id}")
                    if not params.get('value') or not isinstance(params.get('value'), (int, str)):
                        errors.append(ValidationError("Invalid value in transfer operation", step_id=step_id))
                        logger.error(f"Invalid value in step {step_id}")

        # 4. Validate game steps
        for step in workflow.steps:
            step_id = step.get('id', 'unknown')
            if step.get('type') == 'game_render':
                if not step.get('scene') or not step.get('render_target'):
                    errors.append(ValidationError("Missing scene or render_target in game_render step", step_id=step_id))
                    logger.error(f"Missing scene or render_target in step {step_id}")
                camera = step.get('camera', {})
                if not camera.get('position') or len(camera.get('position', [])) != 3:
                    errors.append(ValidationError("Invalid camera position in game_render step", step_id=step_id))
                    logger.error(f"Invalid camera position in step {step_id}")
            elif step.get('type') == 'game_physics':
                if not step.get('objects') or not step.get('simulation') or not step.get('target'):
                    errors.append(ValidationError("Missing objects, simulation, or target in game_physics step", step_id=step_id))
                    logger.error(f"Missing required fields in step {step_id}")
                simulation = step.get('simulation', {})
                if simulation.get('type') not in ['rigid_body', 'soft_body', 'fluid']:
                    errors.append(ValidationError("Invalid simulation type in game_physics step", step_id=step_id))
                    logger.error(f"Invalid simulation type in step {step_id}")

        return errors

class SolidityValidator(CodeValidator):
    """Validator for Solidity code, including game-related blockchain operations."""
    def validate(self, code: str, workflow: 'Workflow') -> List[ValidationError]:
        errors = []
        logger.info("Validating Solidity code")

        # 1. Basic syntax check
        lines = code.split('\n')
        if not any('pragma solidity' in line for line in lines):
            errors.append(ValidationError("Missing Solidity pragma"))
            logger.error("Missing Solidity pragma")
        if not any('contract Workflow' in line for line in lines):
            errors.append(ValidationError("Missing contract definition"))
            logger.error("Missing contract definition")

        # 2. Check for valid transfer operations
        for step in workflow.steps:
            step_id = step.get('id', 'unknown')
            if step.get('type') == 'blockchain_operation' and step.get('chain') == 'ethereum':
                if step.get('action') == 'transfer':
                    params = step.get('params', {})
                    if not params.get('to') or not re.match(r'^0x[a-fA-F0-9]{40}$', params.get('to')):
                        errors.append(ValidationError("Invalid Ethereum address in transfer operation", step_id=step_id))
                        logger.error(f"Invalid Ethereum address in step {step_id}")
                    if not params.get('value') or not isinstance(params.get('value'), (int, str)):
                        errors.append(ValidationError("Invalid value in transfer operation", step_id=step_id))
                        logger.error(f"Invalid value in step {step_id}")

        # 3. Check for balanced braces
        brace_count = 0
        for i, line in enumerate(lines, 1):
            brace_count += line.count('{') - line.count('}')
            if brace_count saccharine < 0:
                errors.append(ValidationError("Unmatched closing brace", line=i))
                logger.error(f"Unmatched closing brace at line {i}")
        if brace_count != 0:
            errors.append(ValidationError("Unmatched opening brace"))
            logger.error("Unmatched opening brace")

        return errors

class JavaScriptValidator(CodeValidator):
    """Validator for JavaScript code, focused on game development steps."""
    def validate(self, code: str, workflow: 'Workflow') -> List[ValidationError]:
        errors = []
        logger.info("Validating JavaScript code")

        # 1. Basic syntax check (simplified, using regex for common issues)
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if re.search(r'\bfunction\s+[a-zA-Z0-9_]+\s*\(', line) and not re.search(r'\}\s*$', code, re.MULTILINE):
                errors.append(ValidationError("Possible unclosed function declaration", line=i))
                logger.error(f"Possible unclosed function at line {i}")

        # 2. Validate game steps
        for step in workflow.steps:
            step_id = step.get('id', 'unknown')
            if step.get('type') == 'game_render':
                if not step.get('scene') or not step.get('render_target'):
                    errors.append(ValidationError("Missing scene or render_target in game_render step", step_id=step_id))
                    logger.error(f"Missing scene or render_target in step {step_id}")
                camera = step.get('camera', {})
                if not camera.get('position') or len(camera.get('position', [])) != 3:
                    errors.append(ValidationError("Invalid camera position in game_render step", step_id=step_id))
                    logger.error(f"Invalid camera position in step {step_id}")
            elif step.get('type') == 'game_input':
                if not step.get('input_type') or not step.get('target'):
                    errors.append(ValidationError("Missing input_type or target in game_input step", step_id=step_id))
                    logger.error(f"Missing input_type or target in step {step_id}")
                if step.get('input_type') not in ['keyboard', 'mouse', 'controller', 'touch', 'vr']:
                    errors.append(ValidationError("Invalid input_type in game_input step", step_id=step_id))
                    logger.error(f"Invalid input_type in step {step_id}")

        return errors

# Registry for validators
validators: Dict[str, Type[CodeValidator]] = {}

def register_validator(language: str, validator_class: Type[CodeValidator]):
    """Register a validator for a specific language."""
    validators[language] = validator_class
    logger.debug(f"Registered validator for {language}")

def get_validator(language: str) -> CodeValidator:
    """Get a validator instance for the specified language."""
    if language not in validators:
        logger.error(f"Unsupported language for validation: {language}")
        raise ValueError(f"Unsupported language for validation: {language}")
    return validators[language]()
