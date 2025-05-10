import ast
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line
        super().__init__(f"ValidationError: {message}" + (f" at line {line}" if line else ""))

class CodeValidator(ABC):
    """Base class for code validators."""

    @abstractmethod
    def validate(self, code: str, workflow: 'Workflow') -> List[ValidationError]:
        """Validate the generated code and return a list of validation errors."""
        pass

class PythonValidator(CodeValidator):
    """Validator for Python code."""

    def validate(self, code: str, workflow: 'Workflow') -> List[ValidationError]:
        errors = []

        # 1. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(ValidationError(f"Syntax error: {str(e)}", line=e.lineno))
            return errors  # Stop if syntax is invalid

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
            undefined_vars = used_vars - defined_vars - set(workflow.schema.get('inputs', {}).keys())
            for var in undefined_vars:
                if var not in {'context', 'web3'}:  # Allow known globals
                    errors.append(ValidationError(f"Undefined variable: {var}"))
        except Exception as e:
            errors.append(ValidationError(f"Error during variable analysis: {str(e)}"))

        # 3. Validate blockchain operations
        for step in workflow.steps:
            if step.get('type') == 'blockchain_operation' and step.get('chain') == 'ethereum':
                if step.get('action') == 'transfer':
                    params = step.get('params', {})
                    if not params.get('to') or not re.match(r'^0x[a-fA-F0-9]{40}$', params.get('to')):
                        errors.append(ValidationError("Invalid Ethereum address in transfer operation"))
                    if not params.get('value') or not isinstance(params.get('value'), (int, str)):
                        errors.append(ValidationError("Invalid value in transfer operation"))

        return errors

class SolidityValidator(CodeValidator):
    """Validator for Solidity code."""

    def validate(self, code: str, workflow: 'Workflow') -> List[ValidationError]:
        errors = []

        # 1. Basic syntax check (simplified, as full Solidity parsing requires a compiler)
        lines = code.split('\n')
        if not any('pragma solidity' in line for line in lines):
            errors.append(ValidationError("Missing Solidity pragma"))
        if not any('contract Workflow' in line for line in lines):
            errors.append(ValidationError("Missing contract definition"))

        # 2. Check for valid transfer operations
        for step in workflow.steps:
            if step.get('type') == 'blockchain_operation' and step.get('chain') == 'ethereum':
                if step.get('action') == 'transfer':
                    params = step.get('params', {})
                    if not params.get('to') or not re.match(r'^0x[a-fA-F0-9]{40}$', params.get('to')):
                        errors.append(ValidationError("Invalid Ethereum address in transfer operation"))
                    if not params.get('value') or not isinstance(params.get('value'), (int, str)):
                        errors.append(ValidationError("Invalid value in transfer operation"))

        # 3. Check for balanced braces
        brace_count = 0
        for i, line in enumerate(lines, 1):
            brace_count += line.count('{') - line.count('}')
            if brace_count < 0:
                errors.append(ValidationError("Unmatched closing brace", line=i))
        if brace_count != 0:
            errors.append(ValidationError("Unmatched opening brace"))

        return errors

# Registry for validators
validators: Dict[str, type[CodeValidator]] = {}

def register_validator(language: str, validator_class: type[CodeValidator]):
    """Register a validator for a specific language."""
    validators[language] = validator_class

def get_validator(language: str) -> CodeValidator:
    """Get a validator instance for the specified language."""
    if language not in validators:
        raise ValueError(f"Unsupported language for validation: {language}")
    return validators[language]()

# Register validators
register_validator('python', PythonValidator)
register_validator('solidity', SolidityValidator)
