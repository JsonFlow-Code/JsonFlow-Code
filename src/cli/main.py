
import click
import logging
import os
import json
import glob
from typing import List, Optional, Dict, Any
from jsonschema import validate, ValidationError
from restrictedpython import compile_restricted, safe_globals, utility_builtins
from src.parser.parser import parse_workflow
from src.engine.generator import get_generator, register_generator
from src.validators.validator import get_validator, ValidationError as ValidatorError
from src.languages.python import PythonGenerator
from src.languages.cpp import CppGenerator
from src.languages.go import GoGenerator
from src.languages.javascript import JavaScriptGenerator
from src.languages.solidity import SolidityGenerator
from src.languages.rust import RustGenerator
from src.languages.typescript import TypeScriptGenerator
from src.languages.react import ReactGenerator
from src.languages.perl import PerlGenerator
from src.languages.julia import JuliaGenerator
from src.languages.mermaid import MermaidGenerator
from src.languages.qiskit import QiskitGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('jsonflow.log')]
)
logger = logging.getLogger(__name__)

# Supported languages
SUPPORTED_LANGUAGES = [
    'cpp', 'go', 'javascript', 'julia', 'mermaid', 'perl', 'python',
    'qiskit', 'react', 'rust', 'solidity', 'typescript'
]

# Default schema path
SCHEMA_PATH = 'schema.json'

def initialize_generators():
    """Register all language generators."""
    logger.debug("Initializing generators")
    register_generator("cpp", CppGenerator)
    register_generator("go", GoGenerator)
    register_generator("javascript", JavaScriptGenerator)
    register_generator("julia", JuliaGenerator)
    register_generator("mermaid", MermaidGenerator)
    register_generator("perl", PerlGenerator)
    register_generator("python", PythonGenerator)
    register_generator("qiskit", QiskitGenerator)
    register_generator("react", ReactGenerator)
    register_generator("rust", RustGenerator)
    register_generator("solidity", SolidityGenerator)
    register_generator("typescript", TypeScriptGenerator)

def load_config() -> Dict[str, Any]:
    """
    Load configuration from jsonflow.json if it exists.
    
    Returns:
        dict: Configuration dictionary with default values.
    """
    config_path = 'jsonflow.json'
    default_config = {
        'log_level': 'INFO',
        'output_dir': '.',
        'validate_by_default': True,
        'schema_path': SCHEMA_PATH
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.debug(f"Loaded configuration from {config_path}")
            return {**default_config, **config}
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load config: {str(e)}")
    return default_config

def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load the JSONFlow schema.
    
    Args:
        schema_path: Path to the schema file.
    
    Returns:
        dict: Schema dictionary.
    """
    if not os.path.exists(schema_path):
        raise click.ClickException(f"Schema file not found: {schema_path}")
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        validate(instance=schema, schema={"$schema": "http://json-schema.org/draft-07/schema#"})
        logger.debug(f"Loaded schema from {schema_path}")
        return schema
    except (json.JSONDecodeError, ValidationError, IOError) as e:
        raise click.ClickException(f"Failed to load schema: {str(e)}")

@click.group()
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Set logging level')
@click.option('--schema-path', type=click.Path(exists=True), help='Path to JSONFlow schema file')
def cli(log_level: Optional[str], schema_path: Optional[str]):
    """JSONFlow CLI: Generate, validate, run, or manage JSONFlow workflows."""
    config = load_config()
    if log_level:
        config['log_level'] = log_level
    if schema_path:
        config['schema_path'] = schema_path
    logging.getLogger().setLevel(getattr(logging, config['log_level']))
    initialize_generators()
    logger.info("JSONFlow CLI initialized")

@cli.command()
@click.argument('workflow_files', nargs=-1, type=click.Path(exists=True))
@click.option('--language', required=True, type=click.Choice(SUPPORTED_LANGUAGES), help='Target language for code generation')
@click.option('--output', default=None, help='Output file or directory for generated code')
@click.option('--validate-only', is_flag=True, help='Validate workflow without generating output')
@click.option('--force', is_flag=True, help='Overwrite existing output files')
@click.option('--dry-run', is_flag=True, help='Show what would be generated without writing files')
def generate(workflow_files: List[str], language: str, output: Optional[str], validate_only: bool, force: bool, dry_run: bool):
    """
    Generate code from one or more JSONFlow workflow files.
    
    WORKFLOW_FILES: Path(s) to workflow JSON file(s) or glob patterns.
    
    Examples:
        jsonflow generate workflow.json --language python --output out.py
        jsonflow generate workflows/*.json --language javascript --output ./out --force
        jsonflow generate workflow.json --language mermaid --validate-only
    """
    config = load_config()
    schema = load_schema(config['schema_path'])
    output_dir = config['output_dir'] if not output else None

    # Expand glob patterns
    expanded_files = []
    for wf in workflow_files:
        expanded_files.extend(glob.glob(wf))
    if not expanded_files:
        logger.error("No workflow files found")
        raise click.ClickException("No workflow files found")

    for workflow_file in expanded_files:
        logger.info(f"Processing workflow: {workflow_file}")
        try:
            # Parse and validate workflow
            workflow = parse_workflow(workflow_file)
            if config['validate_by_default'] or validate_only:
                try:
                    validate(instance=workflow, schema=schema)
                    validator = get_validator(language)
                    errors = validator.validate(workflow, schema)
                    if errors:
                        logger.error(f"Validation failed for {workflow_file}")
                        for error in errors:
                            click.echo(f"  {str(error)}", err=True)
                        if not validate_only:
                            raise click.ClickException("Validation failed")
                    click.echo(f"Validation passed for {workflow_file}")
                except ValidationError as e:
                    logger.error(f"Schema validation failed for {workflow_file}: {str(e)}")
                    raise click.ClickException(f"Schema validation failed: {str(e)}")

            if validate_only:
                continue

            # Generate code
            generator = get_generator(language)
            code = generator.generate(workflow)

            # Determine output path
            output_path = output
            if output_dir and not output_path:
                ext = 'mmd' if language == 'mermaid' else language
                filename = f"{os.path.splitext(os.path.basename(workflow_file))[0]}.{ext}"
                output_path = os.path.join(output_dir, filename)
            elif not output_path:
                output_path = None

            if dry_run:
                click.echo(f"Would generate code for {workflow_file} ({language}):")
                click.echo(code)
                continue

            if output_path:
                if os.path.exists(output_path) and not force:
                    logger.warning(f"Output file {output_path} exists. Use --force to overwrite.")
                    raise click.ClickException(f"Output file {output_path} exists. Use --force to overwrite.")
                os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(code)
                click.echo(f"Code generated and written to {output_path}")
            else:
                click.echo(code)

        except Exception as e:
            logger.error(f"Failed to process {workflow_file}: {str(e)}")
            raise click.ClickException(f"Failed to process {workflow_file}: {str(e)}")

@cli.command()
@click.argument('workflow_file', type=click.Path(exists=True))
@click.option('--language', required=True, type=click.Choice(['python', 'qiskit']), help='Language to run (Python or Qiskit)')
@click.option('--safe', is_flag=True, help='Run code in a restricted environment')
@click.option('--context', type=str, help='JSON string with initial context for execution')
def run(workflow_file: str, language: str, safe: bool, context: Optional[str]):
    """
    Generate and run a JSONFlow workflow.
    
    WORKFLOW_FILE: Path to the workflow JSON file.
    
    Examples:
        jsonflow run workflow.json --language python --safe
        jsonflow run quantum.json --language qiskit --context '{"qubits": 2}'
    """
    config = load_config()
    schema = load_schema(config['schema_path'])
    logger.info(f"Running workflow: {workflow_file} in {language}")
    try:
        # Parse and validate workflow
        workflow = parse_workflow(workflow_file)
        try:
            validate(instance=workflow, schema=schema)
            validator = get_validator(language)
            errors = validator.validate(workflow, schema)
            if errors:
                logger.error(f"Validation failed for {workflow_file}")
                for error in errors:
                    click.echo(f"  {str(error)}", err=True)
                raise click.ClickException("Validation failed")
        except ValidationError as e:
            logger.error(f"Schema validation failed for {workflow_file}: {str(e)}")
            raise click.ClickException(f"Schema validation failed: {str(e)}")

        # Generate code
        generator = get_generator(language)
        code = generator.generate(workflow)

        # Prepare execution context
        execution_context = {'context': {}}
        if context:
            try:
                execution_context['context'] = json.loads(context)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid context JSON: {str(e)}")

        if safe:
            # Configure restricted environment
            safe_globals_dict = safe_globals.copy()
            safe_globals_dict.update(utility_builtins)
            safe_globals_dict['__builtins__'] = {
                'print': print, 'len': len, 'dict': dict, 'list': list,
                'str': str, 'int': int, 'float': float, 'range': range
            }
            if language == 'qiskit':
                from qiskit import QuantumCircuit, execute
                safe_globals_dict['QuantumCircuit'] = QuantumCircuit
                safe_globals_dict['execute'] = execute
            execution_context.update(safe_globals_dict)
            logger.debug("Running in safe mode with restricted globals")
        else:
            if language == 'qiskit':
                from qiskit import QuantumCircuit, execute
                execution_context['QuantumCircuit'] = QuantumCircuit
                execution_context['execute'] = execute

        # Execute code
        compiled_code = compile_restricted(code, '<workflow>', 'exec') if safe else code
        exec(compiled_code, execution_context)
        logger.info(f"Successfully executed {workflow_file}")

    except Exception as e:
        logger.error(f"Run failed for {workflow_file}: {str(e)}")
        raise click.ClickException(f"Run failed: {str(e)}")

@cli.command()
@click.argument('workflow_files', nargs=-1, type=click.Path(exists=True))
@click.option('--language', type=click.Choice(SUPPORTED_LANGUAGES), help='Validate for a specific language')
def validate(workflow_files: List[str], language: Optional[str]):
    """
    Validate one or more JSONFlow workflow files.
    
    WORKFLOW_FILES: Path(s) to workflow JSON file(s) or glob patterns.
    
    Examples:
        jsonflow validate workflow.json
        jsonflow validate workflows/*.json --language python
    """
    config = load_config()
    schema = load_schema(config['schema_path'])

    # Expand glob patterns
    expanded_files = []
    for wf in workflow_files:
        expanded_files.extend(glob.glob(wf))
    if not expanded_files:
        logger.error("No workflow files found")
        raise click.ClickException("No workflow files found")

    for workflow_file in expanded_files:
        logger.info(f"Validating workflow: {workflow_file}")
        try:
            workflow = parse_workflow(workflow_file)
            validate(instance=workflow, schema=schema)
            if language:
                validator = get_validator(language)
                errors = validator.validate(workflow, schema)
                if errors:
                    logger.error(f"Validation failed for {workflow_file} in {language}")
                    for error in errors:
                        click.echo(f"  {str(error)}", err=True)
                    continue
            click.echo(f"Validation passed for {workflow_file}")
        except ValidationError as e:
            logger.error(f"Schema validation failed for {workflow_file}: {str(e)}")
            click.echo(f"Validation failed for {workflow_file}: {str(e)}", err=True)
        except Exception as e:
            logger.error(f"Validation failed for {workflow_file}: {str(e)}")
            click.echo(f"Validation failed for {workflow_file}: {str(e)}", err=True)

@cli.command()
@click.option('--export', type=click.Path(), help='Export the schema to a file')
@click.option('--validate-schema', is_flag=True, help='Validate the schema itself')
def schema(export: Optional[str], validate_schema: bool):
    """
    Manage the JSONFlow schema.
    
    Examples:
        jsonflow schema --export schema.json
        jsonflow schema --validate-schema
    """
    config = load_config()
    schema = load_schema(config['schema_path'])
    
    if validate_schema:
        try:
            validate(instance=schema, schema={"$schema": "http://json-schema.org/draft-07/schema#"})
            click.echo("Schema is valid")
        except ValidationError as e:
            logger.error(f"Schema validation failed: {str(e)}")
            raise click.ClickException(f"Schema validation failed: {str(e)}")
    
    if export:
        try:
            with open(export, 'w') as f:
                json.dump(schema, f, indent=2)
            click.echo(f"Schema exported to {export}")
        except IOError as e:
            logger.error(f"Failed to export schema: {str(e)}")
            raise click.ClickException(f"Failed to export schema: {str(e)}")

@cli.command()
def list():
    """
    List supported languages and features.
    
    Examples:
        jsonflow list
    """
    click.echo("Supported languages:")
    for lang in SUPPORTED_LANGUAGES:
        click.echo(f"  - {lang}")
    click.echo("\nFeatures:")
    click.echo("  - Code generation for all supported languages")
    click.echo("  - Workflow validation with schema and language-specific checks")
    click.echo("  - Safe execution for Python and Qiskit workflows")
    click.echo("  - Game development support (Unity, Unreal, Godot, custom)")
    click.echo("  - Scripting support with sandboxed Python execution")
    click.echo("  - Quantum computing support with Qiskit")
    click.echo("  - Blockchain operations for Solidity")

if __name__ == "__main__":
    cli()
