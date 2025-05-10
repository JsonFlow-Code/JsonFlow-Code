import click
import logging
import os
import json
from typing import List, Optional
from src.parser.parser import parse_workflow
from src.engine.generator import get_generator, register_generator
from src.validators.validator import get_validator, ValidationError
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('jsonflow.log')]
)
logger = logging.getLogger(__name__)

# Supported languages
SUPPORTED_LANGUAGES = [
    'python', 'solidity', 'cpp', 'go', 'javascript', 'rust',
    'typescript', 'react', 'perl', 'julia'
]

def initialize_generators():
    """Register all language generators."""
    logger.debug("Initializing generators")
    register_generator("python", PythonGenerator)
    register_generator("cpp", CppGenerator)
    register_generator("go", GoGenerator)
    register_generator("javascript", JavaScriptGenerator)
    register_generator("solidity", SolidityGenerator)
    register_generator("rust", RustGenerator)
    register_generator("typescript", TypeScriptGenerator)
    register_generator("react", ReactGenerator)
    register_generator("perl", PerlGenerator)
    register_generator("julia", JuliaGenerator)

def load_config() -> dict:
    """
    Load configuration from a JSON file (jsonflow.json) if it exists.
    
    Returns:
        dict: Configuration dictionary with default values if file is not found.
    """
    config_path = 'jsonflow.json'
    default_config = {
        'log_level': 'INFO',
        'output_dir': '.',
        'validate_by_default': True
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

@click.group()
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), help='Set logging level')
def cli(log_level: Optional[str]):
    """JSONFlow CLI: Generate, validate, or run code from JSONFlow workflows."""
    config = load_config()
    if log_level:
        config['log_level'] = log_level
    logging.getLogger().setLevel(getattr(logging, config['log_level']))
    initialize_generators()
    logger.info("JSONFlow CLI initialized")

@cli.command()
@click.argument('workflow_files', nargs=-1, type=click.Path(exists=True))
@click.option('--language', required=True, type=click.Choice(SUPPORTED_LANGUAGES), help='Target language for code generation')
@click.option('--output', default=None, help='Output file or directory for generated code')
@click.option('--validate-only', is_flag=True, help='Validate code without generating output')
@click.option('--force', is_flag=True, help='Overwrite existing output files')
def generate(workflow_files: List[str], language: str, output: Optional[str], validate_only: bool, force: bool):
    """
    Generate code from one or more JSONFlow workflow files.
    
    WORKFLOW_FILES: Path(s) to workflow JSON file(s).
    
    Examples:
        jsonflow generate workflow.json --language python --output out.py
        jsonflow generate *.json --language javascript --output ./out --validate-only
    """
    config = load_config()
    output_dir = config['output_dir'] if output else None

    if not workflow_files:
        logger.error("No workflow files provided")
        raise click.ClickException("At least one workflow file is required")

    for workflow_file in workflow_files:
        logger.info(f"Processing workflow: {workflow_file}")
        try:
            workflow = parse_workflow(workflow_file)
            generator = get_generator(language)
            code = generator.generate(workflow)

            # Validate generated code
            validator = get_validator(language)
            errors = validator.validate(code, workflow)
            if errors:
                logger.error(f"Validation failed for {workflow_file}")
                for error in errors:
                    click.echo(f"  {str(error)}", err=True)
                if not validate_only:
                    raise click.ClickException("Code validation failed")
                continue

            if validate_only:
                click.echo(f"Validation passed for {workflow_file}")
                continue

            # Determine output path
            output_path = output
            if output_dir and not output_path:
                filename = f"{os.path.splitext(os.path.basename(workflow_file))[0]}.{language}"
                output_path = os.path.join(output_dir, filename)
            elif not output_path:
                output_path = None

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
@click.option('--language', required=True, type=click.Choice(SUPPORTED_LANGUAGES), help='Language to run (only Python is executable)')
@click.option('--safe', is_flag=True, help='Run Python code in a restricted environment')
def run(workflow_file: str, language: str, safe: bool):
    """
    Generate and run a JSONFlow workflow.
    
    WORKFLOW_FILE: Path to the workflow JSON file.
    
    Examples:
        jsonflow run workflow.json --language python --safe
    """
    logger.info(f"Running workflow: {workflow_file} in {language}")
    try:
        workflow = parse_workflow(workflow_file)
        generator = get_generator(language)
        code = generator.generate(workflow)

        # Validate before running
        validator = get_validator(language)
        errors = validator.validate(code, workflow)
        if errors:
            logger.error(f"Validation failed for {workflow_file}")
            for error in errors:
                click.echo(f"  {str(error)}", err=True)
            raise click.ClickException("Code validation failed")

        if language != 'python':
            click.echo(f"Generated code for {language}:\n{code}")
            click.echo(f"Running for {language} not implemented yet.")
            return

        # Execute Python code
        execution_context = {'context': {}, 'web3': None, 'THREE': None}  # Simplified context
        if safe:
            # Restrict builtins to prevent unsafe operations
            safe_globals = {'__builtins__': {
                'print': print,
                'len': len,
                'dict': dict,
                'list': list,
                'str': str,
                'int': int,
                'float': float
            }}
            execution_context.update(safe_globals)
            logger.debug("Running in safe mode with restricted globals")
        exec(code, execution_context)
        logger.info(f"Successfully executed {workflow_file}")

    except Exception as e:
        logger.error(f"Run failed for {workflow_file}: {str(e)}")
        raise click.ClickException(f"Run failed: {str(e)}")

if __name__ == "__main__":
    cli()
