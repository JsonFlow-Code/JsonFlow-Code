import click
from parser.parser import parse_workflow
from engine.generator import get_generator
from languages.python import register_python_generator
from languages.solidity import register_solidity_generator

# Initialize generators and validators
def initialize_generators():
    register_python_generator()  # Registers both generator and validator
    register_solidity_generator()

@click.group()
def cli():
    initialize_generators()  # Call before commands
    pass

@cli.command()
@click.argument('workflow_file')
@click.option('--language', required=True)
@click.option('--output', default=None)
def generate(workflow_file, language, output):
    workflow = parse_workflow(workflow_file)
    generator = get_generator(language)
    code = generator.generate(workflow)
    if output:
        with open(output, 'w') as f:
            f.write(code)
    else:
        print(code)

@cli.command()
@click.argument('workflow_file')
@click.option('--language', required=True)
def run(workflow_file, language):
    workflow = parse_workflow(workflow_file)
    generator = get_generator(language)
    code = generator.generate(workflow)
    if language == 'python':
        exec(code)
    else:
        print(f"Generated code for {language}:\n{code}")
        print(f"Running for {language} not implemented yet.")

if __name__ == "__main__":
    cli()