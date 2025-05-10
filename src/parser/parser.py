# src/parser/parser.py
import json
from jsonschema import validate, ValidationError

schema = {
    "type": "object",
    "properties": {
        "function": {"type": "string"},
        "metadata": {"type": "object", "properties": {"target_languages": {"type": "array"}}},
        "schema": {"type": "object"},
        "steps": {"type": "array"}
    },
    "required": ["function", "steps"]
}

def parse_workflow(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    try:
        validate(instance=data, schema=schema)
    except ValidationError as e:
        raise ValueError(f"Invalid workflow: {e}")
    return Workflow(data)

# src/engine/workflow.py
class Workflow:
    def __init__(self, data):
        self.function = data['function']
        self.metadata = data.get('metadata', {})
        self.schema = data.get('schema', {})
        self.steps = data['steps']

# src/engine/generator.py
generators = {}

def register_generator(language, generator_class):
    generators[language] = generator_class

def get_generator(language):
    if language not in generators:
        raise ValueError(f"Unsupported language: {language}")
    return generators[language]()

class LanguageGenerator:
    def generate(self, workflow):
        code = [f"# Generated code for {workflow.function}"]
        for step in workflow.steps:
            method = getattr(self, f"generate_{step['type']}", self.generate_default)
            code.append(method(step))
        return "\n".join(code)
    
    def generate_default(self, step):
        return f"# Unsupported step type: {step['type']}"
    
    def generate_set(self, step):
        return f"{step['target']} = {step['value']}"
    
    def generate_if(self, step):
        condition = step['condition']
        then_code = "\n    ".join(self.generate_step(s) for s in step['then'])
        return f"if {condition}:\n    {then_code}"
    
    def generate_step(self, step):
        method = getattr(self, f"generate_{step['type']}", self.generate_default)
        return method(step)

# src/languages/python.py
from engine.generator import LanguageGenerator, register_generator

class PythonGenerator(LanguageGenerator):
    def generate(self, workflow):
        code = ["context = {}", f"# Workflow: {workflow.function}"]
        for key, value in workflow.schema.get('inputs', {}).items():
            code.append(f"context['{key}'] = {repr(value)}")
        code.extend(super().generate_step(step) for step in workflow.steps)
        return "\n".join(code)
    
    def generate_blockchain_operation(self, step):
        if step['chain'] == "ethereum" and step['action'] == "transfer":
            return f"web3.eth.send_transaction({{ 'from': context['sender'], 'to': '{step['params']['to']}', 'value': {step['params']['value']} }})"
        return f"# Unsupported blockchain operation: {step['chain']} {step['action']}"

register_generator('python', PythonGenerator)

# src/languages/solidity.py
from engine.generator import LanguageGenerator, register_generator

class SolidityGenerator(LanguageGenerator):
    def generate(self, workflow):
        code = [
            "pragma solidity ^0.8.0;",
            "contract Workflow {",
            "    address public sender;",
            "    constructor() { sender = msg.sender; }"
        ]
        for step in workflow.steps:
            code.append(f"    {super().generate_step(step)}")
        code.append("}")
        return "\n".join(code)
    
    def generate_set(self, step):
        return f"// Set: {step['target']} = {step['value']};"
    
    def generate_blockchain_operation(self, step):
        if step['chain'] == "ethereum" and step['action'] == "transfer":
            return f"payable(address({step['params']['to']})).transfer({step['params']['value']});"
        return f"// Unsupported blockchain operation"

register_generator('solidity', SolidityGenerator)

# src/cli/main.py
import click
from parser.parser import parse_workflow
from engine.generator import get_generator

@click.group()
def cli():
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

# tests/test_core.py
import pytest
from parser.parser import parse_workflow
from engine.generator import get_generator

def test_parser():
    wf = parse_workflow('examples/simple_workflow.json')
    assert wf.function == "example"

def test_python_generator():
    wf = parse_workflow('examples/simple_workflow.json')
    gen = get_generator('python')
    code = gen.generate(wf)
    assert "context = {}" in code

# examples/simple_workflow.json (content as Python string for inclusion)
simple_workflow = '''{
    "function": "example",
    "metadata": {"target_languages": ["python", "solidity"]},
    "schema": {"inputs": {"x": 10}},
    "steps": [
        {"type": "set", "target": "y", "value": "context['x'] + 5"},
        {"type": "blockchain_operation", "chain": "ethereum", "action": "transfer", "params": {"to": "0x123", "value": "1 ether"}}
    ]
}'''
with open('examples/simple_workflow.json', 'w') as f:
    f.write(simple_workflow)