import pytest
from parser.parser import parse_workflow
from engine.generator import get_generator
from languages.python import register_python_generator

@pytest.fixture(autouse=True)
def setup_generators():
    register_python_generator()

def test_parser():
    wf = parse_workflow('examples/simple_workflow.json')
    assert wf.function == "example"

def test_python_generator():
    wf = parse_workflow('examples/simple_workflow.json')
    gen = get_generator('python')
    code = gen.generate(wf)
    assert "context = {}" in code