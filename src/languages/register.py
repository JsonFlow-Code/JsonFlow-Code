from engine.generator import register_generator
from languages.python import PythonGenerator
from languages.cpp import CppGenerator
from languages.go import GoGenerator
from languages.javascript import JavaScriptGenerator
from languages.solidity import SolidityGenerator
from languages.rust import RustGenerator
from languages.typescript import TypeScriptGenerator
from languages.react import ReactGenerator
from languages.perl import PerlGenerator
from languages.julia import JuliaGenerator

def initialize_generators():
    register_generator("cpp", CppGenerator)
    register_generator("go", GoGenerator)
    register_generator("javascript", JavaScriptGenerator)
    register_generator("python", PythonGenerator)
    register_generator("solidity", SolidityGenerator)
    register_generator("rust", RustGenerator)
    register_generator("typescript", TypeScriptGenerator)
    register_generator("react", ReactGenerator)
    register_generator("perl", PerlGenerator)
    register_generator("julia", JuliaGenerator)
