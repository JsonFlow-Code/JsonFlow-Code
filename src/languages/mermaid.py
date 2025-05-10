from typing import Dict, Any, Optional
from engine.generator import LanguageGenerator, GeneratorError, register_generator
from engine.workflow import Workflow
from utils.logger import get_logger
from config.config import Config
import re

class MermaidGenerator(LanguageGenerator):
    """Mermaid-specific code generator for JSONFlow workflows, creating flowchart visualizations."""
    
    def __init__(self):
        super().__init__()
        self.logger = get_logger("mermaid_generator")
        self.config = Config()
        self._indent_level = self.config.get("generator.indent_level", 4)
        self._node_counter = 0  # Track unique node IDs
        self._nodes = {}  # Store node IDs for steps
        self._edges = []  # Store flowchart edges
    
    def generate(self, workflow: Workflow) -> str:
        """
        Generate Mermaid flowchart code for a workflow, visualizing steps as nodes and transitions.

        Args:
            workflow: Workflow object containing function, schema, and steps.

        Returns:
            str: Generated Mermaid flowchart code.

        Raises:
            GeneratorError: If code generation fails due to invalid workflow or step.
        """
        self.logger.info(f"Generating Mermaid flowchart for workflow: {workflow.function}")
        
        try:
            # Reset node counter and collections
            self._node_counter = 0
            self._nodes = {}
            self._edges = []
            
            # Initialize code with Mermaid flowchart header
            code = [
                "```mermaid",
                "graph TD",
                f"    Start[Start: {workflow.function}] --> Input",
                "    Input[Initialize Inputs]"
            ]
            
            # Generate nodes for inputs
            inputs = workflow.schema.get('inputs', {})
            for key, spec in inputs.items():
                if not isinstance(spec, dict):
                    self.logger.warning(f"Invalid input spec for {key}")
                    continue
                default = spec.get('default', 'nothing')
                type_str = spec.get('type', 'null')
                node_id = self._get_node_id(f"Input_{key}")
                code.append(f"    {node_id}[{key}: {type_str} = {default}]")
                self._edges.append(f"    Input --> {node_id}")
            
            # Generate nodes and edges for each step
            last_node = "Input"
            for step in workflow.steps:
                last_node = self.generate_step(step, indent=1, parent_node=last_node)
            
            # Add end node
            end_node = self._get_node_id("End")
            code.append(f"    {end_node}[End]")
            self._edges.append(f"    {last_node} --> {end_node}")
            
            # Combine nodes and edges
            code.extend(self._edges)
            code.append("```")
            
            return "\n".join(filter(None, code))  # Remove empty lines
        
        except Exception as e:
            self.logger.error(f"Mermaid code generation failed: {str(e)}")
            raise GeneratorError(f"Mermaid code generation failed: {str(e)}")
    
    def generate_blockchain_operation(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """
        Generate Mermaid nodes and edges for a blockchain_operation step.

        Supports Ethereum transfers; other chains/actions create comment nodes.
        """
        chain = step.get('chain')
        action = step.get('action')
        params = step.get('params', {})
        target = step.get('target', 'result')
        
        node_id = self._get_node_id(f"Blockchain_{self._node_counter}")
        indent_str = " " * (indent * self._indent_level)
        
        if chain == "ethereum" and action == "transfer":
            to_address = params.get('to')
            value = params.get('value')
            if not to_address or not re.match(r'^0x[a-fA-F0-9]{40}$', to_address):
                raise GeneratorError(f"Invalid Ethereum address: {to_address}")
            if not value or not isinstance(value, (int, str, float)):
                raise GeneratorError(f"Invalid value for transfer: {value}")
            node_label = f"Ethereum Transfer<br>to: {to_address}<br>value: {value}<br>target: {target}"
            self._nodes[node_id] = f"{indent_str}{node_id}({node_label})"
        else:
            node_label = f"Unsupported Blockchain<br>{chain} {action}"
            self._nodes[node_id] = f"{indent_str}{node_id}({node_label})"
        
        self._edges.append(f"    {parent_node} --> {node_id}")
        return node_id
    
    def generate_if(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for an if step."""
        condition = self._safe_format_expr(step['condition'])
        indent_str = " " * (indent * self._indent_level)
        
        # Create condition node
        condition_node = self._get_node_id(f"If_{self._node_counter}")
        self._nodes[condition_node] = f"{indent_str}{condition_node}{{Condition: {condition}}}"
        self._edges.append(f"    {parent_node} --> {condition_node}")
        
        # Generate then branch
        then_node = condition_node
        for s in step['then']:
            then_node = self.generate_step(s, indent + 1, then_node)
        self._edges.append(f"    {condition_node} -->|True| {then_node}")
        
        # Generate else branch (if present)
        else_node = condition_node
        if step.get('else'):
            for s in step['else']:
                else_node = self.generate_step(s, indent + 1, else_node)
            self._edges.append(f"    {condition_node} -->|False| {else_node}")
        
        # Create merge node
        merge_node = self._get_node_id(f"Merge_{self._node_counter}")
        self._nodes[merge_node] = f"{indent_str}{merge_node}[Merge]"
        if step.get('else'):
            self._edges.append(f"    {then_node} --> {merge_node}")
            self._edges.append(f"    {else_node} --> {merge_node}")
        else:
            self._edges.append(f"    {then_node} --> {merge_node}")
        
        return merge_node
    
    def generate_set(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for a set step."""
        target = step.get('target')
        if not isinstance(target, str) or not target:
            raise GeneratorError("Invalid or missing target in set step")
        value = self._safe_format_expr(step['value'])
        indent_str = " " * (indent * self._indent_level)
        
        node_id = self._get_node_id(f"Set_{self._node_counter}")
        self._nodes[node_id] = f"{indent_str}{node_id}[Set {target} = {value}]"
        self._edges.append(f"    {parent_node} --> {node_id}")
        return node_id
    
    def generate_return(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for a return step."""
        value = self._safe_format_expr(step['value'])
        indent_str = " " * (indent * self._indent_level)
        
        node_id = self._get_node_id(f"Return_{self._node_counter}")
        self._nodes[node_id] = f"{indent_str}{node_id}[Return {value}]"
        self._edges.append(f"    {parent_node} --> {node_id}")
        return node_id
    
    def generate_try(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for a try step."""
        indent_str = " " * (indent * self._indent_level)
        
        # Create try node
        try_node = self._get_node_id(f"Try_{self._node_counter}")
        self._nodes[try_node] = f"{indent_str}{try_node}[Try]"
        self._edges.append(f"    {parent_node} --> {try_node}")
        
        # Generate body
        body_node = try_node
        for s in step['body']:
            body_node = self.generate_step(s, indent + 1, body_node)
        
        # Generate catch (if present)
        catch = step.get('catch', {})
        catch_node = body_node
        if catch.get('body'):
            error_var = catch.get('error_var', 'e')
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', error_var):
                raise GeneratorError(f"Invalid error variable name: {error_var}")
            catch_node = self._get_node_id(f"Catch_{self._node_counter}")
            self._nodes[catch_node] = f"{indent_str}{catch_node}[Catch {error_var}]"
            self._edges.append(f"    {try_node} -->|Exception| {catch_node}")
            for s in catch['body']:
                catch_node = self.generate_step(s, indent + 1, catch_node)
        
        # Create merge node
        merge_node = self._get_node_id(f"Merge_{self._node_counter}")
        self._nodes[merge_node] = f"{indent_str}{merge_node}[Merge]"
        self._edges.append(f"    {body_node} --> {merge_node}")
        if catch.get('body'):
            self._edges.append(f"    {catch_node} --> {merge_node}")
        
        return merge_node
    
    def generate_while(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for a while step."""
        condition = self._safe_format_expr(step['condition'])
        indent_str = " " * (indent * self._indent_level)
        
        # Create condition node
        condition_node = self._get_node_id(f"While_{self._node_counter}")
        self._nodes[condition_node] = f"{indent_str}{condition_node}{{Condition: {condition}}}"
        self._edges.append(f"    {parent_node} --> {condition_node}")
        
        # Generate body
        body_node = condition_node
        for s in step['body']:
            body_node = self.generate_step(s, indent + 1, body_node)
        self._edges.append(f"    {condition_node} -->|True| {body_node}")
        self._edges.append(f"    {body_node} --> {condition_node}")
        
        # Create exit node
        exit_node = self._get_node_id(f"Exit_{self._node_counter}")
        self._nodes[exit_node] = f"{indent_str}{exit_node}[Exit Loop]"
        self._edges.append(f"    {condition_node} -->|False| {exit_node}")
        
        return exit_node
    
    def generate_foreach(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for a foreach step."""
        collection = self._safe_format_expr(step['collection'])
        iterator = step['iterator']
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', iterator):
            raise GeneratorError(f"Invalid iterator name: {iterator}")
        indent_str = " " * (indent * self._indent_level)
        
        # Create loop node
        loop_node = self._get_node_id(f"ForEach_{self._node_counter}")
        self._nodes[loop_node] = f"{indent_str}{loop_node}[For Each {iterator} in {collection}]"
        self._edges.append(f"    {parent_node} --> {loop_node}")
        
        # Generate body
        body_node = loop_node
        for s in step['body']:
            body_node = self.generate_step(s, indent + 1, body_node)
        self._edges.append(f"    {loop_node} --> {body_node}")
        
        # Create exit node
        exit_node = self._get_node_id(f"Exit_{self._node_counter}")
        self._nodes[exit_node] = f"{indent_str}{exit_node}[Exit Loop]"
        self._edges.append(f"    {body_node} --> {exit_node}")
        
        return exit_node
    
    def generate_assert(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for an assert step."""
        condition = self._safe_format_expr(step['condition'])
        message = step.get('message', 'Assertion failed')
        if not isinstance(message, str):
            raise GeneratorError("Assert message must be a string")
        indent_str = " " * (indent * self._indent_level)
        
        node_id = self._get_node_id(f"Assert_{self._node_counter}")
        self._nodes[node_id] = f"{indent_str}{node_id}[Assert {condition}<br>{message}]"
        self._edges.append(f"    {parent_node} --> {node_id}")
        return node_id
    
    def generate_ai_infer(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for an ai_infer step."""
        model = step.get('model')
        input_data = self._safe_format_expr(step['input'])
        target = step.get('target')
        if not model or not target:
            raise GeneratorError("Missing model or target in ai_infer step")
        indent_str = " " * (indent * self._indent_level)
        
        node_id = self._get_node_id(f"AIInfer_{self._node_counter}")
        self._nodes[node_id] = f"{indent_str}{node_id}[AI Inference<br>model: {model}<br>input: {input_data}<br>target: {target}]"
        self._edges.append(f"    {parent_node} --> {node_id}")
        return node_id
    
    def generate_call_workflow(self, step: Dict[str, Any], indent: int, parent_node: str) -> str:
        """Generate Mermaid nodes and edges for a call_workflow step."""
        workflow_id = step.get('workflow')
        target = step.get('target')
        if not workflow_id or not target:
            raise GeneratorError("Missing workflow or target in call_workflow step")
        args = ", ".join(f"{k}={self._safe_format_expr(v)}" for k, v in step.get('args', {}).items())
        indent_str = " " * (indent * self._indent_level)
        
        node_id = self._get_node_id(f"CallWorkflow_{self._node_counter}")
        self._nodes[node_id] = f"{indent_str}{node_id}[Call Workflow<br>{workflow_id}<br>args: {args}<br>target: {target}]"
        self._edges.append(f"    {parent_node} --> {node_id}")
        return node_id
    
    def _safe_format_expr(self, expr: Any) -> str:
        """
        Format an expression safely for Mermaid labels.

        Args:
            expr: Expression object or value from schema.

        Returns:
            str: Mermaid-compatible expression string.
        """
        formatted = self._format_expr(expr)
        # Convert context.get("key") to context[key]
        formatted = re.sub(
            r"context\.get\(['\"](.*?)['\"]\)",
            r"context[\1]",
            formatted
        )
        # Escape special characters for Mermaid
        formatted = formatted.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
        return formatted
    
    def _get_node_id(self, prefix: str) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"
    
    def _indent_code(self, code: str, indent: int) -> str:
        """Indent code with configured indent level."""
        indent_str = " " * (indent * self._indent_level)
        return "\n".join(indent_str + line for line in code.split("\n") if line.strip())

# Register the generator
register_generator('mermaid', MermaidGenerator)