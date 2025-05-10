from typing import Any, Dict, List

class Workflow:
    def __init__(self, function: str, schema: Dict[str, Any], steps: List[Dict[str, Any]]):
        self.function = function
        self.schema = schema
        self.steps = steps
