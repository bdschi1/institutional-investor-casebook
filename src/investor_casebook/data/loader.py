import json
from pathlib import Path

class CasebookLoader:
    def __init__(self, data_path=None):
        # Default to the directory where this file lives if no path provided
        self.data_path = Path(data_path) if data_path else Path(__file__).parent
            
    def load_cases(self, filename):
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        cases = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    cases.append(json.loads(line))
        return cases
