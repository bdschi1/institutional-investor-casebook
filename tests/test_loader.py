import pytest
from investor_casebook.data.loader import CasebookLoader

def test_load_sample_cases():
    loader = CasebookLoader()
    # We'll make sure the sample file is in the right place in a second
    cases = loader.load_cases("sample_cases.jsonl")
    assert len(cases) == 2
    assert cases[0]["category"] == "Regulatory"

def test_file_not_found():
    loader = CasebookLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_cases("missing.jsonl")
