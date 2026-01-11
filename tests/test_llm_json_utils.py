from paperjn.llm.json_utils import extract_json_object


def test_extract_json_object_strips_code_fences() -> None:
    text = """```json
{"a": 1, "b": [2, 3]}
```"""
    obj = extract_json_object(text)
    assert obj == {"a": 1, "b": [2, 3]}
