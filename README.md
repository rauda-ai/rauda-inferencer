# Rauda Inferencer

Rauda Inferencer is a simple inferencing engine designed to work with OpenAI and Azure OpenAI models. It provides a convenient way to define and use models with different output types.

# Usage

```python
from rauda_inferencer import RaudaInferencer, OutputType

# Initialize with OpenAI API key
oai_inference = RaudaInferencer(api_key="your_openai_api_key")

# Initialize with Azure OpenAI API key and project name
azure_inference = RaudaInferencer(
    project_name="your_project_name", api_key="your_azure_openai_api_key"
)
```
You can then define functions with the @model decorator to specify the model and output type:

### Simple text output
```python
@oai_inference.model("gpt-4o-mini", output_type=OutputType.TEXT)
def output_text_greeting(name: str) -> str:
    """You're a greetings expert. Greet the person in the most pompous way you know of."""
    return f"Hello! My name is {name}"
```

### JSON / Python dictionary output (Azure OpenAI example)
```python
@azure_inference.model("regional-eu-gpt-4o", output_type=OutputType.JSON_OBJECT)
def output_json_greeting_json(name: str) -> Dict:
    """You're a helpful assistant. Greet the person in a friendly way. Return the response in JSON"""
    return f"Hello! My name is {name}, how are you?"
```

### Boolean output
```python
@oai_inference.model("gpt-4o-mini", output_type=OutputType.BOOLEAN)
def output_boolean(name: str) -> bool:
    """If the user's name is 'Rauda', return True. Otherwise, return False."""
    return name == "Rauda"
```

### Pydantic model (using Structured Objects, available in GPT-4o and 4o-mini only)
```python
class CustomModel(BaseModel):
    key: str

@oai_inference.model("gpt-4o-mini", output_type=CustomModel)
def output_custom_model(name: str) -> CustomModel:
    """Return a custom model with a key."""
    return CustomModel(key=f"Hello, {name}")
```
