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
