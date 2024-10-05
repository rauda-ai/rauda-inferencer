import os
from typing import Dict
from pydantic import BaseModel
from rauda_inferencer import RaudaInferencer, OutputType


class Greeting(BaseModel):
    greeting: str


oai_inference = RaudaInferencer(api_key=os.getenv("OPENAI_API_KEY"))
azure_inference = RaudaInferencer(
    project_name="rauda-openai-sweden", api_key=os.getenv("AZURE_OPENAI_API_KEY")
)


@oai_inference.model("gpt-4o-mini", output_type=OutputType.TEXT)
def output_text_greeting(name: str) -> str:
    # Docstring is the system prompt
    """You're a greetings expert. Greet the person in the most pompous way you know of."""

    # Return is the user prompt
    return f"Hello! My name is {name}"


@azure_inference.model("regional-eu-gpt-4o", output_type=OutputType.JSON_OBJECT)
def output_json_greeting_json(name: str) -> Dict:
    # Docstring is the system prompt
    """You're a helpful assistant. Greet the person in a friendly way. Return the response in JSON"""

    # Return is the user prompt
    return f"Hello! My name is {name}, how are you?"

@oai_inference.model("gpt-4o-mini", output_type=OutputType.BOOLEAN)
def output_boolean(name: str) -> bool:
    """If the user's name is 'Rauda', return True. Otherwise, return False."""

    return f"My name is {name}"

@azure_inference.model("regional-eu-gpt-4o", output_type=Greeting)
def output_greeting_pydantic_model(name: str) -> Greeting:
    """You're a helpful assistant. Greet the person in a friendly way"""

    return f"Hello! My name is {name}, how are you?"

text_output = output_text_greeting("Rauda")
print(f"Text output: {text_output}")
print(text_output)
json_output = output_json_greeting_json("Rauda")
print(f"JSON output: {json_output}")
boolean_output = output_boolean("Rauda")
print(f"Boolean output: {boolean_output}")
pydantic_output = output_greeting_pydantic_model("Rauda")
print(f"Pydantic model output: {pydantic_output}")
