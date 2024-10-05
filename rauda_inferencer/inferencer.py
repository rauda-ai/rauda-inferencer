from enum import Enum
import json
import os
from typing import Self
from openai import OpenAI, AzureOpenAI
from functools import wraps
from pydantic import BaseModel
from .enums.output_types import OutputType
import logging


class RaudaInferencer:

    def __init__(
        self,
        api_key: str = None,
        project_name: str = None,
        api_version="2024-08-01-preview",
    ):
        if api_key is None:
            raise ValueError("API key is required")

        self.api_key = api_key
        self.api_version = api_version
        if project_name:
            self.project_name = project_name
            self.openai = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=f"https://{self.project_name}.openai.azure.com",
            )
        else:
            self.openai = OpenAI(api_key=self.api_key)
        
        self.setup_logger()

    def setup_logger(self):
        self.logger = logging.getLogger("rauda-inferencer")
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            ch.setFormatter(formatter)

            self.logger.addHandler(ch)

    def model(
        self, model: str, temperature: int=0.5, output_type: OutputType | BaseModel=OutputType.TEXT
    ):
        """
        Perform inference using the specified model and output type.
        Args:
            model (str): The name of the model to be used for processing (or deployment name, if using Azure OpenAI).
            temperature (int, optional): The temperature setting for the model, which controls the randomness of the output. Defaults to 0.5.
            output_type (OutputType | BaseModel, optional): The expected type of the output. Can be an instance of OutputType or a Pydantic model. Defaults to OutputType.TEXT.
        Returns:
            function: A decorator that processes the output of the decorated function using the specified model and output type.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                prompt = func(*args, **kwargs)

                messages = []
                if func.__doc__:
                    messages.append({"role": "system", "content": func.__doc__})
                messages.append({"role": "user", "content": prompt})

                if output_type == OutputType.JSON_OBJECT:
                    detected_output_type = {"type": "json_object"}
                elif output_type in [OutputType.TEXT, OutputType.BOOLEAN]:
                    detected_output_type = {"type": "text"}
                else:
                    detected_output_type = output_type

                response = self.openai.beta.chat.completions.parse(
                    model=model,
                    response_format=detected_output_type,
                    temperature=temperature,
                    messages=messages,
                )

                self.logger.debug(f"OpenAI Response: {response}")
                self.logger.debug(f"Output type: {output_type}")

                if output_type == OutputType.JSON_OBJECT:
                    self.logger.debug(f"Output type is JSON_OBJECT")    
                    return json.loads(response.choices[0].message.content)

                if output_type == OutputType.BOOLEAN:
                    self.logger.debug(f"Output type is BOOLEAN")
                    return response.choices[0].message.content.strip() == "True"

                if not isinstance(output_type, Enum) and issubclass(
                    output_type, BaseModel
                ):
                    self.logger.debug(f"Output type is not Enum and is subclass of BaseModel")
                    return response.choices[0].message.parsed

                self.logger.debug(f"Output type is TEXT")
                return response.choices[0].message.content

            return wrapper

        return decorator
