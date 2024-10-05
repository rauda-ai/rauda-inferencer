# rauda_inferencer/test_inferencer.py

from unittest.mock import patch, MagicMock

from pydantic import BaseModel
from rauda_inferencer.inferencer import RaudaInferencer
from rauda_inferencer.enums.output_types import OutputType

class TestRaudaInferencer:
    
    @patch('rauda_inferencer.inferencer.OpenAI')
    @patch('rauda_inferencer.inferencer.AzureOpenAI')
    def test_initialization(self, MockAzureOpenAI, MockOpenAI):
        # Test without project name
        inferencer = RaudaInferencer(api_key="test_key")
        MockOpenAI.assert_called_once_with(api_key="test_key")
        assert inferencer.api_key == "test_key"
        assert inferencer.api_version == "2024-08-01-preview"
        
        # Test with project name
        inferencer = RaudaInferencer(api_key="test_key", project_name="test_project")
        MockAzureOpenAI.assert_called_once_with(
            api_key="test_key",
            api_version="2024-08-01-preview",
            azure_endpoint="https://test_project.openai.azure.com"
        )
        assert inferencer.project_name == "test_project"

    @patch('rauda_inferencer.inferencer.OpenAI')
    def test_model_decorator_text_output(self, MockOpenAI):
        mock_openai_instance = MockOpenAI.return_value
        mock_openai_instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
        
        inferencer = RaudaInferencer(api_key="test_key")
        
        @inferencer.model(model="test_model", output_type=OutputType.TEXT)
        def sample_function():
            return "Test prompt"
        
        result = sample_function()
        assert result == "Test response"

    @patch('rauda_inferencer.inferencer.OpenAI')
    def test_model_decorator_json_output(self, MockOpenAI):
        mock_openai_instance = MockOpenAI.return_value
        mock_openai_instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"key": "value"}'))]
        )
        
        inferencer = RaudaInferencer(api_key="test_key")
        
        @inferencer.model(model="test_model", output_type=OutputType.JSON_OBJECT)
        def sample_function():
            return "Test prompt"
        
        result = sample_function()
        assert result == {"key": "value"}

    @patch('rauda_inferencer.inferencer.OpenAI')
    def test_model_decorator_boolean_output(self, MockOpenAI):
        mock_openai_instance = MockOpenAI.return_value
        mock_openai_instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="True"))]
        )
        
        inferencer = RaudaInferencer(api_key="test_key")
        
        @inferencer.model(model="test_model", output_type=OutputType.BOOLEAN)
        def sample_function():
            return "Test prompt"
        
        result = sample_function()
        assert result is True

    @patch('rauda_inferencer.inferencer.OpenAI')
    def test_model_decorator_custom_output(self, MockOpenAI):
        class CustomModel(BaseModel):
            key: str
        
        mock_openai_instance = MockOpenAI.return_value
        mock_openai_instance.beta.chat.completions.parse.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(parsed=CustomModel(key="value")))]
        )
        
        inferencer = RaudaInferencer(api_key="test_key")
        
        @inferencer.model(model="test_model", output_type=CustomModel)
        def sample_function():
            return "Test prompt"
        
        result = sample_function()
        print(result)
        assert result.key == "value"
