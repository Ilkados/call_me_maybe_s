from typing import Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class ParametersType(BaseModel):
    """Schema for a single function parameter."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "integer", "boolean"]


class ReturnsType(BaseModel):
    """Schema for a function return."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["string", "number", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    """Validated function definition from functions_definition.json."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, ParametersType] = Field(default_factory=dict)
    returns: ReturnsType

    @field_validator("parameters")
    @classmethod
    def validate_param_names(
        cls, value: dict[str, ParametersType]
    ) -> dict[str, ParametersType]:
        """Reject blank or whitespace-only parameter names."""
        for key in value:
            if not key.strip():
                raise ValueError("Parameter names cannot be empty.")
        return value


class InputPrompt(BaseModel):
    """Validated prompt item from function_calling_tests.json."""

    model_config = ConfigDict(extra="forbid")
    prompt: str

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        """Reject empty or whitespace-only prompts early."""
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty")
        return v
