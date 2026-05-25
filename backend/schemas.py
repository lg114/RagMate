"""Pydantic 请求/响应模型。"""
import re

from pydantic import BaseModel, Field, field_validator

from errors import ValidationError

_UUID_RE = re.compile(r'^[a-f0-9-]{36}$')


def validate_session_id(v: str) -> str:
    if not _UUID_RE.match(v):
        raise ValidationError("Invalid session_id format")
    return v


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000)
    session_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def _validate(cls, v):
        if v is not None:
            validate_session_id(v)
        return v


class ChatResponse(BaseModel):
    response: str
    session_id: str
