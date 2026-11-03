"""Pydantic 请求/响应模型。"""
import re

from pydantic import BaseModel, Field, field_validator

from backend.domain.errors import ValidationError

# 标准 UUID 格式：8-4-4-4-12，版本位(第13位)为 1-5，变体位(第17位)为 89ab
_UUID_RE = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$')


def validate_session_id(v: str) -> str:
    if not _UUID_RE.match(v):
        raise ValidationError("Invalid session_id format")
    return v


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=10000)
    session_id: str | None = None
    replace_last: bool = False  # 重试/重新生成时，替换 Redis 中的最后一条用户消息
    debug: bool = False  # 为 true 时响应带上检索候选、分数和丢弃原因；默认不返回内部结构

    @field_validator("session_id")
    @classmethod
    def _validate(cls, v):
        if v is not None:
            validate_session_id(v)
        return v


class ChatResponse(BaseModel):
    response: str
    session_id: str
    # 检索置信度：{"level": "high|medium|low", "score": float, "chunks": int}
    # 无检索发生（如简单查询直连 LLM）时为 None
    confidence: dict | None = None
    # 忠实度校验未通过的声明：[{"claim": str, "supported": false}, ...]
    # 默认开启 FAITHFULNESS_CHECK；未校验、全部通过、或已拒答时为 None
    unsupported_claims: list[dict] | None = None
    # 实际检索到的来源：[{"source": "文件名", "page": 1}, ...]，页码从 1 开始
    citations: list[dict] | None = None
    # refused=无依据拒答；hedged=低置信保留；grounded=正常 RAG 回答
    grounding: str | None = None
    # 仅当请求 debug=true 时填充：[{query, chunks: [{score, kept, dropped_by, preview}, ...]}]
    retrieval_debug: list[dict] | None = None


class IngestRequest(BaseModel):
    filenames: list[str] = Field(default_factory=list, max_length=200)

    @field_validator("filenames", mode="before")
    @classmethod
    def _validate_filenames(cls, v):
        if not v:
            return v
        for name in v:
            if not name or len(name) > 255:
                raise ValidationError(f"Invalid filename: {name!r}")
            if any(c in name for c in ('/', '\\', '\x00')):
                raise ValidationError(f"Filename contains invalid characters: {name!r}")
        return v
