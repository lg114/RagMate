class AppError(Exception):
    """所有业务错误的基类。操作型错误会返回给客户端，非操作型错误记录日志后返回 500。"""

    def __init__(self, message: str, code: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class NotFoundError(AppError):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            message=f"{resource} not found: {identifier}",
            code="NOT_FOUND",
            status_code=404,
        )


class ValidationError(AppError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
        )


class ConflictError(AppError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=409,
        )


class ServiceUnavailableError(AppError):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="SERVICE_UNAVAILABLE",
            status_code=503,
        )


class RetrievalError(AppError):
    def __init__(self, message: str = "检索服务异常，请稍后重试"):
        super().__init__(
            message=message,
            code="RETRIEVAL_ERROR",
            status_code=503,
        )
