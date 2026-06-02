"""文档管理端点。"""
from fastapi import APIRouter, File, UploadFile

from backend.infrastructure.config import settings
import backend.application.document_service as document_service
from backend.infrastructure.database import async_session
from backend.infrastructure.milvus import get_milvus_client

router = APIRouter()


@router.get("/documents")
async def list_documents():
    async with async_session() as session:
        docs = await document_service.list_documents(settings.DOCUMENTS_DIR, session)
    return {"documents": docs}


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    async with async_session() as session:
        return await document_service.save_document(
            filename=file.filename or "",
            content=content,
            docs_dir=settings.DOCUMENTS_DIR,
            session=session,
        )


@router.delete("/documents/{filename}")
async def delete_document(filename: str):
    async with async_session() as session:
        await document_service.delete_document(
            filename=filename,
            docs_dir=settings.DOCUMENTS_DIR,
            session=session,
            milvus_client=get_milvus_client(),
        )
    return {"success": True}
