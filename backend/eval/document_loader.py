"""评估用文档加载。"""
import logging
import os

from langchain_core.documents import Document

logger = logging.getLogger("ragmate")


def load_langchain_docs(docs_dir: str, max_docs: int = None) -> list[Document]:
    """Load documents from directory as LangChain Document objects.

    Reuses RagMate's existing loaders but skips chunking — RAGAS handles
    its own splitting internally.
    """
    from ingest import load_document, SUPPORTED_EXTENSIONS

    all_files = sorted(
        f for f in os.listdir(docs_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if max_docs:
        all_files = all_files[:max_docs]

    logger.info(f"Loading {len(all_files)} documents from {docs_dir}")

    documents = []
    for filename in all_files:
        filepath = os.path.join(docs_dir, filename)
        try:
            pages = load_document(filepath)
            for page in pages:
                page.metadata["source"] = os.path.basename(page.metadata.get("source", filename))
            documents.extend(pages)
            logger.info(f"  {filename}: {len(pages)} pages")
        except Exception as e:
            logger.warning(f"  Failed to load {filename}: {e}")

    logger.info(f"Loaded {len(documents)} pages total from {len(all_files)} files")
    return documents
