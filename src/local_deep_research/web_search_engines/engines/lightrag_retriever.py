"""LangChain BaseRetriever backed by LightRAG /query/data."""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ...security.secure_logging import logger
from ..retriever_registry import retriever_registry
from .lightrag_client import LightRAGClient


class LightRAGRetriever(BaseRetriever):
    """Retrieve document chunks from LightRAG as LangChain Documents."""

    client: LightRAGClient
    mode: str = "mix"
    top_k: int = 60

    def __init__(
        self,
        client: LightRAGClient,
        mode: str = "mix",
        top_k: int = 60,
    ) -> None:
        super().__init__(client=client, mode=mode, top_k=top_k)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        result = self.client.query_data(query, mode=self.mode, top_k=self.top_k)
        if result.get("status") != "success":
            logger.error(f"LightRAG query_data failed: {result.get('message')}")
            return []
        chunks = result.get("data", {}).get("chunks", [])
        return [self._chunk_to_document(chunk) for chunk in chunks]

    @staticmethod
    def _chunk_to_document(chunk: dict) -> Document:
        file_path = chunk.get("file_path", "")
        title = Path(file_path).name if file_path else ""
        return Document(
            page_content=chunk.get("content", ""),
            metadata={
                "source": file_path,
                "title": title,
                "reference_id": chunk.get("reference_id", ""),
                "chunk_id": chunk.get("chunk_id", ""),
            },
        )


def register_lightrag_retriever(
    client: LightRAGClient,
    name: str = "lightrag",
    username: Optional[str] = None,
) -> None:
    """Register a LightRAGRetriever in the shared (or per-user) registry."""
    retriever_registry.register(
        name, LightRAGRetriever(client=client), is_local=True, username=username
    )
