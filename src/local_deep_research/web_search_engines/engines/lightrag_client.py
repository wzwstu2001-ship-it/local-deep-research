"""HTTP client for the LightRAG service (document insert, query, track status)."""

from typing import Any, Dict, Optional

import httpx

from ...security.secure_logging import logger

DEFAULT_LIGHTRAG_BASE_URL = "http://127.0.0.1:9621"
DEFAULT_TIMEOUT = 300.0


class LightRAGUnavailableError(RuntimeError):
    """Raised when the LightRAG service cannot be reached or returns an error."""


class LightRAGClient:
    """Thin HTTP wrapper over the LightRAG REST API used by LDR."""

    def __init__(
        self,
        base_url: str = DEFAULT_LIGHTRAG_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            resp = self._client.post(path, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.error(f"LightRAG request {path} failed: {exc}")
            raise LightRAGUnavailableError(
                f"LightRAG request {path} failed: {exc}"
            ) from exc

    def insert_text(
        self, text: str, file_source: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._post("/documents/text", {"text": text, "file_source": file_source})

    def query(
        self, query: str, mode: str = "mix", top_k: int = 60
    ) -> Dict[str, Any]:
        return self._post("/query", {"query": query, "mode": mode, "top_k": top_k})

    def query_data(
        self, query: str, mode: str = "mix", top_k: int = 60
    ) -> Dict[str, Any]:
        return self._post("/query/data", {"query": query, "mode": mode, "top_k": top_k})

    def track_status(self, track_id: str) -> Dict[str, Any]:
        """Get document processing status by the track_id returned from insert.

        ``/documents/track_status/{track_id}`` reports the processing status of
        the documents inserted under ``track_id`` (from ``/documents/text``,
        ``/texts`` or ``/upload``). Distinct from ``/documents/scan/status``,
        which tracks *scan* jobs.
        """
        try:
            resp = self._client.get(f"/documents/track_status/{track_id}")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError as exc:
            logger.error(f"LightRAG track_status {track_id} failed: {exc}")
            raise LightRAGUnavailableError(
                f"LightRAG track_status {track_id} failed: {exc}"
            ) from exc
