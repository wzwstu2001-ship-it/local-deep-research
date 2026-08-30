"""Smoke-test the LightRAG service through LDR's HTTP client.

Verifies the three integration seams the LightRAG switch depends on, in
isolation (no LDR web app / DB required):

  1. ``POST /documents/text`` returns a ``track_id``
  2. ``GET /documents/track_status/{track_id}`` reaches a terminal state
  3. ``POST /query/data`` returns the inserted text back

Usage::

    python scripts/lightrag_smoke.py
    LDR_LIGHTRAG_BASE_URL=http://127.0.0.1:9621 python scripts/lightrag_smoke.py

Exit code 0 = all three passed; 1 = a step failed. The LightRAG service must
be running first (``HOST=127.0.0.1 PORT=9621`` by default).
"""

import os
import sys
import time
import uuid

DEFAULT_BASE_URL = "http://127.0.0.1:9621"

# Terminal states of LightRAG's DocStatus enum (lightrag/base.py).
_TERMINAL_STATUSES = ("processed", "failed")


def main() -> int:
    base_url = os.environ.get("LDR_LIGHTRAG_BASE_URL") or DEFAULT_BASE_URL

    from local_deep_research.web_search_engines.engines.lightrag_client import (
        LightRAGClient,
        LightRAGUnavailableError,
    )

    print(f"[1/3] connecting to LightRAG at {base_url} ...")
    client = LightRAGClient(base_url=base_url)
    try:
        # 1) Insert under a unique file_source AND a unique content marker so a
        #    re-run avoids both the filename 409 and LightRAG's content-hash
        #    dedup (which would otherwise mark the doc DUPLICATE + FAILED).
        marker = uuid.uuid4().hex
        payload = f"lightrag smoke test {marker}: 深度学习是机器学习的一个子领域"
        file_source = f"lightrag_smoke_{marker[:12]}"
        r = client.insert_text(payload, file_source=file_source)
        print(f"      insert -> {r}")
        track_id = r.get("track_id")
        if not track_id:
            print(
                "FAIL: insert response carried no 'track_id' "
                "(check the /documents/text response shape)."
            )
            return 1

        # 2) Poll track status until every document reaches a terminal state.
        print(f"[2/3] polling track status for track_id={track_id}")
        deadline = time.time() + 120
        final_summary = None
        first = True
        while time.time() < deadline:
            s = client.track_status(track_id)
            if first:
                print(f"      full track_status response -> {s}")
                first = False
            docs = s.get("documents") or []
            statuses = [d.get("status") for d in docs]
            print(f"      document statuses -> {statuses!r}")
            if docs and all(st in _TERMINAL_STATUSES for st in statuses):
                final_summary = s.get("status_summary")
                break
            time.sleep(2)

        if final_summary is None:
            print("FAIL: documents did not reach a terminal state within 120s")
            return 1
        if "failed" in final_summary or "DocStatus.FAILED" in final_summary:
            print(f"FAIL: a document ended in FAILED: {final_summary}")
            return 1

        # 3) Query back for the inserted text.
        print("[3/3] querying back for the inserted text ...")
        q = client.query_data("深度学习 机器学习", mode="mix", top_k=20)
        print(f"      query -> {q}")
        if q.get("status") != "success":
            print(
                f"FAIL: query_data returned status={q.get('status')!r}: "
                f"{q.get('message')}"
            )
            return 1
        chunks = (q.get("data") or {}).get("chunks", [])
        joined = " ".join(c.get("content", "") for c in chunks)
        if "深度学习" in joined or "机器学习" in joined:
            print("PASS: inserted text was retrieved back.")
            return 0
        print("FAIL: inserted text was not found in the query results")
        return 1
    except LightRAGUnavailableError as exc:
        print(f"FAIL: LightRAG service unreachable or error: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
