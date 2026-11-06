"""MCP tool server for a (synthetic) civic services office. Run: python -m civic_agent.mcp_server

The server re-checks permissions itself (defence in depth): even a client that skips the agent's
policy layer cannot read staff documents as a citizen or see another citizen's request.
"""

from __future__ import annotations

import itertools
import json

from mcp.server.fastmcp import FastMCP

from civic_agent.retrieval import HybridIndex, load_documents

SERVICES = {"BIRTH-REG", "ID-RENEW", "DOMICILE", "BIZ-LICENSE", "PROP-TAX", "DL-RENEW", "PENSION-ENQ", "COMPLAINT"}


def build_server(index: HybridIndex | None = None) -> FastMCP:
    mcp = FastMCP("civic-services", log_level="WARNING")
    idx = index or HybridIndex(load_documents())
    requests = {
        "REQ-1001": {"owner": "citizen-1", "service": "ID-RENEW", "status": "in progress, photo verified"},
        "REQ-1002": {"owner": "citizen-2", "service": "DOMICILE", "status": "ready for collection"},
    }
    next_id = itertools.count(2001)

    @mcp.tool()
    def search_documents(query: str, role: str = "citizen", requester_id: str = "", k: int = 3) -> str:
        """Search official service documents. Returns the best matches with their document ids for citation."""
        hits = idx.search(query, role, k=max(1, min(k, 5)))
        return json.dumps([{"id": h.doc.id, "title": h.doc.title, "text": h.doc.text} for h in hits], ensure_ascii=False)

    @mcp.tool()
    def get_request_status(request_id: str, role: str = "citizen", requester_id: str = "") -> str:
        """Look up the status of a service request by its id (for example REQ-1001)."""
        req = requests.get(request_id.strip().upper())
        if req is None or (role != "caseworker" and req["owner"] != requester_id):
            # Same answer for "missing" and "not yours", so ids cannot be probed.
            return json.dumps({"error": f"No request {request_id} found for this user."})
        return json.dumps({"request_id": request_id.upper(), "service": req["service"], "status": req["status"]})

    @mcp.tool()
    def submit_service_request(service: str, details: str, role: str = "citizen", requester_id: str = "") -> str:
        """Submit a new service request (a write action) for the current citizen. Service is a code such as ID-RENEW."""
        if role != "citizen" or not requester_id:
            return json.dumps({"error": "Only a signed-in citizen can submit a request."})
        code = service.strip().upper()
        if code not in SERVICES:
            return json.dumps({"error": f"Unknown service {service}. Valid codes: {', '.join(sorted(SERVICES))}"})
        rid = f"REQ-{next(next_id)}"
        requests[rid] = {"owner": requester_id, "service": code, "status": "received"}
        return json.dumps({"request_id": rid, "service": code, "status": "received"})

    return mcp


if __name__ == "__main__":
    build_server().run()
