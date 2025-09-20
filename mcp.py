# mcp.py
from datetime import datetime
import math
import json

class MCPServer:
    """
    Local simulator for MCP-like tool calls. Replace with real tool connectors as needed.
    """

    def __init__(self, enabled=None):
        if enabled is None:
            enabled = {"web_search": True, "calculator": True, "document_store": True}
        self.enabled = enabled

    def call(self, resource: str, query: str):
        if resource == "web_search" and self.enabled.get("web_search", False):
            # simple simulated web search
            return f"Sample web snippet for '{query}' (local demo)."
        if resource == "calculator" and self.enabled.get("calculator", False):
            # try safe calculation (use eval with restricted env)
            try:
                safe_env = {"__builtins__": None, "math": math}
                # only simple arithmetic: disallow letters
                if any(c.isalpha() for c in query):
                    return "Calculator refused: uses only numeric expressions."
                val = eval(query, safe_env)
                return f"Calculation result: {val}"
            except Exception:
                return "Could not calculate."
        if resource == "document_store" and self.enabled.get("document_store", False):
            # simulated document hits
            return f"Found reference to '{query}' in Report.docx (local demo)."
        return f"No data for resource '{resource}'."

    def gather_context(self, query: str):
        parts = []
        for r, ena in self.enabled.items():
            if not ena:
                continue
            parts.append(f"- {r}: {self.call(r, query)}")
        return "MCP resources:\n" + "\n".join(parts)
