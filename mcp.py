import os
import hashlib
import json
import streamlit as st

class MCPServer:
    def __init__(self, enabled_resources=None):
        if enabled_resources is None:
            enabled_resources = {
                "web_search": True,
                "calculator": True,
                "calendar": False,
                "document_store": True,
            }
        self.resources = enabled_resources

    def call(self, resource, query: str):
        """Simulated MCP calls (replace with real server later)"""
        if resource == "web_search":
            return f"Web search results for '{query}': example.com/info"

        elif resource == "calculator":
            try:
                result = eval(query, {"__builtins__": {}})
                return f"Calculation result: {result}"
            except Exception:
                return "Could not calculate."

        elif resource == "calendar":
            return f"Calendar: No events found for '{query}'."

        elif resource == "document_store":
            return f"Document store: Found reference to '{query}' in Report.docx."

        return "No resource available."

    def gather_context(self, query: str):
        """Aggregate info from enabled MCP resources"""
        context = ""
        for resource, enabled in self.resources.items():
            if enabled:
                context += f"- {resource}: {self.call(resource, query)}\n"
        return context
