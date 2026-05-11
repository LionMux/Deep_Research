#!/usr/bin/env python3
"""Entry point: python -m scirag runs the MCP server."""

from .mcp_server import mcp

if __name__ == "__main__":
    mcp.run(transport="stdio")
