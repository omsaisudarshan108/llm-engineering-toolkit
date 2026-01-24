"""
Daily Tools MCP Server
======================

An MCP server providing useful daily tools for JSON/YAML processing,
image manipulation, PDF utilities, code formatting, HTTP requests,
and text processing.

Usage:
    uv run daily-tools

Or register with Claude Code:
    claude mcp add daily-tools -- uv run --directory /path/to/daily-tools-mcp daily-tools
"""

from mcp.server.fastmcp import FastMCP

# Import tool registration functions
from .tools.json_yaml import register_json_yaml_tools
from .tools.images import register_image_tools
from .tools.pdf import register_pdf_tools
from .tools.code import register_code_tools
from .tools.http import register_http_tools
from .tools.text import register_text_tools

# Create the MCP server
mcp = FastMCP(
    "daily-tools",
    description="A collection of useful daily tools for developers",
)

# Register all tool categories
register_json_yaml_tools(mcp)
register_image_tools(mcp)
register_pdf_tools(mcp)
register_code_tools(mcp)
register_http_tools(mcp)
register_text_tools(mcp)


def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
