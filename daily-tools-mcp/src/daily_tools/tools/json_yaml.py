"""JSON and YAML processing tools using jq and yq."""

import json
from mcp.server.fastmcp import FastMCP
from ..utils import check_command, run_command, format_error


def register_json_yaml_tools(mcp: FastMCP) -> None:
    """Register JSON/YAML tools with the MCP server."""

    @mcp.tool()
    async def jq_query(json_input: str, query: str) -> str:
        """Query and transform JSON data using jq syntax.

        Args:
            json_input: JSON string to query
            query: jq query expression (e.g., '.data[] | .name', '.users | length')
        """
        if not check_command("jq"):
            return format_error("jq_query", "jq is not installed", "brew install jq")

        success, output = run_command(["jq", query], input_data=json_input)
        if not success:
            return format_error("jq_query", output)
        return output

    @mcp.tool()
    async def yq_query(yaml_input: str, query: str) -> str:
        """Query and transform YAML data using yq syntax.

        Args:
            yaml_input: YAML string to query
            query: yq query expression (e.g., '.data[].name', '.spec.containers')
        """
        if not check_command("yq"):
            return format_error("yq_query", "yq is not installed", "brew install yq")

        success, output = run_command(["yq", query], input_data=yaml_input)
        if not success:
            return format_error("yq_query", output)
        return output

    @mcp.tool()
    async def json_to_yaml(json_input: str) -> str:
        """Convert JSON to YAML format.

        Args:
            json_input: JSON string to convert
        """
        if not check_command("yq"):
            return format_error("json_to_yaml", "yq is not installed", "brew install yq")

        success, output = run_command(["yq", "-P", "-o", "yaml"], input_data=json_input)
        if not success:
            return format_error("json_to_yaml", output)
        return output

    @mcp.tool()
    async def yaml_to_json(yaml_input: str) -> str:
        """Convert YAML to JSON format.

        Args:
            yaml_input: YAML string to convert
        """
        if not check_command("yq"):
            return format_error("yaml_to_json", "yq is not installed", "brew install yq")

        success, output = run_command(["yq", "-o", "json"], input_data=yaml_input)
        if not success:
            return format_error("yaml_to_json", output)
        return output

    @mcp.tool()
    async def json_format(json_input: str, minify: bool = False) -> str:
        """Pretty-print or minify JSON.

        Args:
            json_input: JSON string to format
            minify: If True, output compact JSON; if False, pretty-print with indentation
        """
        try:
            parsed = json.loads(json_input)
            if minify:
                return json.dumps(parsed, separators=(",", ":"))
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            return format_error("json_format", f"Invalid JSON: {e}")
