"""Code formatting tools using black, prettier, and sqlfluff."""

import json
import tempfile
import os
from mcp.server.fastmcp import FastMCP
from ..utils import check_command, run_command, format_error


def register_code_tools(mcp: FastMCP) -> None:
    """Register code formatting tools with the MCP server."""

    @mcp.tool()
    async def format_python(code: str, line_length: int = 88) -> str:
        """Format Python code using Black.

        Args:
            code: Python code to format
            line_length: Maximum line length (default: 88)
        """
        try:
            import black
            mode = black.Mode(line_length=line_length)
            formatted = black.format_str(code, mode=mode)
            return formatted
        except ImportError:
            return format_error("format_python", "black is not installed", "pip install black")
        except Exception as e:
            return format_error("format_python", str(e))

    @mcp.tool()
    async def format_js(code: str, parser: str = "babel") -> str:
        """Format JavaScript/TypeScript code using Prettier.

        Args:
            code: JavaScript or TypeScript code to format
            parser: Parser to use (babel, typescript, css, html, json, markdown)
        """
        if not check_command("prettier"):
            return format_error("format_js", "prettier is not installed", "npm install -g prettier")

        # Write to temp file since prettier works best with files
        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            success, output = run_command([
                "prettier",
                "--parser", parser,
                "--write",
                temp_path
            ])

            if not success:
                return format_error("format_js", output)

            with open(temp_path) as f:
                return f.read()
        finally:
            os.unlink(temp_path)

    @mcp.tool()
    async def format_json(json_input: str, indent: int = 2) -> str:
        """Format JSON with consistent indentation.

        Args:
            json_input: JSON string to format
            indent: Number of spaces for indentation (default: 2)
        """
        try:
            parsed = json.loads(json_input)
            return json.dumps(parsed, indent=indent, ensure_ascii=False)
        except json.JSONDecodeError as e:
            return format_error("format_json", f"Invalid JSON: {e}")

    @mcp.tool()
    async def format_sql(sql: str, dialect: str = "ansi") -> str:
        """Format SQL queries using sqlfluff.

        Args:
            sql: SQL query to format
            dialect: SQL dialect (ansi, mysql, postgres, bigquery, snowflake, etc.)
        """
        try:
            import sqlfluff
            result = sqlfluff.fix(sql, dialect=dialect)
            return result
        except ImportError:
            return format_error("format_sql", "sqlfluff is not installed", "pip install sqlfluff")
        except Exception as e:
            return format_error("format_sql", str(e))

    @mcp.tool()
    async def lint_python(code: str) -> str:
        """Check Python code for style issues using Black's check mode.

        Args:
            code: Python code to check
        """
        try:
            import black
            mode = black.Mode()
            try:
                black.format_str(code, mode=mode)
                # If no exception, check if it would change
                formatted = black.format_str(code, mode=mode)
                if formatted == code:
                    return "Code is already properly formatted."
                else:
                    return "Code needs formatting. Use format_python to fix."
            except black.InvalidInput as e:
                return format_error("lint_python", f"Syntax error: {e}")
        except ImportError:
            return format_error("lint_python", "black is not installed", "pip install black")
