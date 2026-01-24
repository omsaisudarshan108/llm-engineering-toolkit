"""HTTP utilities using httpx."""

import json
import os
from mcp.server.fastmcp import FastMCP
from ..utils import format_error


def register_http_tools(mcp: FastMCP) -> None:
    """Register HTTP tools with the MCP server."""

    @mcp.tool()
    async def http_get(url: str, headers: str = "") -> str:
        """Perform an HTTP GET request.

        Args:
            url: URL to fetch
            headers: Optional JSON object of headers (e.g., '{"Authorization": "Bearer xyz"}')
        """
        try:
            import httpx
        except ImportError:
            return format_error("http_get", "httpx is not installed", "pip install httpx")

        try:
            header_dict = json.loads(headers) if headers else {}
        except json.JSONDecodeError:
            return format_error("http_get", "Invalid headers JSON")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=header_dict)

                result = [
                    f"Status: {response.status_code}",
                    f"URL: {response.url}",
                    "",
                    "Response:",
                    response.text[:5000]  # Limit response size
                ]

                if len(response.text) > 5000:
                    result.append(f"\n... (truncated, {len(response.text)} total bytes)")

                return "\n".join(result)

        except httpx.TimeoutException:
            return format_error("http_get", "Request timed out")
        except httpx.RequestError as e:
            return format_error("http_get", str(e))

    @mcp.tool()
    async def http_post(url: str, body: str = "", headers: str = "") -> str:
        """Perform an HTTP POST request with JSON body.

        Args:
            url: URL to post to
            body: JSON body to send
            headers: Optional JSON object of headers
        """
        try:
            import httpx
        except ImportError:
            return format_error("http_post", "httpx is not installed", "pip install httpx")

        try:
            header_dict = json.loads(headers) if headers else {}
        except json.JSONDecodeError:
            return format_error("http_post", "Invalid headers JSON")

        try:
            body_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return format_error("http_post", "Invalid body JSON")

        # Set content-type if not provided
        if "Content-Type" not in header_dict and "content-type" not in header_dict:
            header_dict["Content-Type"] = "application/json"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=body_data, headers=header_dict)

                result = [
                    f"Status: {response.status_code}",
                    f"URL: {response.url}",
                    "",
                    "Response:",
                    response.text[:5000]
                ]

                if len(response.text) > 5000:
                    result.append(f"\n... (truncated, {len(response.text)} total bytes)")

                return "\n".join(result)

        except httpx.TimeoutException:
            return format_error("http_post", "Request timed out")
        except httpx.RequestError as e:
            return format_error("http_post", str(e))

    @mcp.tool()
    async def download_file(url: str, output_path: str) -> str:
        """Download a file from URL to local path.

        Args:
            url: URL of the file to download
            output_path: Local path to save the file
        """
        try:
            import httpx
        except ImportError:
            return format_error("download_file", "httpx is not installed", "pip install httpx")

        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()

                # Ensure directory exists
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

                with open(output_path, "wb") as f:
                    f.write(response.content)

                size = len(response.content)
                return f"Downloaded {size:,} bytes to: {output_path}"

        except httpx.TimeoutException:
            return format_error("download_file", "Download timed out")
        except httpx.HTTPStatusError as e:
            return format_error("download_file", f"HTTP {e.response.status_code}: {e.response.reason_phrase}")
        except httpx.RequestError as e:
            return format_error("download_file", str(e))

    @mcp.tool()
    async def check_url(url: str) -> str:
        """Check if a URL is reachable and get basic info.

        Args:
            url: URL to check
        """
        try:
            import httpx
        except ImportError:
            return format_error("check_url", "httpx is not installed", "pip install httpx")

        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.head(url)

                result = [
                    f"URL: {url}",
                    f"Status: {response.status_code}",
                    f"Reachable: {'Yes' if response.status_code < 400 else 'No'}",
                ]

                if response.headers.get("content-type"):
                    result.append(f"Content-Type: {response.headers['content-type']}")
                if response.headers.get("content-length"):
                    size = int(response.headers["content-length"])
                    result.append(f"Size: {size:,} bytes")

                return "\n".join(result)

        except httpx.TimeoutException:
            return f"URL: {url}\nStatus: Timeout\nReachable: No"
        except httpx.RequestError as e:
            return f"URL: {url}\nStatus: Error\nReachable: No\nError: {e}"
