"""Text processing utilities."""

import base64
import hashlib
import re
import difflib
import os
from mcp.server.fastmcp import FastMCP
from ..utils import format_error


def register_text_tools(mcp: FastMCP) -> None:
    """Register text processing tools with the MCP server."""

    @mcp.tool()
    async def base64_encode(text: str = "", file_path: str = "") -> str:
        """Encode text or file contents to base64.

        Args:
            text: Text to encode (if no file_path provided)
            file_path: Path to file to encode (takes precedence over text)
        """
        try:
            if file_path:
                if not os.path.exists(file_path):
                    return format_error("base64_encode", f"File not found: {file_path}")
                with open(file_path, "rb") as f:
                    data = f.read()
            else:
                data = text.encode("utf-8")

            encoded = base64.b64encode(data).decode("ascii")
            return encoded
        except Exception as e:
            return format_error("base64_encode", str(e))

    @mcp.tool()
    async def base64_decode(encoded: str, output_path: str = "") -> str:
        """Decode base64 to text or save to file.

        Args:
            encoded: Base64 encoded string
            output_path: Optional file path to save decoded content (for binary files)
        """
        try:
            decoded = base64.b64decode(encoded)

            if output_path:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(decoded)
                return f"Decoded {len(decoded):,} bytes to: {output_path}"
            else:
                return decoded.decode("utf-8")
        except base64.binascii.Error as e:
            return format_error("base64_decode", f"Invalid base64: {e}")
        except UnicodeDecodeError:
            return format_error("base64_decode", "Binary content detected. Use output_path to save as file.")
        except Exception as e:
            return format_error("base64_decode", str(e))

    @mcp.tool()
    async def hash_text(text: str, algorithm: str = "sha256") -> str:
        """Generate hash of text using specified algorithm.

        Args:
            text: Text to hash
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        """
        algorithms = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }

        if algorithm.lower() not in algorithms:
            return format_error("hash_text", f"Unknown algorithm: {algorithm}. Use: md5, sha1, sha256, sha512")

        hasher = algorithms[algorithm.lower()]()
        hasher.update(text.encode("utf-8"))
        return f"{algorithm.upper()}: {hasher.hexdigest()}"

    @mcp.tool()
    async def hash_file(file_path: str, algorithm: str = "sha256") -> str:
        """Generate hash of file contents.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        """
        algorithms = {
            "md5": hashlib.md5,
            "sha1": hashlib.sha1,
            "sha256": hashlib.sha256,
            "sha512": hashlib.sha512,
        }

        if algorithm.lower() not in algorithms:
            return format_error("hash_file", f"Unknown algorithm: {algorithm}. Use: md5, sha1, sha256, sha512")

        if not os.path.exists(file_path):
            return format_error("hash_file", f"File not found: {file_path}")

        hasher = algorithms[algorithm.lower()]()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return f"{algorithm.upper()}: {hasher.hexdigest()}"

    @mcp.tool()
    async def word_count(text: str) -> str:
        """Count words, lines, and characters in text.

        Args:
            text: Text to analyze
        """
        lines = text.split("\n")
        words = text.split()
        chars = len(text)
        chars_no_space = len(text.replace(" ", "").replace("\n", ""))

        return f"""Lines: {len(lines)}
Words: {len(words)}
Characters: {chars:,}
Characters (no spaces): {chars_no_space:,}"""

    @mcp.tool()
    async def regex_extract(text: str, pattern: str, group: int = 0) -> str:
        """Extract all matches of a regex pattern from text.

        Args:
            text: Text to search
            pattern: Regular expression pattern
            group: Capture group to extract (0 for full match)
        """
        try:
            regex = re.compile(pattern)
            matches = []

            for match in regex.finditer(text):
                try:
                    matches.append(match.group(group))
                except IndexError:
                    return format_error("regex_extract", f"Group {group} not found in pattern")

            if not matches:
                return "No matches found"

            return f"Found {len(matches)} match(es):\n" + "\n".join(matches)

        except re.error as e:
            return format_error("regex_extract", f"Invalid regex: {e}")

    @mcp.tool()
    async def regex_replace(text: str, pattern: str, replacement: str) -> str:
        """Replace all matches of a regex pattern in text.

        Args:
            text: Text to modify
            pattern: Regular expression pattern
            replacement: Replacement string (can use \\1, \\2 for groups)
        """
        try:
            result = re.sub(pattern, replacement, text)
            return result
        except re.error as e:
            return format_error("regex_replace", f"Invalid regex: {e}")

    @mcp.tool()
    async def diff_text(text1: str, text2: str, context_lines: int = 3) -> str:
        """Show unified diff between two texts.

        Args:
            text1: First text (original)
            text2: Second text (modified)
            context_lines: Number of context lines around changes
        """
        lines1 = text1.splitlines(keepends=True)
        lines2 = text2.splitlines(keepends=True)

        diff = difflib.unified_diff(
            lines1, lines2,
            fromfile="original",
            tofile="modified",
            n=context_lines
        )

        result = "".join(diff)
        return result if result else "No differences found"
