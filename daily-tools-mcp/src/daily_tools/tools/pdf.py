"""PDF utilities using poppler-utils."""

import os
from mcp.server.fastmcp import FastMCP
from ..utils import check_command, run_command, format_error


def register_pdf_tools(mcp: FastMCP) -> None:
    """Register PDF tools with the MCP server."""

    @mcp.tool()
    async def pdf_to_text(pdf_path: str, page_start: int = 0, page_end: int = 0) -> str:
        """Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            page_start: First page to extract (0 for first page, default: all pages)
            page_end: Last page to extract (0 for last page, default: all pages)
        """
        if not check_command("pdftotext"):
            return format_error("pdf_to_text", "poppler-utils is not installed", "brew install poppler")

        if not os.path.exists(pdf_path):
            return format_error("pdf_to_text", f"File not found: {pdf_path}")

        args = ["pdftotext"]

        if page_start > 0:
            args.extend(["-f", str(page_start)])
        if page_end > 0:
            args.extend(["-l", str(page_end)])

        args.extend([pdf_path, "-"])  # Output to stdout

        success, output = run_command(args, timeout=60)
        if not success:
            return format_error("pdf_to_text", output)
        return output if output else "No text content found in PDF"

    @mcp.tool()
    async def pdf_info(pdf_path: str) -> str:
        """Get PDF metadata (pages, size, title, author, etc.).

        Args:
            pdf_path: Path to the PDF file
        """
        if not check_command("pdfinfo"):
            return format_error("pdf_info", "poppler-utils is not installed", "brew install poppler")

        if not os.path.exists(pdf_path):
            return format_error("pdf_info", f"File not found: {pdf_path}")

        success, output = run_command(["pdfinfo", pdf_path])
        if not success:
            return format_error("pdf_info", output)
        return output

    @mcp.tool()
    async def pdf_merge(pdf_paths: str, output_path: str) -> str:
        """Merge multiple PDF files into one.

        Args:
            pdf_paths: Comma-separated list of PDF file paths to merge
            output_path: Path for the merged output PDF
        """
        if not check_command("pdfunite"):
            return format_error("pdf_merge", "poppler-utils is not installed", "brew install poppler")

        paths = [p.strip() for p in pdf_paths.split(",")]

        for path in paths:
            if not os.path.exists(path):
                return format_error("pdf_merge", f"File not found: {path}")

        args = ["pdfunite"] + paths + [output_path]
        success, output = run_command(args, timeout=120)

        if not success:
            return format_error("pdf_merge", output)
        return f"Merged {len(paths)} PDFs into: {output_path}"

    @mcp.tool()
    async def pdf_split(
        pdf_path: str,
        output_dir: str,
        page_start: int = 1,
        page_end: int = 0,
    ) -> str:
        """Extract specific pages from a PDF into separate files.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted pages
            page_start: First page to extract (default: 1)
            page_end: Last page to extract (0 for last page)
        """
        if not check_command("pdfseparate"):
            return format_error("pdf_split", "poppler-utils is not installed", "brew install poppler")

        if not os.path.exists(pdf_path):
            return format_error("pdf_split", f"File not found: {pdf_path}")

        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)

        # Get base name for output files
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_pattern = os.path.join(output_dir, f"{base_name}_page_%d.pdf")

        args = ["pdfseparate"]

        if page_start > 0:
            args.extend(["-f", str(page_start)])
        if page_end > 0:
            args.extend(["-l", str(page_end)])

        args.extend([pdf_path, output_pattern])

        success, output = run_command(args, timeout=120)

        if not success:
            return format_error("pdf_split", output)

        # Count created files
        created = [f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith(".pdf")]
        return f"Split PDF into {len(created)} files in: {output_dir}"
