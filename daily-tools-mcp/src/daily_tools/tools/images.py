"""Image processing tools using ImageMagick."""

import os
from mcp.server.fastmcp import FastMCP
from ..utils import check_command, run_command, format_error


def register_image_tools(mcp: FastMCP) -> None:
    """Register image processing tools with the MCP server."""

    @mcp.tool()
    async def image_resize(
        input_path: str,
        output_path: str,
        width: int = 0,
        height: int = 0,
    ) -> str:
        """Resize an image to specified dimensions.

        Args:
            input_path: Path to input image
            output_path: Path for output image
            width: Target width in pixels (0 to auto-calculate from height)
            height: Target height in pixels (0 to auto-calculate from width)
        """
        if not check_command("convert"):
            return format_error("image_resize", "ImageMagick is not installed", "brew install imagemagick")

        if not os.path.exists(input_path):
            return format_error("image_resize", f"Input file not found: {input_path}")

        if width <= 0 and height <= 0:
            return format_error("image_resize", "Must specify at least width or height")

        # Build resize geometry
        if width > 0 and height > 0:
            geometry = f"{width}x{height}"
        elif width > 0:
            geometry = f"{width}x"
        else:
            geometry = f"x{height}"

        success, output = run_command([
            "convert", input_path,
            "-resize", geometry,
            output_path
        ])

        if not success:
            return format_error("image_resize", output)
        return f"Image resized successfully: {output_path}"

    @mcp.tool()
    async def image_convert(
        input_path: str,
        output_path: str,
        quality: int = 85,
    ) -> str:
        """Convert image between formats (png, jpg, webp, gif).

        Args:
            input_path: Path to input image
            output_path: Path for output image (format determined by extension)
            quality: Output quality for lossy formats (1-100, default 85)
        """
        if not check_command("convert"):
            return format_error("image_convert", "ImageMagick is not installed", "brew install imagemagick")

        if not os.path.exists(input_path):
            return format_error("image_convert", f"Input file not found: {input_path}")

        success, output = run_command([
            "convert", input_path,
            "-quality", str(min(100, max(1, quality))),
            output_path
        ])

        if not success:
            return format_error("image_convert", output)
        return f"Image converted successfully: {output_path}"

    @mcp.tool()
    async def image_compress(
        input_path: str,
        output_path: str,
        quality: int = 75,
    ) -> str:
        """Compress/optimize an image to reduce file size.

        Args:
            input_path: Path to input image
            output_path: Path for output image
            quality: Output quality (1-100, lower = smaller file, default 75)
        """
        if not check_command("convert"):
            return format_error("image_compress", "ImageMagick is not installed", "brew install imagemagick")

        if not os.path.exists(input_path):
            return format_error("image_compress", f"Input file not found: {input_path}")

        success, output = run_command([
            "convert", input_path,
            "-strip",  # Remove metadata
            "-quality", str(min(100, max(1, quality))),
            output_path
        ])

        if not success:
            return format_error("image_compress", output)

        # Report size reduction
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        reduction = ((input_size - output_size) / input_size) * 100

        return f"Image compressed: {output_path}\nSize: {input_size:,} -> {output_size:,} bytes ({reduction:.1f}% reduction)"

    @mcp.tool()
    async def image_info(image_path: str) -> str:
        """Get image metadata (dimensions, format, size, color space).

        Args:
            image_path: Path to image file
        """
        if not check_command("identify"):
            return format_error("image_info", "ImageMagick is not installed", "brew install imagemagick")

        if not os.path.exists(image_path):
            return format_error("image_info", f"File not found: {image_path}")

        success, output = run_command([
            "identify", "-verbose", image_path
        ])

        if not success:
            return format_error("image_info", output)

        # Parse key information
        lines = output.split("\n")
        info = {}
        for line in lines:
            line = line.strip()
            if line.startswith("Format:"):
                info["format"] = line.split(":", 1)[1].strip()
            elif line.startswith("Geometry:"):
                info["dimensions"] = line.split(":", 1)[1].strip()
            elif line.startswith("Colorspace:"):
                info["colorspace"] = line.split(":", 1)[1].strip()
            elif line.startswith("Filesize:"):
                info["filesize"] = line.split(":", 1)[1].strip()

        result = [f"Image: {image_path}"]
        for key, value in info.items():
            result.append(f"  {key}: {value}")

        return "\n".join(result) if info else output
