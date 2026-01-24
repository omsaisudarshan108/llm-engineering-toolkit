"""Shared utilities for Daily Tools MCP Server."""

import shutil
import subprocess
from typing import Optional


def check_command(command: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(command) is not None


def run_command(
    args: list[str],
    input_data: Optional[str] = None,
    timeout: int = 30,
) -> tuple[bool, str]:
    """
    Run a shell command and return the result.

    Args:
        args: Command and arguments as list
        input_data: Optional stdin input
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    try:
        result = subprocess.run(
            args,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            return False, result.stderr.strip() or f"Command failed with exit code {result.returncode}"

        return True, result.stdout.strip()

    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except FileNotFoundError:
        return False, f"Command not found: {args[0]}"
    except Exception as e:
        return False, f"Error running command: {str(e)}"


def format_error(tool_name: str, message: str, install_hint: Optional[str] = None) -> str:
    """Format an error message consistently."""
    error = f"Error in {tool_name}: {message}"
    if install_hint:
        error += f"\n\nTo install: {install_hint}"
    return error
