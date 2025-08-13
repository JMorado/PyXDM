"""Logging utilities for PyXDM."""

import sys
import contextlib


from typing import Generator
import io

@contextlib.contextmanager
def suppress_horton_output(show_warnings: bool = True) -> Generator[io.StringIO, None, None]:
    """
    Context manager to suppress Horton output.

    Parameters
    ----------
    show_warnings : bool, optional
        Whether to show warnings, default True

    Yields
    ------
    io.StringIO
        Captured output buffer
    """
    # Store original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create a string buffer to capture output
    captured_output = io.StringIO()

    try:
        # Redirect stdout/stderr
        sys.stdout = captured_output
        if not show_warnings:
            sys.stderr = captured_output

        yield captured_output

    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def info(message: str, indent: int = 0) -> None:
    """
    Print an informational message with optional indentation.

    Parameters
    ----------
    message : str
        Message to print
    indent : int, optional
        Number of spaces to indent, default 0

    Returns
    -------
    None
    """
    print(" " * indent + message)


def success(message: str, indent: int = 0) -> None:
    """
    Print a success message with checkmark.

    Parameters
    ----------
    message : str
        Success message to print
    indent : int, optional
        Number of spaces to indent, default 0

    Returns
    -------
    None
    """
    print(" " * indent + f"✓ {message}")


def error(message: str, indent: int = 0) -> None:
    """
    Print an error message with X mark.

    Parameters
    ----------
    message : str
        Error message to print
    indent : int, optional
        Number of spaces to indent, default 0

    Returns
    -------
    None
    """
    print(" " * indent + f"✗ {message}")


def warning(message: str, indent: int = 0) -> None:
    """
    Print a warning message.

    Parameters
    ----------
    message : str
        Warning message to print
    indent : int, optional
        Number of spaces to indent, default 0

    Returns
    -------
    None
    """
    print(" " * indent + f"Warning: {message}")


def header(title: str) -> None:
    """
    Print a formatted header.

    Parameters
    ----------
    title : str
        Header title

    Returns
    -------
    None
    """
    print(f"{title}")
    print("=" * len(title))


def subheader(title: str) -> None:
    """
    Print a formatted subheader.

    Parameters
    ----------
    title : str
        Subheader title

    Returns
    -------
    None
    """
    print(f"{title}")
    print("-" * len(title))
