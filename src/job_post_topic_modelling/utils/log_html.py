import contextlib
from pathlib import Path


def log_html(live, fig_name, fig):
    """
    Log an HTML representation of a figure, automatically deleting the file afterward.

    Args:
        live: The live logging object
        fig: The figure to log
        fig_name: Name for the figure file (will be deleted after logging)
    """
    try:
        fig.write_image(fig_name)
        live.log_image(fig_name, fig_name)
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(fig_name).unlink()
