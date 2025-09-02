import contextlib
from pathlib import Path


def log_html(live, fig_name, fig):
    """
    Log an image representation of a figure and delete the file afterward.

    Args:
        live: The live logging object.
        fig_name: Name for the temporary image file.
        fig: The figure to log (should support write_image or savefig).
    """
    try:
        # Try Plotly-style export
        fig.write_image(fig_name)
        live.log_image(fig_name, fig_name)
    except Exception:
        try:
            # Try Matplotlib-style export
            fig.savefig(fig_name, bbox_inches="tight")
            live.log_image(fig_name, fig_name)
        except Exception as e:
            print(f"Error saving figure {fig_name}: {e}")
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(fig_name).unlink()
