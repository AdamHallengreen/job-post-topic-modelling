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
    # Ensure the figure file is within the DVC project directory
    fig_path = Path(live.dir) / fig_name
    try:
        # Try Plotly-style export
        fig.write_image(str(fig_path))
        live.log_image(fig_name, str(fig_path))
    except Exception:
        try:
            # Try Matplotlib-style export
            fig.savefig(str(fig_path), bbox_inches="tight")
            live.log_image(fig_name, str(fig_path))
        except Exception as e:
            print(f"Error saving figure {fig_name}: {e}")
    finally:
        with contextlib.suppress(FileNotFoundError):
            fig_path.unlink()
