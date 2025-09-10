import contextlib
from pathlib import Path
from plotly.graph_objs import Figure as PlotlyFigure
from matplotlib.figure import Figure as MatplotlibFigure
import matplotlib.pyplot as plt


def plotly_to_matplotlib(fig):
    '''
    Chat gpt code to make a simple matpllib figure from a plotly figure.
    '''
    plt.figure()
    ax = plt.gca()
    for tr in fig.data:
        t = tr.type
        if t in ("scatter", "scattergl"):
            mode = (tr.mode or "lines").split("+")
            linestyle = "-" if "lines" in mode else "None"
            marker = "o" if "markers" in mode else None
            ax.plot(tr.x, tr.y, linestyle=linestyle, marker=marker, label=tr.name)
        elif t == "bar":
            ax.bar(tr.x, tr.y, label=tr.name)
        elif t == "histogram":
            # Reconstructing histogram edges precisely can be tricky; as a simple fallback:
            ax.hist(tr.x, bins=getattr(tr, "nbinsx", None))
        else:
            print(f"Trace type not handled: {t}")
    if any(getattr(tr, "name", None) for tr in fig.data):
        ax.legend()
    ax.set_title(fig.layout.title.text if fig.layout.title and fig.layout.title.text else "")
    ax.set_xlabel(fig.layout.xaxis.title.text if fig.layout and fig.layout.xaxis and fig.layout.xaxis.title else "")
    ax.set_ylabel(fig.layout.yaxis.title.text if fig.layout and fig.layout.yaxis and fig.layout.yaxis.title else "")
    plt.tight_layout()
    return ax

def log_html(live, fig_name, fig):
    """
    Log an image representation of a figure and delete the file afterward.
    Args:
        live: The live logging object.
        fig_name: Name for the temporary image file.
        fig: The figure to log (should support write_image or savefig).
    """
    fig_path = Path(live.dir) / fig_name

    if isinstance(fig, PlotlyFigure):
        # Plotly figure
        try:
            fig.write_image(str(fig_path))
            live.log_image(fig_name, str(fig_path))
        except Exception as e:
            print("Plotly write_image failed, trying conversion to Matplotlib")
            print(f"Error was: \n{e}")
            # Fallback: convert to Matplotlib and save
            try:
                mpl_fig = plotly_to_matplotlib(fig)
                mpl_fig.figure.savefig(str(fig_path), bbox_inches="tight")
                live.log_image(fig_name, str(fig_path))
            except Exception as e2:
                print("Conversion to Matplotlib also failed, skipping figure logging")
                print(f"Error was: \n{e2}")

    elif isinstance(fig, MatplotlibFigure):
        # Matplotlib figure
        fig.savefig(str(fig_path), bbox_inches="tight")
        live.log_image(fig_name, str(fig_path))
    else:
        print(f"Unsupported figure type for {fig_name}")

    with contextlib.suppress(FileNotFoundError):
        fig_path.unlink()
