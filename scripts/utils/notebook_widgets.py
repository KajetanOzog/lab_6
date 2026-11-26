"""Wyświetlanie aktualizowanych figur Matplotlib w notebookach."""

from io import BytesIO

import ipywidgets as widgets
import matplotlib.pyplot as plt


def figure_png(figure):
    """Zapisuje figurę jako PNG i zwalnia ją w Matplotlib."""
    try:
        with BytesIO() as buffer:
            figure.savefig(buffer, format="png", bbox_inches="tight")
            return buffer.getvalue()
    finally:
        plt.close(figure)


def live_figure(draw, controls):
    """Podmienia obraz po zmianie wartości dowolnej kontrolki."""
    image = widgets.Image(format="png", layout=widgets.Layout(max_width="100%"))

    def update(_change=None):
        values = {name: control.value for name, control in controls.items()}
        with plt.ioff():
            image.value = figure_png(draw(**values))

    update()
    for control in controls.values():
        control.observe(update, names="value")
    return image
