import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

"""
visualization.py
Plotting utilities for comparing Bloom pattern sensitivity results.
"""

def plot_sensitivity_violin(
    pattern_sensitivities: dict[str, np.ndarray],
    use_absolute: bool = True,
    show_points: bool = True,
    figsize: tuple = (9, 5),
    color: str = "#4C72B0",
    title: str = "Hinge Sensitivity Distribution",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Violin plot comparing the distribution of hinge sensitivity values
    across multiple Bloom patterns.

    Parameters
    ----------
    pattern_sensitivities : dict
        Maps pattern name -> sensitivity vector s (one entry per hinge).
    use_absolute : bool
        If True, plot |s_i| so that M/V sign convention does not split the
        distribution. If False, plot signed values.
    show_points : bool
        Overlay individual hinge values as jittered points. Strongly
        recommended when hinge count is small (h < 30).
    figsize : tuple
        Figure size in inches (width, height).
    color : str
        Hex color for violin bodies.
    title : str
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects.

    Example
    -------
    sensitivities = {
        "Bloom-4"  : s_bloom4,
        "Bloom-6"  : s_bloom6,
        "Yoshimura": s_yoshi,
    }
    fig, ax = plot_sensitivity_violin(sensitivities)
    fig.savefig("sensitivity_violin.pdf", bbox_inches="tight")
    """

    names = list(pattern_sensitivities.keys())
    data = [
        np.abs(s) if use_absolute else np.asarray(s)
        for s in pattern_sensitivities.values()
    ]
    positions = np.arange(len(names))

    fig, ax = plt.subplots(figsize=figsize)

    # --- Violin bodies ---
    parts = ax.violinplot(
        data,
        positions=positions,
        showmedians=True,
        showextrema=True,
        widths=0.6,
    )

    edge_color = _darken(color, factor=0.6)

    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(edge_color)
        body.set_alpha(0.75)
        body.set_linewidth(1.2)

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_edgecolor(edge_color)
        parts[key].set_linewidth(1.5)
    parts["cmedians"].set_edgecolor("white")
    parts["cmedians"].set_linewidth(2.0)

    # --- Individual hinge points (jittered) ---
    if show_points:
        rng = np.random.default_rng(seed=0)   # reproducible jitter
        for i, d in enumerate(data):
            jitter = rng.uniform(-0.07, 0.07, size=len(d))
            ax.scatter(
                positions[i] + jitter, d,
                s=28, zorder=4,
                color="white", edgecolors=edge_color,
                linewidths=0.8, alpha=0.95,
            )

    # --- Axes formatting ---
    ylabel = (
        r"Normalized Sensitivity $|s_i|$"
        if use_absolute
        else r"Normalized Sensitivity $s_i$"
    )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Pattern", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)
    ax.grid(axis="y", which="minor", linestyle=":", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax


def _darken(hex_color: str, factor: float = 0.7) -> str:
    """Scale RGB channels of a hex color by factor (darkens when factor < 1)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        int(r * factor), int(g * factor), int(b * factor)
    )


from typing import Union, Dict

def calculate_stats(data: Union[np.ndarray, list], use_sample: bool = True) -> Dict[str, float]:
    """Calculates mean, median, standard deviation, and CV."""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot calculate statistics on an empty array.")
        
    ddof = 1 if use_sample else 0
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    std_dev = np.std(arr, ddof=ddof)
    
    cv = np.nan if mean_val == 0 else std_dev / mean_val
        
    return {
        "mean": mean_val,
        "median": median_val,
        "std_dev": std_dev,
        "cv": cv
    }

import matplotlib.pyplot as plt

def plot_fold_pattern(fold_data, title="Crease Pattern"):
    """
    Plots a .fold dictionary using Matplotlib.
    Mountain folds (M) are blue.
    Valley folds (V) are red.
    Boundaries (B) are black.
    Unassigned/other (U/F) are gray.
    """
    
    # Extract data from the dictionary
    vertices = fold_data.get("vertices_coords", [])
    edges = fold_data.get("edges_vertices", [])
    assignments = fold_data.get("edges_assignment", [])
    
    # Setup color and linewidth mapping — standard origami drafting convention:
    # M = Mountain: bold red dashed  (----)
    # V = Valley:   light blue dash-dot  (-.-.)
    # B = Boundary: thick black solid
    style_map = {
        "M": {"color": "#D62728", "linewidth": 2.5, "linestyle": "--"},
        "V": {"color": "#5BC8F5", "linewidth": 1.5, "linestyle": "-."},
        "B": {"color": "black",   "linewidth": 3.0, "linestyle": "-"},
        "U": {"color": "gray",    "linewidth": 1.0, "linestyle": ":"},
        "F": {"color": "gray",    "linewidth": 1.0, "linestyle": ":"}
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Iterate through edges and plot them
    for i, edge in enumerate(edges):
        v1_idx, v2_idx = edge
        x_coords = [vertices[v1_idx][0], vertices[v2_idx][0]]
        y_coords = [vertices[v1_idx][1], vertices[v2_idx][1]]
        
        # Default to 'U' (Unassigned) if assignment is missing or unknown
        assign = assignments[i] if i < len(assignments) else "U"
        style = style_map.get(assign, style_map["U"])
        
        ax.plot(x_coords, y_coords, 
                color=style["color"], 
                linewidth=style["linewidth"], 
                linestyle=style["linestyle"])

    # Format the plot for accurate geometric viewing
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Create a custom legend matching the style_map above
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#D62728', lw=2.5, linestyle='--', label='Mountain (M)'),
        Line2D([0], [0], color='#5BC8F5', lw=1.5, linestyle='-.', label='Valley (V)'),
        Line2D([0], [0], color='black',   lw=3.0, linestyle='-',  label='Boundary (B)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()
