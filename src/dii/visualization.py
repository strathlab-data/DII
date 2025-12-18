"""
Visualization module for the DII Calculator.

This module provides plotting functions for visualizing DII scores
and nutrient contributions.

Note: matplotlib is a required dependency of the dii-calculator package.
"""

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_dii_distribution(
    dii_scores: Union[pd.Series, pd.DataFrame],
    title: str = "Distribution of Dietary Inflammatory Index Scores",
    figsize: tuple = (10, 6),
    color: str = "#4C72B0",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Create a histogram of DII scores with reference lines.
    
    Parameters
    ----------
    dii_scores : pd.Series or pd.DataFrame
        DII scores. If DataFrame, will look for 'DII_score' column.
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height) in inches.
    color : str
        Histogram bar color.
    save_path : str or Path, optional
        If provided, save figure to this path.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if show=False, otherwise None.
    
    Examples
    --------
    >>> from dii import calculate_dii, plot_dii_distribution
    >>> results = calculate_dii(nutrient_data)
    >>> plot_dii_distribution(results)
    """
    
    # Extract scores
    if isinstance(dii_scores, pd.DataFrame):
        if 'DII_score' in dii_scores.columns:
            scores = dii_scores['DII_score'].dropna()
        else:
            raise ValueError("DataFrame must contain 'DII_score' column")
    else:
        scores = dii_scores.dropna()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    n, bins, patches = ax.hist(
        scores, 
        bins=50, 
        edgecolor='white',
        linewidth=0.5,
        alpha=0.85,
    )
    
    # Color bars by category
    # Note: BarContainer is iterable but mypy stubs don't reflect this correctly
    for i, patch in enumerate(list(patches)):  # type: ignore[arg-type]
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < -1:
            patch.set_facecolor('#2ecc71')  # Green - anti-inflammatory
        elif bin_center > 1:
            patch.set_facecolor('#e74c3c')  # Red - pro-inflammatory
        else:
            patch.set_facecolor('#f39c12')  # Orange - neutral
    
    # Reference lines
    ax.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=2, alpha=0.7)
    ax.axvline(x=-1, color='#27ae60', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=1, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Labels and styling
    ax.set_xlabel('DII Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', label=f'Anti-inflammatory (< -1): {(scores < -1).sum()}'),
        mpatches.Patch(facecolor='#f39c12', label=f'Neutral (-1 to 1): {((scores >= -1) & (scores <= 1)).sum()}'),
        mpatches.Patch(facecolor='#e74c3c', label=f'Pro-inflammatory (> 1): {(scores > 1).sum()}'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Stats annotation
    stats_text = f'n = {len(scores):,}\nMean = {scores.mean():.2f}\nSD = {scores.std():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
        return None
    
    return fig


def plot_nutrient_contributions(
    detailed_results: pd.DataFrame,
    top_n: int = 20,
    title: str = "Average Nutrient Contributions to DII Score",
    figsize: tuple = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Create a horizontal bar chart showing nutrient contributions.
    
    Parameters
    ----------
    detailed_results : pd.DataFrame
        Detailed DII results from calculate_dii_detailed().
    top_n : int
        Number of top contributors to show (from each direction).
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height) in inches.
    save_path : str or Path, optional
        If provided, save figure to this path.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if show=False, otherwise None.
    """
    # Find DII contribution columns (could be _contribution or _dii)
    contrib_cols = [col for col in detailed_results.columns 
                    if col.endswith('_contribution')]
    
    if not contrib_cols:
        # Try alternative naming
        contrib_cols = [col for col in detailed_results.columns 
                        if col.endswith('_dii') and col != 'DII_score']
    
    if not contrib_cols:
        raise ValueError("No nutrient contribution columns found. Use calculate_dii_detailed().")
    
    # Calculate mean contributions
    contributions = detailed_results[contrib_cols].mean()
    contributions.index = [col.replace('_contribution', '').replace('_dii', '') 
                          for col in contributions.index]
    contributions = contributions.sort_values()
    
    # Select top contributors
    if len(contributions) > top_n:
        # Get top anti-inflammatory and pro-inflammatory
        n_each = top_n // 2
        top_anti = contributions.head(n_each)
        top_pro = contributions.tail(n_each)
        contributions = pd.concat([top_anti, top_pro])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Colors based on direction
    colors = ['#2ecc71' if v < 0 else '#e74c3c' for v in contributions.to_numpy()]
    
    # Create bars
    y_pos = np.arange(len(contributions))
    ax.barh(y_pos, contributions.values, color=colors, edgecolor='white', linewidth=0.5)
    
    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(contributions.index, fontsize=10)
    ax.set_xlabel('Average Contribution to DII Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Reference line at zero
    ax.axvline(x=0, color='#2c3e50', linestyle='-', linewidth=1)
    
    # Direction labels
    ax.text(contributions.min() * 0.5, len(contributions) + 0.5, 
            '← Anti-inflammatory', fontsize=10, color='#27ae60', fontweight='bold')
    ax.text(contributions.max() * 0.3, len(contributions) + 0.5, 
            'Pro-inflammatory →', fontsize=10, color='#c0392b', fontweight='bold')
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
        return None
    
    return fig


def plot_dii_categories_pie(
    dii_scores: Union[pd.Series, pd.DataFrame],
    title: str = "DII Score Categories",
    figsize: tuple = (8, 8),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """
    Create a pie chart showing DII category distribution.
    
    Parameters
    ----------
    dii_scores : pd.Series or pd.DataFrame
        DII scores. If DataFrame, will look for 'DII_score' column.
    title : str
        Plot title.
    figsize : tuple
        Figure size (width, height) in inches.
    save_path : str or Path, optional
        If provided, save figure to this path.
    show : bool
        Whether to display the plot.
    
    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if show=False, otherwise None.
    """
    # Extract scores
    if isinstance(dii_scores, pd.DataFrame):
        if 'DII_score' in dii_scores.columns:
            scores = dii_scores['DII_score'].dropna()
        else:
            raise ValueError("DataFrame must contain 'DII_score' column")
    else:
        scores = dii_scores.dropna()
    
    # Calculate categories
    categories = {
        'Anti-inflammatory\n(DII < -1)': (scores < -1).sum(),
        'Neutral\n(-1 ≤ DII ≤ 1)': ((scores >= -1) & (scores <= 1)).sum(),
        'Pro-inflammatory\n(DII > 1)': (scores > 1).sum(),
    }
    
    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v > 0}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c'][:len(categories)]
    
    # Note: ax.pie returns 3 values when autopct is provided, but stubs don't reflect this
    wedges, texts, autotexts = ax.pie(  # type: ignore[misc]
        list(categories.values()),
        labels=list(categories.keys()),
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*len(scores)):,})',
        startangle=90,
        explode=[0.02] * len(categories),
        textprops={'fontsize': 11},
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    if show:
        plt.show()
        return None
    
    return fig

