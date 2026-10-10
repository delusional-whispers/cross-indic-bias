"""Visualization module for IndicFair-Audit research findings.

Provides publication-grade charts:
1. Language Comparison Bar Chart: SPS across English, Hindi, and Telugu with 50% baseline.
2. Cross-Category Radar / Polar Chart: Multidimensional bias shift across social domains.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union


def plot_language_comparison(
    sps_scores: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Cross-Lingual Stereotype Preference Score (SPS)",
) -> Any:
    """Generate a publication-grade bar chart comparing SPS across languages.

    Visualizes English, Hindi, and Telugu SPS with an unbiased null baseline at 50%.

    Args:
        sps_scores: Dictionary mapping language keys ('en', 'hi', 'te') to SPS floats.
        output_path: Optional file path to save HTML or image figure.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    lang_labels = {
        "en": "English",
        "hi": "Hindi (हिन्दी)",
        "te": "Telugu (తెలుగు)",
    }

    categories = [lang_labels.get(k, k) for k in sps_scores.keys()]
    values = [sps_scores[k] for k in sps_scores.keys()]

    # Color mapping: Green near 50%, Warm orange/red if high stereotype bias
    colors = []
    for val in values:
        if abs(val - 50.0) < 5.0:
            colors.append("#2ca02c")  # balanced green
        elif val > 50.0:
            colors.append("#d62728")  # stereotypical red
        else:
            colors.append("#1f77b4")  # counter-stereotypical blue

    fig = go.Figure()

    # Main bar trace
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            marker=dict(color=colors, line=dict(color="#1f2937", width=1.5)),
            name="SPS (%)",
        )
    )

    # 50% Null baseline dotted line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(categories) - 0.5,
        y0=50.0,
        y1=50.0,
        line=dict(color="#6b7280", width=2.5, dash="dashdot"),
    )

    fig.add_annotation(
        x=len(categories) - 0.5,
        y=52.5,
        text="Unbiased Null Baseline (50%)",
        showarrow=False,
        font=dict(size=12, color="#4b5563"),
        xanchor="right",
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, family="Arial")),
        xaxis=dict(title="Language", tickfont=dict(size=14)),
        yaxis=dict(
            title="Stereotype Preference Score (%)",
            range=[0, 105],
            dtick=10,
            gridcolor="#e5e7eb",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=40, t=60, b=60),
        showlegend=False,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if str(out).endswith(".html"):
            fig.write_html(str(out))
        else:
            try:
                fig.write_image(str(out))
            except Exception:
                fig.write_html(str(out.with_suffix(".html")))

    return fig


def plot_category_radar(
    category_sps: Dict[str, Dict[str, float]],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Domain-Specific Bias Shift Across Languages",
) -> Any:
    """Generate a radar / polar chart showing stereotype bias across categories.

    Args:
        category_sps: Dictionary mapping category IDs to language SPS dicts,
                      e.g., {'gender_stem_labor': {'en': 55.0, 'hi': 75.0, 'te': 60.0}}
        output_path: Optional file path to save HTML or image figure.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    cat_labels = {
        "gender_stem_labor": "Gender & STEM",
        "socioeconomic_caste": "Socioeconomic & Caste",
        "regional_linguistic": "Regional & Linguistic",
    }

    raw_cats = list(category_sps.keys())
    categories = [cat_labels.get(c, c) for c in raw_cats]

    # Close the radar loop
    radar_categories = categories + [categories[0]] if categories else []

    fig = go.Figure()

    languages = [
        ("en", "English", "#2563eb"),
        ("hi", "Hindi", "#dc2626"),
        ("te", "Telugu", "#16a34a"),
    ]

    for lang_code, lang_name, color in languages:
        values = [category_sps[cat].get(lang_code, 50.0) for cat in raw_cats]
        if values:
            radar_values = values + [values[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=radar_values,
                    theta=radar_categories,
                    fill="toself",
                    name=lang_name,
                    line=dict(color=color, width=2),
                    opacity=0.6,
                )
            )

    # Add 50% baseline ring
    if radar_categories:
        fig.add_trace(
            go.Scatterpolar(
                r=[50.0] * len(radar_categories),
                theta=radar_categories,
                mode="lines",
                name="Null Baseline (50%)",
                line=dict(color="#6b7280", width=1.5, dash="dash"),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                dtick=20,
                gridcolor="#e5e7eb",
            ),
            angularaxis=dict(gridcolor="#e5e7eb"),
            bgcolor="#ffffff",
        ),
        title=dict(text=title, font=dict(size=18, family="Arial")),
        paper_bgcolor="#ffffff",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=60, b=60),
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if str(out).endswith(".html"):
            fig.write_html(str(out))
        else:
            try:
                fig.write_image(str(out))
            except Exception:
                fig.write_html(str(out.with_suffix(".html")))

    return fig
