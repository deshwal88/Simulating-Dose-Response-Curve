from flask import Flask, render_template, request
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import functions


# ---------- Plot helpers ----------

def make_main_figure(concs, responses_list, labels=None):
    """
    Create the main plotly figure showing continuous curves and dilution points.

    Parameters
    ----------
    concs : list[np.ndarray]
        List of x arrays (linear concentrations) for each curve. MUST align with responses_list.
    responses_list : list[np.ndarray]
        List of y arrays (responses) computed on the corresponding concs[i].
    labels : list[str]
        Labels for legend (defaults to ["40%", "100%", "160%"]).
    """
    fig = go.Figure()
    labels = labels or ["40%", "100%", "160%"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for x_vals, y_vals, lab, col in zip(concs, responses_list, labels, colors):
        # continuous curve
        fig.add_trace(
            go.Scatter(
                x=np.log10(x_vals),
                y=y_vals,
                mode="lines",
                name=f"{lab} curve",
                line=dict(color=col),
            )
        )
        # dilution points: choose 8 evenly spaced indices along this curve
        try:
            idxs = np.linspace(0, len(x_vals) - 1, 8, dtype=int)
        except Exception:
            idxs = np.arange(min(len(x_vals), 8))

        fig.add_trace(
            go.Scatter(
                x=np.log10(np.array(x_vals)[idxs]),
                y=np.array(y_vals)[idxs],
                mode="markers",
                name=f"{lab} dilution points",
                marker=dict(size=8, color=col),
            )
        )

    fig.update_layout(
        xaxis_title="Log10(Concentration)",
        yaxis_title="Response",
        title="Main dose-response curves",
        legend_tracegroupgap=8,
    )
    return fig


def make_edge_case_grid(top_conc, dilution_points, param_sets):
    """
    Create a 4x4 grid showing edge-case curves.

    For each subplot (one (A,B,C,D) tuple), we:
      - keep A,B,D FIXED
      - create a common dense x_grid (linear) for smooth lines
      - draw 3 curves with C scaled ONLY by potency: C_eff = C / conc_scale
      - overlay the 8 dilution markers for each potency at the corresponding concentrations
    """
    rows, cols = 4, 4
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"A={a}, B={b}, C={c}, D={d}" for (a, b, c, d) in param_sets],
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["40%", "100%", "160%"]
    conc_scales = [0.4, 1.0, 1.6]

    # Common dense grid for smooth curves across all subplots
    # (cover the dilution range with a small margin)
    x_min = max(min(dilution_points) * 0.9, 1e-12)  # guard non-positive
    x_max = max(dilution_points) * 1.1
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    for idx, (A, B, C, D) in enumerate(param_sets):
        r = idx // cols + 1
        c = idx % cols + 1

        for conc_scale, lab, col in zip(conc_scales, labels, colors):
            # potency shift -> lateral shift -> scale C ONLY
            C_eff = C / conc_scale

            # smooth line on shared x_grid
            y_line = functions.compute_4pl(A, B, C_eff, D, x_grid)
            fig.add_trace(
                go.Scatter(
                    x=np.log10(x_grid),
                    y=y_line,
                    mode="lines",
                    name=f"{lab} curve",
                    line=dict(color=col),
                    showlegend=(idx == 0),
                ),
                row=r,
                col=c,
            )

            # 8 dilution markers for this potency (scale absolute dilution concentrations)
            rel_concs = np.array(dilution_points) 
            y_pts = functions.compute_4pl(A, B, C_eff, D, rel_concs)
            fig.add_trace(
                go.Scatter(x=np.log10(rel_concs),
                           y=y_pts,
                           mode="markers",
                           name=f"{lab} dilution points",
                           marker=dict(size=7, color=col),
                           showlegend=(idx == 0),
                ),
                row=r,
                col=c,
            )

    fig.update_layout(
        height=900,
        width=1100,
        title_text="Edge-case parameter combinations (16)",
        legend_tracegroupgap=8,
    )
    fig.update_xaxes(title_text="Log10(Concentration)")
    return fig


# ---------- Flask app ----------

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    # defaults
    defaults = dict(
        A_min=0.5,
        A_max=0.5,
        B_min=1.0,
        B_max=2.0,
        C_min=2.0,
        C_max=8.0,
        D_min=3.5,
        D_max=3.5,
        top_conc=100.0,
        dil_factor=5,
        scheme_mode="even",
        custom_factors="3,3,2,2,2,3,3",
    )
    params = defaults.copy()
    error = None

    if request.method == "POST":
        try:
            for k in defaults.keys():
                if k in request.form and request.form[k] != "":
                    params[k] = request.form[k]
            # convert numeric types
            params["A_min"] = float(params["A_min"])
            params["A_max"] = float(params["A_max"])
            params["B_min"] = float(params["B_min"])
            params["B_max"] = float(params["B_max"])
            params["C_min"] = float(params["C_min"])
            params["C_max"] = float(params["C_max"])
            params["D_min"] = float(params["D_min"])
            params["D_max"] = float(params["D_max"])
            params["top_conc"] = float(params["top_conc"])
            params["dil_factor"] = float(params["dil_factor"])
        except Exception as e:
            error = f"Invalid numeric input: {e}"

    # build dilution points
    try:
        if params.get("scheme_mode", "even") == "even":
            dilution_points = functions.calculate_even_dilution(
                params["top_conc"], float(params["dil_factor"]), points=8
            )
        else:
            factors = [
                float(x.strip())
                for x in params.get("custom_factors", "").split(",")
                if x.strip()
            ]
            dilution_points = functions.calculate_custom_dilution(
                params["top_conc"], factors
            )
    except Exception as e:
        error = f"Error generating dilution points: {e}"
        dilution_points = functions.calculate_even_dilution(
            defaults["top_conc"], defaults["dil_factor"], points=8
        )

    # main figure mids
    A_mid = 0.5 * (params["A_min"] + params["A_max"])
    B_mid = 0.5 * (params["B_min"] + params["B_max"])
    C_mid = 0.5 * (params["C_min"] + params["C_max"])
    D_mid = 0.5 * (params["D_min"] + params["D_max"])

    # smooth grid that covers the dilution range with a margin
    x_min = max(min(dilution_points) * 0.9, 1e-12)
    x_max = max(dilution_points) * 1.1
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 200)

    # curves for 40%, 100%, 160% (Fix #1: append SCALED x for each curve)
    concs_for_curve = []
    responses = []
    for scale in (0.4, 1.0, 1.6):
        scaled_x = x_grid * scale
        y = functions.compute_4pl(A_mid, B_mid, C_mid, D_mid, scaled_x)
        concs_for_curve.append(x_grid)  # <-- FIXED: use scaled_x (not x_grid)
        responses.append(y)

    main_fig = make_main_figure(concs_for_curve, responses)

    # 16 edge-case parameter sets
    edge_param_sets = functions.edge_case_parameter_sets(
        (params["A_min"], params["A_max"]),
        (params["B_min"], params["B_max"]),
        (params["C_min"], params["C_max"]),
        (params["D_min"], params["D_max"]),
    )

    # edge-case grid (Fix #2 done inside)
    edge_fig = make_edge_case_grid(params["top_conc"], dilution_points, edge_param_sets)

    # render
    main_html = pio.to_html(main_fig, full_html=False, include_plotlyjs="cdn")
    edge_html = pio.to_html(edge_fig, full_html=False, include_plotlyjs=False)

    notes = f"Dilution points (first->last): {', '.join([f'{v:.6g}' for v in dilution_points])}"
    # recommended even dilution factors based on top concentration and midpoint C
    try:
        recommendations = functions.recommend_even_dilution_factors(
            params["top_conc"], C_mid, points=8
        )
    except Exception:
        recommendations = []
    return render_template(
        "index.html",
        main_plot=main_html,
        edge_plot=edge_html,
        params=params,
        notes=notes,
        error=error,
        recommendations=recommendations,
    )


if __name__ == "__main__":
    app.run(debug=True)
