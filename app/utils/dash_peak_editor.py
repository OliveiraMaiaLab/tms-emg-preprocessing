# app/utils/dash_peak_editor.py
"""
dash_peak_editor.py
-------------------
Embedded Dash editor for min/max peak correction.

- Reads current MEP waveform (from session pulses + raw data)
- Shows draggable vertical cursors (min/max)
- Auto-writes updated min/max to session JSON on drag (no explicit save needed)
- Optional "Save min/max" button remains as a manual action (still useful)
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, parse_qs

from flask import request

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, no_update

from app.utils.tms_module import (
    load_meps_for_block,
    read_json,
    write_json,
    get_epoch_from_session,
)

# -------------------------
# Config
# -------------------------
SHOW_DEBUG = False


# -------------------------
# Small helpers
# -------------------------
def _interp_y(xq: float, x: np.ndarray, y: np.ndarray) -> float:
    return float(np.interp(float(xq), x, y))


def _debug_style() -> dict:
    return {
        "fontSize": "12px",
        "background": "#f6f8fa",
        "padding": "8px",
        "border": "1px solid #ddd",
        "whiteSpace": "pre-wrap",
    }


def _error_figure(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Dash Peak Editor (error)",
        annotations=[
            dict(
                text=msg.replace("\n", "<br>"),
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=16),
            )
        ],
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _ctx_from_request() -> tuple[dict, list[str]]:
    """
    Dash loads layout via GET /_dash-layout (often without query string).
    If request.args is empty, parse ctx from Referer.
    """
    dbg: list[str] = []

    args = dict(request.args or {})
    dbg.append(f"request.path={request.path}")
    dbg.append(f"request.query_string={request.query_string.decode('utf-8', errors='ignore')}")
    dbg.append(f"request.args={args}")

    block   = str(args.get("block", "")).strip()
    hemi    = str(args.get("hemi", "left")).strip()
    mep_raw = args.get("mep", "0")

    referer = request.headers.get("Referer", "") or ""
    dbg.append(f"referer={referer}")

    if not block and referer:
        try:
            u = urlparse(referer)
            q = parse_qs(u.query)
            block   = (q.get("block") or [""])[0].strip()
            hemi    = (q.get("hemi")  or ["left"])[0].strip()
            mep_raw = (q.get("mep")   or ["0"])[0]
            dbg.append(f"parsed_from_referer={dict((k, v[0] if v else '') for k, v in q.items())}")
        except Exception as e:
            dbg.append(f"referer_parse_error={type(e).__name__}: {e}")

    try:
        mep_idx = int(mep_raw)
    except Exception:
        mep_idx = 0

    ctx = {}
    if block:
        ctx = {"block": block, "hemi": hemi or "left", "mep_idx": int(mep_idx)}

    dbg.append(f"parsed_ctx={ctx}")
    return ctx, dbg


# -------------------------
# Plot construction
# -------------------------
def _cursor_shapes(x_min: float, x_max: float) -> list[dict]:
    """
    Two draggable vertical cursors as shapes.
    shapes[0] → min cursor, shapes[1] → max cursor.
    """
    def _one(x):
        return dict(
            type="line",
            x0=float(x),
            x1=float(x),
            y0=-0.05,
            y1=1.05,
            xref="x",
            yref="paper",
            line=dict(width=2, dash="solid"),
            layer="above",
        )

    return [_one(x_min), _one(x_max)]


def _epoch_shapes(epoch) -> list[dict]:
    return [
        dict(
            type="line",
            x0=float(epoch.tmin_ms),
            x1=float(epoch.tmin_ms),
            y0=-0.05,
            y1=1.05,
            xref="x",
            yref="paper",
            line=dict(width=1, dash="dash"),
            layer="above",
        ),
        dict(
            type="line",
            x0=float(epoch.tmax_ms),
            x1=float(epoch.tmax_ms),
            y0=-0.05,
            y1=1.05,
            xref="x",
            yref="paper",
            line=dict(width=1, dash="dash"),
            layer="above",
        ),
    ]


def _trace_ylim(y: np.ndarray) -> tuple[float, float]:
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        ymin, ymax = -1.0, 1.0
    pad = 0.05 * (ymax - ymin)
    if pad <= 0:
        pad = 1.0
    return ymin - pad, ymax + pad


def make_figure(t_ms: np.ndarray, y: np.ndarray, points: dict, epoch, title: str) -> go.Figure:
    x0 = float(points["x"][0])
    x1 = float(points["x"][1])
    ylo, yhi = _trace_ylim(y)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=y,
            mode="lines",
            name="MEP",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[float(points["y"][0]), float(points["y"][1])],
            mode="markers",
            name="Min/Max",
            marker=dict(size=16),
            hoverinfo="skip",
        )
    )

    shapes = _cursor_shapes(x0, x1) + _epoch_shapes(epoch)

    fig.update_layout(
        title=title,
        margin=dict(l=40, r=20, t=60, b=40),
        uirevision="keep-zoom",
        dragmode="pan",
        shapes=shapes,
        hovermode=False,
    )

    fig.update_xaxes(title="Time (ms)", fixedrange=True)
    fig.update_yaxes(title="EMG (uV)", fixedrange=True, range=[ylo, yhi])

    return fig


def _extract_cursor_x_from_relayout(relayout: dict, fallback_points: dict) -> tuple[float, float] | None:
    if not relayout:
        return None

    x0 = None
    x1 = None

    for k, v in relayout.items():
        if k in ("shapes[0].x0", "shapes[0].x1"):
            try:
                x0 = float(v)
            except Exception:
                pass
        if k in ("shapes[1].x0", "shapes[1].x1"):
            try:
                x1 = float(v)
            except Exception:
                pass

    if x0 is not None or x1 is not None:
        prev0 = float(fallback_points["x"][0])
        prev1 = float(fallback_points["x"][1])
        return (x0 if x0 is not None else prev0, x1 if x1 is not None else prev1)

    if "shapes" in relayout and isinstance(relayout["shapes"], list) and len(relayout["shapes"]) >= 2:
        try:
            s0 = relayout["shapes"][0]
            s1 = relayout["shapes"][1]
            return (
                float(s0.get("x0", fallback_points["x"][0])),
                float(s1.get("x0", fallback_points["x"][1])),
            )
        except Exception:
            return None

    return None


def _assign_min_max(
    xa: float, ya: float, xb: float, yb: float
) -> tuple[list[float], list[float]]:
    """
    Bug #14: assign min/max based on which cursor has the lower signal amplitude,
    not by cursor order (shapes[0] vs shapes[1]).

    Previously, shapes[0] was always written to payload["min"] and shapes[1] to
    payload["max"]. If the user dragged the "min" cursor to a position with a
    higher value than the "max" cursor, the labels silently swapped, breaking the
    downstream peak-to-peak calculation and the marker display in step_peakChecking.
    """
    if ya <= yb:
        return [xa, ya], [xb, yb]
    else:
        return [xb, yb], [xa, ya]


# -------------------------
# App factory
# -------------------------
def create_dash_peak_editor(meta: dict, session_file: str | Path) -> Dash:
    session_file = Path(session_file)

    app = Dash(__name__)
    app.index_string = """
    <!DOCTYPE html>
    <html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
        html, body { height: 100%; margin: 0; overflow: hidden; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
        </footer>
    </body>
    </html>
"""
    app.config.suppress_callback_exceptions = True

    @app.server.route("/health")
    def _health():
        return "ok", 200

    def _serve_layout():
        ctx, dbg_lines = _ctx_from_request()

        fig = _error_figure("Loading...")
        points = {"x": [], "y": []}
        ctx_store = {}

        if not ctx:
            fig = _error_figure(
                "Missing query params.\n"
                "Expected: ?block=<name>&hemi=<left/right>&mep=<index>"
            )
        else:
            try:
                session = read_json(session_file)
                epoch   = get_epoch_from_session(session)

                meps_root = session.get("meps", {}) or {}
                block = str(ctx["block"])
                hemi  = str(ctx["hemi"])
                i     = int(ctx["mep_idx"])

                payload = ((meps_root.get(block) or {}).get(hemi) or {})
                mins = payload.get("min", []) if isinstance(payload, dict) else []
                maxs = payload.get("max", []) if isinstance(payload, dict) else []

                t_ms, meps = load_meps_for_block(
                    meta,
                    session_file,
                    block_name=block,
                    hemi=hemi,
                    epoch=epoch,
                    detrend=False,
                )

                if not isinstance(meps, np.ndarray) or meps.ndim != 2 or meps.size == 0:
                    fig = _error_figure("No MEP data returned (empty pulses or mismatch).")
                elif i < 0 or i >= meps.shape[0]:
                    fig = _error_figure(f"MEP index out of range: {i} (n={meps.shape[0]})")
                else:
                    y = meps[i, :]

                    stored_min = mins[i] if isinstance(mins, list) and i < len(mins) else None
                    stored_max = maxs[i] if isinstance(maxs, list) and i < len(maxs) else None

                    x0 = (
                        float(stored_min[0])
                        if isinstance(stored_min, (list, tuple)) and len(stored_min) == 2 and stored_min[0] is not None
                        else float(epoch.tmin_ms)
                    )
                    x1 = (
                        float(stored_max[0])
                        if isinstance(stored_max, (list, tuple)) and len(stored_max) == 2 and stored_max[0] is not None
                        else float(epoch.tmax_ms)
                    )

                    points = {
                        "x": [x0, x1],
                        "y": [_interp_y(x0, t_ms, y), _interp_y(x1, t_ms, y)],
                    }
                    fig = make_figure(t_ms, y, points, epoch, f"{block} / {hemi} / MEP {i}")
                    ctx_store = {"block": block, "hemi": hemi, "mep_idx": i}

            except Exception as e:
                dbg_lines.append(f"EXCEPTION: {type(e).__name__}: {e}")
                fig = _error_figure(f"{type(e).__name__}: {e}")

        children = [
            dcc.Store(id="ctx-store",    data=ctx_store),
            dcc.Store(id="points-store", data=points),
        ]

        if SHOW_DEBUG:
            children.append(html.Pre("\n".join(dbg_lines), style=_debug_style()))

        children.extend(
            [
                dcc.Graph(
                    id="graph",
                    figure=fig,
                    config={
                        "displayModeBar": False,
                        "displaylogo": False,
                        "scrollZoom": False,
                        "doubleClick": False,
                        "showTips": False,
                        "edits": {"shapePosition": True},
                    },
                    style={"flex": "1 1 auto", "height": "100%", "minHeight": "0"},
                ),
                html.Div(
                    style={"display": "flex", "gap": "12px", "alignItems": "center"},
                    children=[
                        html.Button("Save min/max", id="save-btn"),
                        html.Div(id="status", style={"whiteSpace": "pre-wrap"}),
                    ],
                ),
            ]
        )

        return html.Div(
            style={
                "height": "100vh",
                "width": "100vw",
                "margin": "0",
                "padding": "10px",
                "boxSizing": "border-box",
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",
                "overflow": "hidden",
            },
            children=children,
        )

    app.layout = _serve_layout

    @app.callback(
        Output("points-store", "data"),
        Output("graph", "figure"),
        Input("graph", "relayoutData"),
        State("points-store", "data"),
        State("ctx-store", "data"),
        prevent_initial_call=True,
    )
    def on_drag(relayout_data, points, ctx):
        if not relayout_data or not ctx or not points or len(points.get("x", [])) < 2:
            return no_update, no_update

        new_xs = _extract_cursor_x_from_relayout(relayout_data, points)
        if new_xs is None:
            return no_update, no_update

        session = read_json(session_file)
        epoch   = get_epoch_from_session(session)

        t_ms, meps = load_meps_for_block(
            meta,
            session_file,
            block_name=ctx["block"],
            hemi=ctx["hemi"],
            epoch=epoch,
            detrend=False,
        )

        i = int(ctx["mep_idx"])
        y = meps[i, :]

        xa, xb = float(new_xs[0]), float(new_xs[1])
        ya = _interp_y(xa, t_ms, y)
        yb = _interp_y(xb, t_ms, y)

        # Bug #14: assign min/max by amplitude value, not cursor order.
        # This prevents the labels from silently swapping when the user drags
        # cursor 0 (nominally "min") to a position with a higher y than cursor 1.
        min_pt, max_pt = _assign_min_max(xa, ya, xb, yb)

        new_points = {
            "x": [float(min_pt[0]), float(max_pt[0])],
            "y": [float(min_pt[1]), float(max_pt[1])],
        }

        # AUTO-SAVE min/max to session JSON
        block = str(ctx["block"])
        hemi  = str(ctx["hemi"])

        session.setdefault("meps", {}).setdefault(block, {}).setdefault(hemi, {})
        payload = session["meps"][block][hemi]

        pulses = payload.get("pulses", [])
        n_p = len(pulses) if isinstance(pulses, list) else (i + 1)

        for k, fill in (("min", None), ("max", None)):
            if not isinstance(payload.get(k), list):
                payload[k] = []
            if len(payload[k]) < n_p:
                payload[k].extend([fill] * (n_p - len(payload[k])))

        payload["min"][i] = min_pt
        payload["max"][i] = max_pt

        session["meps"][block][hemi] = payload
        write_json(session_file, session)

        fig = make_figure(t_ms, y, new_points, epoch, f"{block} / {hemi} / MEP {i}")
        return new_points, fig

    @app.callback(
        Output("status", "children"),
        Input("save-btn", "n_clicks"),
        State("points-store", "data"),
        State("ctx-store", "data"),
        prevent_initial_call=True,
    )
    def save(_n, points, ctx):
        if not ctx or not points or len(points.get("x", [])) < 2:
            return "Nothing to save."

        session = read_json(session_file)
        block = str(ctx["block"])
        hemi  = str(ctx["hemi"])
        i     = int(ctx["mep_idx"])

        session.setdefault("meps", {}).setdefault(block, {}).setdefault(hemi, {})
        payload = session["meps"][block][hemi]

        pulses = payload.get("pulses", [])
        n_p = len(pulses) if isinstance(pulses, list) else (i + 1)

        for k, fill in (("min", None), ("max", None)):
            if not isinstance(payload.get(k), list):
                payload[k] = []
            if len(payload[k]) < n_p:
                payload[k].extend([fill] * (n_p - len(payload[k])))

        xa, xb = float(points["x"][0]), float(points["x"][1])
        ya, yb = float(points["y"][0]), float(points["y"][1])

        # Bug #14: same value-based assignment on manual save.
        min_pt, max_pt = _assign_min_max(xa, ya, xb, yb)

        payload["min"][i] = min_pt
        payload["max"][i] = max_pt

        session["meps"][block][hemi] = payload
        write_json(session_file, session)

        return f"Saved:\nmin={payload['min'][i]}\nmax={payload['max'][i]}"

    return app
