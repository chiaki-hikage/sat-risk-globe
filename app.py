# app.py
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skyfield.api import load, wgs84

# ----------------------------
# Page
# ----------------------------
st.set_page_config(layout="wide")
st.title("衛星軌道と宇宙天気の可視化デモ")

# ----------------------------
# Paths (local)
# ----------------------------
TLE_DIR = os.path.join("data", "tle")
MAP_DIR = os.path.join("data", "map")

TLE_FILES = {
    "stations": os.path.join(TLE_DIR, "stations.tle"),
    "gnss": os.path.join(TLE_DIR, "gnss.tle"),
    "weather": os.path.join(TLE_DIR, "weather.tle"),
    "starlink_sample": os.path.join(TLE_DIR, "starlink_sample.tle"),
}

# Resolution is fixed to 50m per request
NE_RES_FIXED = "50m"
MAP_FILES = {
    "50m": {
        "borders": os.path.join(MAP_DIR, "50m_borders.geojson"),
        "coast": os.path.join(MAP_DIR, "50m_coastline.geojson"),
    },
}

SAT_GROUPS = {
    "ISS / Stations (LEO)": {"type": "tle", "group": "stations", "note": "ISSや地上局関連の低軌道衛星"},
    "GNSS (GPS, Galileo etc.)": {"type": "tle", "group": "gnss", "note": "中軌道を周回する測位衛星群"},
    "Weather Satellites": {"type": "tle", "group": "weather", "note": "気象観測衛星（静止軌道および低軌道）"},
    "Starlink (sample)": {"type": "tle", "group": "starlink_sample", "note": "Starlink 衛星群の一部（低軌道衛星"},
}

SAT_COLORS = {
    "ISS / Stations (LEO)": "red",
    "GNSS (GPS, Galileo etc.)": "blue",
    "Weather Satellites": "green",
    "Starlink (sample)": "orange",
}

# ----------------------------
# Helpers (math / mesh)
# ----------------------------
def ll_to_xyz(lat_deg, lon_deg, R=1.0):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z

@st.cache_data(show_spinner=False)
def get_globe_mesh(res_lon=160, res_lat=80, R=1.0):
    lon = np.linspace(-180, 180, res_lon)
    lat = np.linspace(-90, 90, res_lat)
    Lon, Lat = np.meshgrid(lon, lat)
    X, Y, Z = ll_to_xyz(Lat, Lon, R=R)
    return Lon, Lat, X, Y, Z

# ----------------------------
# Helpers (GeoJSON lines)
# ----------------------------
@st.cache_data(show_spinner=False)
def load_geojson_lines(path: str):
    """
    Read a local GeoJSON and return list of polylines.
    Each polyline is Nx2 array of [lon, lat].
    Supports LineString / MultiLineString.
    """
    if not os.path.exists(path):
        return None, f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    lines = []
    feats = gj.get("features", [])
    for feat in feats:
        geom = feat.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if not coords:
            continue

        if gtype == "LineString":
            arr = np.asarray(coords, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                lines.append(arr[:, :2])
        elif gtype == "MultiLineString":
            for part in coords:
                arr = np.asarray(part, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    lines.append(arr[:, :2])
    return lines, None

def lines_to_xyz(lines, stride=8, R=1.008):
    xs, ys, zs = [], [], []
    if not lines:
        return xs, ys, zs

    for arr in lines:
        if arr.shape[0] < 2:
            continue
        pts = arr[::stride] if stride > 1 else arr
        if pts.shape[0] < 2:
            continue
        lons = pts[:, 0]
        lats = pts[:, 1]
        x, y, z = ll_to_xyz(lats, lons, R=R)
        xs += x.tolist() + [None]
        ys += y.tolist() + [None]
        zs += z.tolist() + [None]
    return xs, ys, zs

@st.cache_data(show_spinner=False)
def get_lines_xyz_cached(kind: str, stride: int, R: float):
    # kind: "borders" or "coast"
    path = MAP_FILES[NE_RES_FIXED]["borders"] if kind == "borders" else MAP_FILES[NE_RES_FIXED]["coast"]
    lines, err = load_geojson_lines(path)
    if err:
        return [], [], [], err
    xs, ys, zs = lines_to_xyz(lines, stride=stride, R=R)
    return xs, ys, zs, None

# ----------------------------
# Helpers (TLE local)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_timescale():
    return load.timescale()

@st.cache_resource(show_spinner=False)
def load_tles_local(group: str):
    path = TLE_FILES.get(group)
    if path is None:
        raise ValueError(f"Unknown TLE group: {group}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"TLE file not found: {path}\n"
            f"Please put local TLE at data/tle/*.tle (see TLE_FILES mapping)."
        )
    return load.tle_file(path, reload=False)

def compute_sat_points(sats, t, max_sats=60, R=1.03):
    rows = []
    for sat in sats[:max_sats]:
        pos = sat.at(t)
        lat, lon = wgs84.latlon_of(pos)
        rows.append((sat.name, lat.degrees, lon.degrees))

    xs, ys, zs, names = [], [], [], []
    for name, lat, lon in rows:
        x, y, z = ll_to_xyz(lat, lon, R=R)
        xs.append(x); ys.append(y); zs.append(z); names.append(name)
    return xs, ys, zs, names

def compute_sat_latlon(sats, t, max_sats=60):
    names, lats, lons = [], [], []
    for sat in sats[:max_sats]:
        pos = sat.at(t)
        lat, lon = wgs84.latlon_of(pos)
        names.append(sat.name)
        lats.append(lat.degrees)
        lons.append(lon.degrees)
    return names, np.array(lats), np.array(lons)

# ----------------------------
# Dummy risk field (time + sats)
# ----------------------------
def dummy_risk_from_time_and_sats(Lat, Lon, t_seconds, sat_lats, sat_lons,
                                 aurora_amp=0.6, sat_amp=1.0,
                                 sigma_deg=8.0, time_speed=1.0):
    lat = np.radians(Lat)
    lon = np.radians(Lon)

    # (A) time-varying high-lat band
    phase = (t_seconds * 2*np.pi / 600.0) * time_speed
    aurora_band = np.exp(-((np.abs(lat) - np.radians(65)) ** 2) / (2 * np.radians(10) ** 2))
    wave = 0.5 + 0.5 * np.sin(3 * lon + phase)
    risk_bg = aurora_amp * aurora_band * wave

    # (B) sat spots
    risk_sat = 0.0
    if sat_lats.size > 0:
        sat_lat = np.radians(sat_lats)[None, None, :]
        sat_lon = np.radians(sat_lons)[None, None, :]
        lat_g = lat[:, :, None]
        lon_g = lon[:, :, None]

        cosd = np.sin(lat_g)*np.sin(sat_lat) + np.cos(lat_g)*np.cos(sat_lat)*np.cos(lon_g - sat_lon)
        cosd = np.clip(cosd, -1.0, 1.0)
        d = np.arccos(cosd)

        sigma = np.radians(sigma_deg)
        blobs = np.exp(-(d**2) / (2 * sigma**2))
        risk_sat = sat_amp * np.sum(blobs, axis=2)

        mx = np.max(risk_sat)
        if mx > 0:
            risk_sat = risk_sat / mx

    risk = risk_bg + risk_sat
    risk = np.clip(risk, 0, None)
    mx = np.max(risk)
    if mx > 0:
        risk = risk / mx
    return np.clip(risk, 0, 1)

# ----------------------------
# Build static background + frames (satellites + risk)
# ----------------------------
def build_figure_with_frames(
    sats,
    ts,
    start_dt_utc: datetime,
    frame_step_sec: int,
    n_frames: int,
    max_sats: int,
    sat_color: str,
    show_surface: bool,
    show_borders: bool,
    show_coast: bool,
    outline_stride: int,
    show_risk: bool,
    risk_alpha: float,
    risk_sigma_deg: float,
    risk_sat_count: int,
    risk_time_speed: float,
    aurora_amp: float,
    sat_amp: float,
    play_speed_ms: int,
    globe_res_lon: int = 160,
    globe_res_lat: int = 80,
):
    fig = go.Figure()

    # Mesh
    Lon, Lat, X, Y, Z = get_globe_mesh(res_lon=globe_res_lon, res_lat=globe_res_lat, R=1.0)

    # Base sphere
    base = np.zeros_like(X)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=base,
        showscale=False,
        opacity=0.22,
        hoverinfo="skip",
        name="Base"
    ))

    # Optional shading
    if show_surface:
        surface_color = np.clip((np.abs(Lat) - 20) / 70, 0, 1)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=surface_color,
            showscale=False,
            opacity=0.55,
            hoverinfo="skip",
            name="Shading"
        ))

    # Borders / Coastlines (same color per request)
    LINE_R = 1.008
    outline_color = "rgba(230,230,255,1.0)"  # borders and coastlines identical
    outline_width = 3

    if show_borders:
        bx, by, bz, err = get_lines_xyz_cached("borders", int(outline_stride), LINE_R)
        if err:
            st.warning(err)
        else:
            fig.add_trace(go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="lines",
                line=dict(width=outline_width, color=outline_color),
                hoverinfo="skip",
                name="Borders"
            ))

    if show_coast:
        cx, cy, cz, err = get_lines_xyz_cached("coast", int(outline_stride), LINE_R)
        if err:
            st.warning(err)
        else:
            fig.add_trace(go.Scatter3d(
                x=cx, y=cy, z=cz,
                mode="lines",
                line=dict(width=outline_width, color=outline_color),
                hoverinfo="skip",
                name="Coastlines"
            ))

    # Risk surface (initial)
    risk_trace_index = None
    t0_dt = start_dt_utc
    t0 = ts.from_datetime(t0_dt)

    if show_risk:
        if len(sats) > 0 and risk_sat_count > 0:
            _, sat_lats, sat_lons = compute_sat_latlon(sats, t0, max_sats=max_sats)
            sat_lats = sat_lats[:risk_sat_count]
            sat_lons = sat_lons[:risk_sat_count]
        else:
            sat_lats = np.array([])
            sat_lons = np.array([])

        risk0 = dummy_risk_from_time_and_sats(
            Lat, Lon,
            t_seconds=t0_dt.timestamp(),
            sat_lats=sat_lats,
            sat_lons=sat_lons,
            aurora_amp=aurora_amp,
            sat_amp=sat_amp,
            sigma_deg=risk_sigma_deg,
            time_speed=risk_time_speed,
        )

        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            surfacecolor=risk0,
            showscale=False,
            opacity=float(risk_alpha),
            hoverinfo="skip",
            name="Risk"
        ))
        risk_trace_index = len(fig.data) - 1

    # Satellites (initial)
    sx, sy, sz, names = compute_sat_points(sats, t0, max_sats=max_sats, R=1.03)
    fig.add_trace(go.Scatter3d(
        x=sx, y=sy, z=sz,
        mode="markers",
        marker=dict(size=3, color=sat_color),
        text=names,
        hovertemplate="%{text}<extra></extra>",
        name="Satellites"
    ))
    sat_trace_index = len(fig.data) - 1

    # Frames: update satellites (+ risk if enabled)
    frames = []
    jst = timezone(timedelta(hours=9))

    for k in range(n_frames):
        dt_k = start_dt_utc + timedelta(seconds=k * frame_step_sec)
        t_k = ts.from_datetime(dt_k)

        sxk, syk, szk, _ = compute_sat_points(sats, t_k, max_sats=max_sats, R=1.03)

        frame_data = []
        frame_traces = []

        # (1) risk update
        if show_risk and risk_trace_index is not None:
            if len(sats) > 0 and risk_sat_count > 0:
                _, sat_lats, sat_lons = compute_sat_latlon(sats, t_k, max_sats=max_sats)
                sat_lats = sat_lats[:risk_sat_count]
                sat_lons = sat_lons[:risk_sat_count]
            else:
                sat_lats = np.array([])
                sat_lons = np.array([])

            risk_k = dummy_risk_from_time_and_sats(
                Lat, Lon,
                t_seconds=dt_k.timestamp(),
                sat_lats=sat_lats,
                sat_lons=sat_lons,
                aurora_amp=aurora_amp,
                sat_amp=sat_amp,
                sigma_deg=risk_sigma_deg,
                time_speed=risk_time_speed,
            )

            # Update only surfacecolor (3D is safer with redraw=True)
            frame_data.append(go.Surface(surfacecolor=risk_k, opacity=float(risk_alpha), showscale=False))
            frame_traces.append(risk_trace_index)

        # (2) satellite update
        frame_data.append(go.Scatter3d(x=sxk, y=syk, z=szk))
        frame_traces.append(sat_trace_index)

        frames.append(go.Frame(
            name=str(k),
            data=frame_data,
            traces=frame_traces,
        ))

    fig.frames = frames

    # Slider labels (JST HH:MM)
    steps = []
    for k in range(n_frames):
        dt_k = start_dt_utc + timedelta(seconds=k * frame_step_sec)
        label = dt_k.astimezone(jst).strftime("%H:%M")
        steps.append({
            "args": [[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": label,
            "method": "animate"
        })

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision="globe_static",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(
                eye=dict(
                x=-1.0,   # 日本（東アジア）方向
                y=0.8,
                z=0.6
            )),
        ),
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.02,
            "y": 0.02,
            "xanchor": "left",
            "yanchor": "bottom",
            "showactive": False,
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"color": "black", "size": 13},
            "pad": {"r": 10, "t": 10},
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": int(play_speed_ms), "redraw": True},
                            "transition": {"duration": 0},
                            "fromcurrent": True,
                            "mode": "immediate",
                        }
                    ],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                            "mode": "immediate",
                        },
                    ],
                },
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.02,
            "y": -0.2,
            "xanchor": "left",
            "yanchor": "bottom",
            "len": 0.96,
            "pad": {"b": 0, "t": 0},
            "currentvalue": {"prefix": "JST(HH:MM): "},
            "steps": steps,
        }]
    )

    return fig

# ----------------------------
# Sidebar UI
# ----------------------------
with st.sidebar:

    generate = st.button("🎬 Generate animation", type="primary")

    st.divider()
    st.header("Satellites")
    sat_options = list(SAT_GROUPS.keys())
    sat_label = st.selectbox(
        "Satellite type",
        options=sat_options,
        index=sat_options.index("Starlink (sample)")
    )
    sat_conf = SAT_GROUPS[sat_label]
    st.caption(sat_conf["note"])
    max_sats = st.slider("Max satellites", 1, 300 if "Starlink" in sat_label else 150, 10)

    st.divider()
    st.header("Animation (Plotly frames)")
    frame_step_sec = st.select_slider("Frame step (seconds)", options=[10, 30, 60, 120, 300], value=60)
    n_frames = st.slider("#frames", 5, 120, 20, help="例: step=60秒, frames=20 なら20分ぶん")
    play_speed_ms = st.slider("Play frame duration (ms)", 50, 1500, 500, 50,
                              help="小さいほど速く再生（重い場合は大きく）")

    st.divider()
    st.header("Start time")
    start_offset_sec = st.slider(
        "Start offset from now (seconds)",
        min_value=-24*3600,
        max_value=24*3600,
        step=60,
        value=0,
        help="動画の開始時刻を現在(UTC)からずらします"
    )

    #st.divider()
    #st.header("Map (50m fixed)")
    show_surface = True
    show_borders = True
    show_coast   = True
    outline_stride = 5

    st.divider()
    st.header("Risk layer (dummy)")
    show_risk = st.checkbox("Show risk layer", True)
    risk_alpha = st.slider("Risk opacity", 0.0, 1.0, 0.65, 0.05)
    risk_sigma_deg = st.slider("Spot radius (deg)", 2.0, 30.0, 10.0, 1.0)
    risk_sat_count = st.slider("Satellites used for risk", 0, max_sats, min(30, max_sats))
    risk_time_speed = st.select_slider("Risk time speed", [0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
    aurora_amp = st.slider("Aurora band amp", 0.0, 2.0, 0.6, 0.05)
    sat_amp = st.slider("Satellite spots amp", 0.0, 2.0, 1.0, 0.05)

    #st.divider()
    #st.header("Performance")
    #globe_res_lon = st.select_slider("Globe grid res_lon", options=[80, 120, 160, 200], value=160)
    globe_res_lon = 80
    #globe_res_lat = st.select_slider("Globe grid res_lat", options=[40, 60, 80, 100], value=80)
    globe_res_lat = 40

# ----------------------------
# Load satellites
# ----------------------------
ts = get_timescale()
if sat_conf["type"] == "tle":
    sats = load_tles_local(sat_conf["group"])
else:
    sats = []

# ----------------------------
# Show time range
# ----------------------------
now_utc = datetime.now(timezone.utc)
start_dt = now_utc + timedelta(seconds=int(start_offset_sec))
jst = timezone(timedelta(hours=9))
end_dt = start_dt + timedelta(seconds=int(frame_step_sec) * (int(n_frames) - 1))

st.caption(
    f"Start: **{start_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC** / **{start_dt.astimezone(jst).strftime('%Y-%m-%d %H:%M:%S')} JST**  |  "
    f"End: **{end_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC** / **{end_dt.astimezone(jst).strftime('%Y-%m-%d %H:%M:%S')} JST**"
)

# ----------------------------
# Generate & Show
# ----------------------------
if not generate:
    st.info("サイドバーで設定して **Generate animation** を押すと、衛星点＋リスク面が時間で動くアニメを生成します。")
else:
    with st.spinner("Building frames (satellites + risk)..."):
        fig = build_figure_with_frames(
            sats=sats,
            ts=ts,
            start_dt_utc=start_dt,
            frame_step_sec=int(frame_step_sec),
            n_frames=int(n_frames),
            max_sats=int(max_sats),
            sat_color=SAT_COLORS[sat_label],
            show_surface=show_surface,
            show_borders=show_borders,
            show_coast=show_coast,
            outline_stride=int(outline_stride),
            show_risk=show_risk,
            risk_alpha=float(risk_alpha),
            risk_sigma_deg=float(risk_sigma_deg),
            risk_sat_count=int(risk_sat_count),
            risk_time_speed=float(risk_time_speed),
            aurora_amp=float(aurora_amp),
            sat_amp=float(sat_amp),
            play_speed_ms=int(play_speed_ms),
            globe_res_lon=int(globe_res_lon),
            globe_res_lat=int(globe_res_lat),
        )

    st.plotly_chart(fig, use_container_width=True, key="globe_animation")
    st.caption("Drag to rotate / Scroll to zoom / ▶ Play で再生（背景固定、衛星＋リスクのみ更新）")
