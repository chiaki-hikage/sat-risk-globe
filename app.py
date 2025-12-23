# app.py
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skyfield.api import load, wgs84
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("3D Globe + Outlines + Moving Satellites (Local TLE + Local GeoJSON)")

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

MAP_FILES = {
    "110m": {
        "borders": os.path.join(MAP_DIR, "110m_borders.geojson"),
        "coast": os.path.join(MAP_DIR, "110m_coastline.geojson"),
    },
    "50m": {
        "borders": os.path.join(MAP_DIR, "50m_borders.geojson"),
        "coast": os.path.join(MAP_DIR, "50m_coastline.geojson"),
    },
    "10m": {
        "borders": os.path.join(MAP_DIR, "10m_borders.geojson"),
        "coast": os.path.join(MAP_DIR, "10m_coastline.geojson"),
    },
}

SAT_GROUPS = {
    "ISS / Stations (LEO)": {"type": "tle", "group": "stations", "note": "LEO, 動きが速くデモ向き"},
    "GNSS (GPS, Galileo etc.)": {"type": "tle", "group": "gnss", "note": "MEO, 測位衛星"},
    "Weather Satellites": {"type": "tle", "group": "weather", "note": "気象観測"},
    "Starlink (sample)": {"type": "tle", "group": "starlink_sample", "note": "一部のみ表示（PoC / ローカルサンプル）"},
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
            arr = np.asarray(coords, dtype=float)  # (N,2)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                lines.append(arr[:, :2])
        elif gtype == "MultiLineString":
            for part in coords:
                arr = np.asarray(part, dtype=float)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    lines.append(arr[:, :2])
        # Polygon等はここでは扱わない（軽量化のため）
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

    # local file -> no web access
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
# Dummy risk field
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
# Session state init
# ----------------------------
if "t_offset_sec" not in st.session_state:
    st.session_state.t_offset_sec = 0
if "scrub_offset_sec" not in st.session_state:
    st.session_state.scrub_offset_sec = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("Satellites")
    sat_label = st.selectbox("Satellite type", options=list(SAT_GROUPS.keys()), index=0)
    sat_conf = SAT_GROUPS[sat_label]

    max_sats = st.slider("Max satellites", 1, 300 if "Starlink" in sat_label else 150, 50)
    st.caption(sat_conf["note"])

    st.divider()
    st.header("Time")
    step_sec = st.select_slider("Step (seconds)", [5, 10, 30, 60, 120, 300], value=60)
    speed_x  = st.select_slider("Play speed (x)", [1, 2, 5, 10, 20, 50], value=10)

    colA, colB = st.columns(2)
    with colA:
        if st.button("⏮ Reset"):
            st.session_state.t_offset_sec = 0
            st.session_state.scrub_offset_sec = 0
    with colB:
        st.session_state.playing = st.toggle("▶ Play", value=st.session_state.playing)

    st.slider(
        "Jump offset from now (seconds)",
        min_value=-24*3600,
        max_value=24*3600,
        step=int(step_sec),
        key="scrub_offset_sec",
        help="Play停止中はこの値でジャンプします。Play中は内部カウンタで時間が進みます。"
    )

    st.divider()
    st.header("Map detail (local GeoJSON)")
    ne_res = st.selectbox("Resolution", ["110m", "50m", "10m"], index=1)
    show_surface = st.checkbox("Show globe shading", True)
    show_borders = st.checkbox("Show borders", True)
    show_coast   = st.checkbox("Show coastlines", True)

    default_stride = 12 if ne_res == "10m" else (8 if ne_res == "110m" else 6)
    outline_stride = st.slider("Line sampling stride", 1, 30, default_stride)

    st.divider()
    st.header("Risk layer (dummy)")
    show_risk = st.checkbox("Show risk layer", True)
    risk_alpha = st.slider("Risk opacity", 0.0, 1.0, 0.65, 0.05)
    risk_sigma_deg = st.slider("Spot radius (deg)", 2.0, 30.0, 10.0, 1.0)
    risk_sat_count = st.slider("Satellites used for risk", 0, max_sats, min(30, max_sats))
    risk_time_speed = st.select_slider("Risk time speed", [0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)

# ----------------------------
# Time progression
# ----------------------------
if st.session_state.playing:
    st_autorefresh(interval=1000, key="tick")
    st.session_state.t_offset_sec += int(step_sec) * int(speed_x)

offset_sec = int(st.session_state.t_offset_sec) if st.session_state.playing else int(st.session_state.scrub_offset_sec)

ts = get_timescale()
now_utc = datetime.now(timezone.utc)
target_dt = now_utc + timedelta(seconds=offset_sec)
t = ts.from_datetime(target_dt)

jst = timezone(timedelta(hours=9))
st.caption(
    f"Time: **{target_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC**"
    f"  /  **{target_dt.astimezone(jst).strftime('%Y-%m-%d %H:%M:%S')} JST**"
    f"  (offset={offset_sec}s)"
)

# ----------------------------
# Build globe
# ----------------------------
fig = go.Figure()

# Mesh (shared)
Lon, Lat, X, Y, Z = get_globe_mesh(res_lon=160, res_lat=80, R=1.0)

# Base sphere (always)
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

# Borders / Coast (local GeoJSON)
LINE_R = 1.008

if show_borders:
    path = MAP_FILES[ne_res]["borders"]
    lines, err = load_geojson_lines(path)
    if err:
        st.warning(err)
    else:
        bx, by, bz = lines_to_xyz(lines, stride=outline_stride, R=LINE_R)
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(width=2), hoverinfo="skip", name="Borders"))

if show_coast:
    path = MAP_FILES[ne_res]["coast"]
    lines, err = load_geojson_lines(path)
    if err:
        st.warning(err)
    else:
        cx, cy, cz = lines_to_xyz(lines, stride=outline_stride, R=LINE_R)
        fig.add_trace(go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", line=dict(width=2), hoverinfo="skip", name="Coast"))

# Satellites (local TLE)
if sat_conf["type"] == "tle":
    sats = load_tles_local(sat_conf["group"])
else:
    sats = []

# Risk layer (dummy; depends on sat subpoints + time)
if show_risk:
    if len(sats) > 0 and risk_sat_count > 0:
        _, sat_lats, sat_lons = compute_sat_latlon(sats, t, max_sats=max_sats)
        sat_lats = sat_lats[:risk_sat_count]
        sat_lons = sat_lons[:risk_sat_count]
    else:
        sat_lats = np.array([])
        sat_lons = np.array([])

    risk = dummy_risk_from_time_and_sats(
        Lat, Lon,
        t_seconds=target_dt.timestamp(),
        sat_lats=sat_lats,
        sat_lons=sat_lons,
        sigma_deg=risk_sigma_deg,
        time_speed=risk_time_speed,
    )

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=risk,
        showscale=False,
        opacity=float(risk_alpha),
        hoverinfo="skip",
        name="Risk"
    ))

# Satellite markers
sx, sy, sz, names = compute_sat_points(sats, t, max_sats=max_sats, R=1.03)
fig.add_trace(go.Scatter3d(
    x=sx, y=sy, z=sz,
    mode="markers",
    marker=dict(size=3, color=SAT_COLORS[sat_label]),
    text=names,
    hovertemplate="%{text}<extra></extra>",
    name="Satellites"
))

fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        camera=dict(eye=dict(x=0.8, y=0.8, z=0.9)),
    ),
    showlegend=False,
)

st.plotly_chart(fig, use_container_width=True)
st.caption("Drag to rotate / Scroll to zoom")
st.info("ローカルTLE/ローカルGeoJSONのみで動作します（外部アクセスなし）。")
