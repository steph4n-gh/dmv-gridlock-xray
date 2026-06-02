import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
import os
import json

st.set_page_config(layout="wide", page_title="DMV Gridlock X-Ray 3D")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Safe DC Anchor Coordinates to prevent Plotly Bounding Box from exploding on empty traces
DC_LON, DC_LAT = -77.0369, 38.9072

def inject_sim_node(node_id):
    sim_file = "sim_state.json"
    injected = []
    if os.path.exists(sim_file):
        with open(sim_file, "r") as f: injected = json.load(f).get("injected_nodes", [])
    if node_id not in injected:
        injected.append(node_id)
        with open(sim_file, "w") as f: json.dump({"injected_nodes": injected}, f)

def clear_sim():
    if os.path.exists("sim_state.json"): os.remove("sim_state.json")

def load_data_resilient():
    for _ in range(5):
        if os.path.exists("network_state.npz"):
            try:
                data = np.load("network_state.npz", allow_pickle=True)
                if 'node_friction' in data:
                    return data
            except: pass
        time.sleep(0.5)
    return None

def main():
    st.title("🛰️ DMV Topological Terrain X-Ray")
    
    if "pulse_history" not in st.session_state:
        st.session_state.pulse_history = []

    data = load_data_resilient()
    
    if data is None:
        st.warning("Waiting for data from engine.py... Ensure the engine is running in a terminal.")
        time.sleep(2)
        st.rerun()
        return

    # --- 1. SIDEBAR & CONTROLS ---
    with st.sidebar:
        st.header("🧪 Simulation Lab")
        node_names = data['names']; node_ids = data['nodes']
        sim_target = st.selectbox("Select Node for Injection:", options=[""] + list(node_names), index=0)
        c1, c2 = st.columns(2)
        if c1.button("Inject Critical Failure") and sim_target:
            t_idx = np.where(node_names == sim_target)[0][0]
            inject_sim_node(node_ids[t_idx])
            st.success(f"Injected: {sim_target}")
        if c2.button("Clear Simulations"):
            clear_sim()
            st.info("Cleared.")

        st.divider()
        st.header("🎮 Controls")
        viz_density = st.slider("Connection Density", 0.1, 1.0, 1.0, 0.1)
        fracture_threshold = st.slider("Fracture Sensitivity", 0.05, 0.8, 0.4, 0.05)
        
        st.divider()
        try:
            current_lambda = data['lambda_2'][0]
            st.session_state.pulse_history.append(current_lambda)
            if len(st.session_state.pulse_history) > 100: st.session_state.pulse_history.pop(0)
            
            if len(st.session_state.pulse_history) >= 2:
                diff = current_lambda - st.session_state.pulse_history[-2]
                trend = "↗️ Improving" if diff > 0 else "↘️ Degrading" if diff < 0 else "➡️ Stable"
            else:
                trend = "➡️ Stable"

            st.header("📊 System Pulse")
            st.metric("λ2 Health", f"{current_lambda:.6f}", delta=trend, delta_color="normal" if "Improving" in trend else "inverse" if "Degrading" in trend else "off")
            
            st.subheader("📍 Regional Pulse")
            st.write(f"🌤️ {data['weather_desc'][0]} ({data['weather_penalty'][0]:.2f})")
            c1, c2, c3 = st.columns(3)
            c1.metric("DC", data['incidents_dc'][0])
            c2.metric("MD", data['incidents_md'][0])
            c3.metric("VA", data['incidents_va'][0])
            st.metric("🚲 Bikeshare Depletion", data['bikeshare_depleted'][0])
            if 'active_alerts' in data:
                st.metric("📢 Active WMATA Alerts", data['active_alerts'][0])
        except: pass

        st.divider()
        st.header("🗺️ Map Legend")
        st.info(
            "**Large Red/Yellow Hubs**: Jammed Traffic (High Friction).\n\n"
            "**Faint Green Dots**: Flowing Traffic (Clear).\n\n"
            "**Z-Axis (Height)**: Isolation. High peaks are torn from the grid.\n\n"
            "**Thin Cyan Lines**: The baseline neural web.\n\n"
            "**Thick Pink/Red Lines**: Active Gridlock Fractures.\n\n"
            "**White Circles**: Bikeshare Depletion (High alternative demand)."
        )

    # --- 2. 3D MAP DATA PREPARATION ---
    coords = data['coords']; v_2 = data['v_2']; weights = data['weights']
    z_vals = v_2 * 150 
    centrality = data['centrality']; node_friction = data['node_friction']
    indptr, indices = data['indptr'], data['indices']
    depleted_idx = data['depleted_indices']

    fig = go.Figure()

    # TRACE 1: GROUND PLANE
    fig.add_trace(go.Scatter3d(
        x=coords[:, 1], y=coords[:, 0], z=np.zeros_like(z_vals),
        mode='markers', marker=dict(size=1.5, color='rgba(50,50,50,0.3)'),
        name="Ground Plane", hoverinfo='none'
    ))

    # TRACE 2: NEURAL WEB
    step = int(1.0 / max(viz_density, 0.01))
    ex, ey, ez = [], [], []
    for i in range(0, len(indptr)-1, step):
        for j in range(indptr[i], indptr[i+1]):
            target = indices[j]
            ex.extend([coords[i, 1], coords[target, 1], None])
            ey.extend([coords[i, 0], coords[target, 0], None])
            ez.extend([z_vals[i], z_vals[target], None])

    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color='rgba(0, 242, 255, 0.12)', width=1.0),
        name="Connectivity Web", hoverinfo='none'
    ))

    # TRACE 3 & 4: ACTIVE FRACTURES (Core and Halo)
    stressed_idx = np.where(weights < fracture_threshold)[0]
    if len(stressed_idx) > 0:
        fx, fy, fz = [], [], []
        for idx in stressed_idx:
            r = np.searchsorted(indptr, idx, side='right') - 1
            c = indices[idx]
            fx.extend([coords[r, 1], coords[c, 1], None])
            fy.extend([coords[r, 0], coords[c, 0], None])
            fz.extend([z_vals[r], z_vals[c], None])
        
        fig.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='lines', line=dict(color='rgba(255, 0, 85, 0.9)', width=4), name="Gridlock Core"))
        fig.add_trace(go.Scatter3d(x=fx, y=fy, z=fz, mode='lines', line=dict(color='rgba(255, 0, 85, 0.2)', width=12), name="Spectral Tear", hoverinfo='none'))
    else:
        # Anchor dummy traces to DC to prevent bounding box explosion
        fig.add_trace(go.Scatter3d(x=[DC_LON], y=[DC_LAT], z=[0], mode='lines', line=dict(color='rgba(0,0,0,0)', width=0), name="Gridlock Core", hoverinfo='none'))
        fig.add_trace(go.Scatter3d(x=[DC_LON], y=[DC_LAT], z=[0], mode='lines', line=dict(color='rgba(0,0,0,0)', width=0), name="Spectral Tear", hoverinfo='none'))

    # TRACE 5 & 6: SIGNAL-TO-NOISE NODES
    clear_idx = np.where(node_friction >= 0.85)[0]
    jammed_idx = np.where(node_friction < 0.85)[0]
    centrality_np = np.array(centrality)
    node_names_arr = np.array(data['names'])
    
    # Clear Flow
    if len(clear_idx) > 0:
        fig.add_trace(go.Scatter3d(
            x=coords[clear_idx, 1], y=coords[clear_idx, 0], z=z_vals[clear_idx],
            mode='markers', marker=dict(size=2, color='rgba(0, 255, 100, 0.15)'),
            text=node_names_arr[clear_idx], hoverinfo='text', name="Clear Flow"
        ))
    else:
        fig.add_trace(go.Scatter3d(x=[DC_LON], y=[DC_LAT], z=[0], mode='markers', marker=dict(size=0.1, color='rgba(0,0,0,0)'), name="Clear Flow", hoverinfo='none'))

    # Jammed Hubs
    if len(jammed_idx) > 0:
        jammed_sizes = np.clip(centrality_np[jammed_idx] * 1.5, 6, 25)
        fig.add_trace(go.Scatter3d(
            x=coords[jammed_idx, 1], y=coords[jammed_idx, 0], z=z_vals[jammed_idx],
            mode='markers',
            marker=dict(
                size=jammed_sizes, 
                color=node_friction[jammed_idx], 
                colorscale=[[0.0, 'red'], [1.0, 'yellow']], 
                cmin=0.0, cmax=0.85,
                opacity=1.0, line=dict(color='white', width=1),
                colorbar=dict(title="Jammed Flow", thickness=15, x=1.05)
            ),
            text=node_names_arr[jammed_idx], hoverinfo='text', name="Jammed Hubs"
        ))
    else:
        fig.add_trace(go.Scatter3d(x=[DC_LON], y=[DC_LAT], z=[0], mode='markers', marker=dict(size=0.1, color='rgba(0,0,0,0)'), name="Jammed Hubs", hoverinfo='none'))

    # TRACE 7: MULTIMODAL OVERLAY (Bikeshare Demand)
    if len(depleted_idx) > 0:
        fig.add_trace(go.Scatter3d(
            x=coords[depleted_idx, 1], y=coords[depleted_idx, 0], z=z_vals[depleted_idx],
            mode='markers', marker=dict(size=12, color='white', symbol='circle-open', line=dict(color='white', width=2.5)),
            name="Bikeshare Depletion", hoverinfo='none'
        ))
    else:
        fig.add_trace(go.Scatter3d(x=[DC_LON], y=[DC_LAT], z=[0], mode='markers', marker=dict(size=0.1, color='rgba(0,0,0,0)'), name="Bikeshare Depletion", hoverinfo='none'))

    # --- 3. LAYOUT & RENDERING ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Lon', backgroundcolor="black", showgrid=False, zeroline=False),
            yaxis=dict(title='Lat', backgroundcolor="black", showgrid=False, zeroline=False),
            zaxis=dict(title='Isolation', backgroundcolor="black", showgrid=False, zeroline=False),
            aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.4),
        ),
        # THE MAGIC BULLETS FOR CAMERA LOCKING
        uirevision='locked_camera_forever',
        scene_uirevision='locked_camera_forever',
        margin=dict(r=0, l=0, b=0, t=0),
        height=950,
        template="plotly_dark",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white"))
    )

    st.plotly_chart(fig, width='stretch', key="dmv_stable_xray", theme=None)
    st.caption(f"🚀 Render Density: {viz_density*100:.0f}% | Auto-Refreshing Every 15s")

    # --- 4. THE LOOP TRIGGER ---
    time.sleep(15)
    st.rerun()

if __name__ == "__main__":
    main()
