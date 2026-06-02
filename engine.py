import os
import asyncio
import aiohttp
import time
import zipfile
import io
import requests
import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
import networkx as nx
from google.transit import gtfs_realtime_pb2
import json
import csv
import math

POTOMAC_BARRIER = [
    ((38.995, -77.162), (38.960, -77.130)),
    ((38.960, -77.130), (38.930, -77.115)),
    ((38.930, -77.115), (38.900, -77.070)),
    ((38.900, -77.070), (38.888, -77.060)),
    ((38.888, -77.060), (38.875, -77.043)),
    ((38.875, -77.043), (38.850, -77.040)),
    ((38.850, -77.040), (38.790, -77.035))
]

ANACOSTIA_BARRIER = [
    ((38.935, -76.940), (38.915, -76.955)),
    ((38.915, -76.955), (38.900, -76.965)),
    ((38.900, -76.965), (38.875, -76.980)),
    ((38.875, -76.980), (38.860, -77.010)),
    ((38.860, -77.010), (38.858, -77.025))
]

def ccw(A, B, C):
    return (C[0] - A[0]) * (B[1] - A[1]) > (B[0] - A[0]) * (C[1] - A[1])

def segments_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def crosses_river(lat1, lon1, lat2, lon2):
    p1 = (lat1, lon1)
    p2 = (lat2, lon2)
    for seg in POTOMAC_BARRIER:
        if segments_intersect(p1, p2, seg[0], seg[1]):
            return True
    for seg in ANACOSTIA_BARRIER:
        if segments_intersect(p1, p2, seg[0], seg[1]):
            return True
    return False

def smooth_floor(x, k=100.0):
    if isinstance(x, np.ndarray):
        kx = k * x
        return np.where(kx > 50.0, x, np.log1p(np.exp(np.clip(kx, -50.0, 50.0))) / k)
    else:
        kx = k * x
        if kx > 50.0:
            return x
        return math.log1p(math.exp(kx)) / k

# --- CONFIGURATION ---
API_KEY = os.environ.get("WMATA_API_KEY", "YOUR_API_KEY_HERE")
GTFS_DIR = "gtfs"
STATIC_GTFS_URL = "https://api.wmata.com/gtfs/bus-gtfs-static.zip"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast?latitude=38.9072&longitude=-77.0369&current_weather=true&hourly=precipitation"
LOG_FILE = "telemetry_history.csv"
SCOREBOARD_FILE = "prediction_scoreboard.csv"

# Regional Incident Feeds
INCIDENTS_DC = "https://maps2.dcgis.dc.gov/dcgis/rest/services/DDOT/HSEMA_RoadClosures/MapServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
INCIDENTS_MD = "https://chartimap1.sha.maryland.gov/arcgis/rest/services/CHART/Incidents/MapServer/0/query?where=County+%3D+%27Montgomery+County%27+OR+County+%3D+%27Prince+George%27%27s+County%27&outSR=4326&f=geojson"
INCIDENTS_VA = ""  # Bypassed due to SmarterRoads auth constraints / local DNS blocks

# Capital Bikeshare GBFS
BIKESHARE_INFO = "https://gbfs.capitalbikeshare.com/gbfs/en/station_information.json"
BIKESHARE_STATUS = "https://gbfs.capitalbikeshare.com/gbfs/en/station_status.json"

POLL_INTERVAL_GTFS = 30
POLL_INTERVAL_WEATHER = 900
POLL_INTERVAL_INCIDENTS = 300
POLL_INTERVAL_BIKESHARE = 120
K_FRICTION = 0.005
ALPHA_DIFFUSION = 0.2
GAMMA_CENTRALITY = 0.5

# --- RATE LIMITER GOVERNOR ---
class RateLimiter:
    def __init__(self, state_file="rate_limit_state.json"):
        self.state_file = state_file
        self.link_history = {} # URL -> list of timestamps
        self.overall_history = [] # list of timestamps
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    raw_link_history = data.get("link_history", {})
                    raw_overall_history = data.get("overall_history", [])
                else:
                    raw_link_history = {}
                    raw_overall_history = []
                    
                # Defensive sanitization and validation
                self.link_history = {}
                if isinstance(raw_link_history, dict):
                    for url, timestamps in raw_link_history.items():
                        if isinstance(timestamps, list):
                            valid_ts = []
                            for t in timestamps:
                                try:
                                    valid_ts.append(float(t))
                                except (ValueError, TypeError):
                                    pass
                            if valid_ts:
                                self.link_history[str(url)] = valid_ts
                                
                self.overall_history = []
                if isinstance(raw_overall_history, list):
                    for t in raw_overall_history:
                        try:
                            self.overall_history.append(float(t))
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                print(f"⚠️ Warning: Failed to load rate limit state file: {e}")
                self.link_history = {}
                self.overall_history = []

    def save_state(self):
        try:
            # Atomic write pattern to prevent file corruption during dropouts
            tmp_file = self.state_file + ".tmp"
            with open(tmp_file, "w") as f:
                json.dump({
                    "link_history": self.link_history,
                    "overall_history": self.overall_history
                }, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.state_file)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save rate limit state atomically: {e}")

    def clean_history(self, now):
        cutoff = now - 60.0
        for url in list(self.link_history.keys()):
            cleaned = [t for t in self.link_history[url] if t > cutoff]
            if cleaned:
                self.link_history[url] = cleaned
            else:
                del self.link_history[url] # Prevent memory and state file leakage/bloating
        self.overall_history = [t for t in self.overall_history if t > cutoff]

    def can_request(self, url, now=None):
        if now is None: now = time.time()
        self.clean_history(now)
        
        # Rule 1: <= 3 requests/min/link
        url_history = self.link_history.get(url, [])
        if len(url_history) >= 3: return False
        
        # Rule 2: min 20s interval between consecutive requests to same link
        if url_history and (now - url_history[-1]) < 20.0: return False
        
        # Rule 3: <= 10 requests/min overall
        if len(self.overall_history) >= 10: return False
        
        return True

    def record_request(self, url, now=None):
        if now is None: now = time.time()
        if url not in self.link_history: self.link_history[url] = []
        self.link_history[url].append(now)
        self.overall_history.append(now)
        self.clean_history(now)
        self.save_state()

rate_limiter = RateLimiter()

# --- GLOBAL STATE ---
state = {
    "weather_penalty": 1.0,
    "weather_desc": "Clear",
    "precipitation_rate": 0.0,
    "history_bikes": {},
    "history_distances": {},
    "weather_ema": {"V": 14.5, "P": 0.0, "VP": 0.0, "P2": 0.0},
    "incidents": {"dc": [], "md": [], "va": []},
    "injections": [], 
    "bikeshare": {"total_bikes": 0, "active_stations": 0, "depleted_stations": 0, "depleted_node_indices": [], "node_to_metadata": {}},
    "gtfs_delays": {},
    "trip_delays": {},
    "trip_to_shape": {},
    "live_speeds": {}, # stop_id -> current speed in mph
    "live_speeds_list": {}, # stop_id -> list of raw speeds reported
    "live_buses": {}, # route_id -> list of vehicle_ids
    "predictive_friction": {},
    "wmata_official_alerts": set(), 
    "rail_alerts": 0,
    "rail_surges": set(),
    # Track officially alerted route_ids
    "canary_buses": {}, # vehicle_id -> {route_id, target_u, target_v, status, start_speed}
    "active_predictions": {}, # route_id -> {t_engine, t_physical, t_wmata, cause, resolved}
    "bus_positions": [], # list of active bus GPS locations and bearings
    "bus_positions_dict": {}, # vehicle_id -> position info
    "train_positions": [], # list of active train GPS locations, speeds, and stations
    "prev_trains": {}, # vehicle_id -> (lat, lon, timestamp, speed_mph)
    "scoreboard_stats": {"xray_wins": 0, "wmata_wins": 0, "avg_lead_time_sec": 0, "total_races": 0},
    "last_payload_ts": 0,
    "engine_start": time.time(),
    "total_cycles": 0,
    "graph_stats": {
        "avg_friction": 1.0, "spectral_gap": 0.0, "active_nodes": 0, "active_edges": 0,
        "system_tension": 0.0, "total_delay_sec": 0, "compute_latency": 0.0,
        "peak_latency": 0.0, "fiedler_max": 0.0, "fiedler_pole_name": "Core",
        "worst_node_name": "None", "centrality": [], "active_alerts": 0 
    }
}

def init_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "lambda_2", "spectral_gap", "avg_friction", "incidents_dc", "incidents_md", "incidents_va", "bikeshare_depleted", "latency_ms"])
    if not os.path.exists(SCOREBOARD_FILE):
        with open(SCOREBOARD_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "route_id", "cause", "t_engine", "t_physical", "t_wmata", "lead_time_sec", "winner"])

def log_telemetry(ts, l2, gap, friction, dc, md, va, bike, lat):
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([ts, l2, gap, friction, dc, md, va, bike, lat])
    except: pass

def download_gtfs_static():
    print(f"Downloading GTFS Static from {STATIC_GTFS_URL}...")
    headers = {"api_key": API_KEY}
    try:
        response = requests.get(STATIC_GTFS_URL, headers=headers)
        response.raise_for_status()
        if not os.path.exists(GTFS_DIR): os.makedirs(GTFS_DIR)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z: z.extractall(GTFS_DIR)
    except Exception as e: print(f"❌ Failed: {e}"); raise

def download_rideon_gtfs_static():
    rideon_dir = os.path.join(GTFS_DIR, "rideon")
    required_files = ["stops.txt", "stop_times.txt", "trips.txt"]
    files_exist = all(os.path.exists(os.path.join(rideon_dir, f)) for f in required_files)
    if files_exist:
        print("RideOn static GTFS files already exist. Skipping download.")
        return True
    
    url = "https://www.montgomerycountymd.gov/DOT-Transit/Resources/Files/GTFS/RideOnGTFS.zip"
    print(f"Downloading RideOn GTFS Static from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        os.makedirs(rideon_dir, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(rideon_dir)
        print("RideOn static GTFS downloaded and extracted successfully.")
        return True
    except (requests.exceptions.RequestException, zipfile.BadZipFile) as e:
        print(f"⚠️ Failed to download/extract RideOn GTFS: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Unexpected error downloading RideOn GTFS: {e}")
        return False

def load_static_topology():
    if not os.path.exists(os.path.join(GTFS_DIR, "stops.txt")): download_gtfs_static()
    wmata_stops_df = pd.read_csv(os.path.join(GTFS_DIR, "stops.txt"), low_memory=False)
    wmata_stop_times_df = pd.read_csv(os.path.join(GTFS_DIR, "stop_times.txt"), low_memory=False)
    wmata_trips_df = pd.read_csv(os.path.join(GTFS_DIR, "trips.txt"), low_memory=False)
    
    # Prefix WMATA stops_df
    wmata_stops_df['stop_id'] = 'wmata_' + wmata_stops_df['stop_id'].astype(str)
    if 'parent_station' in wmata_stops_df.columns:
        wmata_stops_df['parent_station'] = wmata_stops_df['parent_station'].apply(
            lambda x: 'wmata_' + str(x) if pd.notna(x) and str(x).strip() != '' else x
        )
    
    # Prefix WMATA stop_times_df
    wmata_stop_times_df['stop_id'] = 'wmata_' + wmata_stop_times_df['stop_id'].astype(str)
    wmata_stop_times_df['trip_id'] = 'wmata_' + wmata_stop_times_df['trip_id'].astype(str)
    
    # Prefix WMATA trips_df
    wmata_trips_df['trip_id'] = 'wmata_' + wmata_trips_df['trip_id'].astype(str)
    wmata_trips_df['route_id'] = 'wmata_' + wmata_trips_df['route_id'].astype(str)
    if 'shape_id' in wmata_trips_df.columns:
        wmata_trips_df['shape_id'] = wmata_trips_df['shape_id'].apply(
            lambda x: 'wmata_' + str(x) if pd.notna(x) and str(x).strip() != '' else x
        )
    
    stops_dfs = [wmata_stops_df]
    stop_times_dfs = [wmata_stop_times_df]
    trips_dfs = [wmata_trips_df]
    
    # Check if RideOn static files exist
    rideon_dir = os.path.join(GTFS_DIR, "rideon")
    rideon_stops_path = os.path.join(rideon_dir, "stops.txt")
    rideon_stop_times_path = os.path.join(rideon_dir, "stop_times.txt")
    rideon_trips_path = os.path.join(rideon_dir, "trips.txt")
    
    required_files = ["stops.txt", "stop_times.txt", "trips.txt"]
    files_exist = all(os.path.exists(os.path.join(rideon_dir, f)) for f in required_files)
    if not files_exist:
        download_rideon_gtfs_static()
        
    if os.path.exists(rideon_stops_path) and os.path.exists(rideon_stop_times_path) and os.path.exists(rideon_trips_path):
        print("Fusing RideOn static topology...")
        rideon_stops_df = pd.read_csv(rideon_stops_path, low_memory=False)
        rideon_stop_times_df = pd.read_csv(rideon_stop_times_path, low_memory=False)
        rideon_trips_df = pd.read_csv(rideon_trips_path, low_memory=False)
        
        # Prefix RideOn stops_df
        rideon_stops_df['stop_id'] = 'rideon_' + rideon_stops_df['stop_id'].astype(str)
        if 'parent_station' in rideon_stops_df.columns:
            rideon_stops_df['parent_station'] = rideon_stops_df['parent_station'].apply(
                lambda x: 'rideon_' + str(x) if pd.notna(x) and str(x).strip() != '' else x
            )
        
        # Prefix RideOn stop_times_df
        rideon_stop_times_df['stop_id'] = 'rideon_' + rideon_stop_times_df['stop_id'].astype(str)
        rideon_stop_times_df['trip_id'] = 'rideon_' + rideon_stop_times_df['trip_id'].astype(str)
        
        # Prefix RideOn trips_df
        rideon_trips_df['trip_id'] = 'rideon_' + rideon_trips_df['trip_id'].astype(str)
        rideon_trips_df['route_id'] = 'rideon_' + rideon_trips_df['route_id'].astype(str)
        if 'shape_id' in rideon_trips_df.columns:
            rideon_trips_df['shape_id'] = rideon_trips_df['shape_id'].apply(
                lambda x: 'rideon_' + str(x) if pd.notna(x) and str(x).strip() != '' else x
            )
        
        stops_dfs.append(rideon_stops_df)
        stop_times_dfs.append(rideon_stop_times_df)
        trips_dfs.append(rideon_trips_df)
    else:
        print("RideOn static topology files not found in gtfs/rideon/, proceeding with WMATA only.")
        
    stops_df = pd.concat(stops_dfs, ignore_index=True)
    stop_times_df = pd.concat(stop_times_dfs, ignore_index=True)
    trips_df = pd.concat(trips_dfs, ignore_index=True)
    
    stops_info = {}
    for _, row in stops_df.iterrows():
        stops_info[str(row['stop_id'])] = {
            'name': row['stop_name'],
            'lat': float(row['stop_lat']),
            'lon': float(row['stop_lon'])
        }
        
    all_stop_ids = stops_df['stop_id'].astype(str).unique()
    s_to_i = {s_id: i for i, s_id in enumerate(all_stop_ids)}
    n_all = len(all_stop_ids)
    st_df = stop_times_df.sort_values(by=['trip_id', 'stop_sequence'])
    
    trip_to_route = dict(zip(trips_df['trip_id'], trips_df['route_id']))
    st_df['route_id'] = st_df['trip_id'].map(trip_to_route)
    route_to_stops = st_df.groupby('route_id')['stop_id'].unique().apply(lambda x: [str(i) for i in x]).to_dict()
    
    if 'shape_id' in trips_df.columns:
        state['trip_to_shape'] = trips_df.dropna(subset=['shape_id']).set_index('trip_id')['shape_id'].astype(str).to_dict()
    else:
        state['trip_to_shape'] = {}
    
    trip_ids = st_df['trip_id'].values
    stop_ids = st_df['stop_id'].astype(str).values
    rows, cols = [], []
    for i in range(len(st_df) - 1):
        if trip_ids[i] == trip_ids[i+1]:
            u_id, v_id = stop_ids[i], stop_ids[i+1]
            if u_id in s_to_i and v_id in s_to_i:
                u_idx, v_idx = s_to_i[u_id], s_to_i[v_id]
                rows.append(u_idx)
                cols.append(v_idx)
                rows.append(v_idx)
                cols.append(u_idx)
                
    # Inject walk-transfer edges between WMATA and RideOn stops within 300 meters
    wmata_stops = []
    rideon_stops = []
    for s_id in all_stop_ids:
        info = stops_info.get(s_id)
        if not info: continue
        lat, lon = info['lat'], info['lon']
        if math.isnan(lat) or math.isnan(lon): continue
        if s_id.startswith('wmata_'):
            wmata_stops.append((s_id, lat, lon))
        elif s_id.startswith('rideon_'):
            rideon_stops.append((s_id, lat, lon))
            
    if wmata_stops and rideon_stops:
        wmata_coords = np.array([[lat, lon] for _, lat, lon in wmata_stops])
        wmata_tree = KDTree(wmata_coords)
        for r_id, r_lat, r_lon in rideon_stops:
            r_idx = s_to_i[r_id]
            indices = wmata_tree.query_ball_point([r_lat, r_lon], 0.0035)
            for idx in indices:
                w_id, w_lat, w_lon = wmata_stops[idx]
                w_idx = s_to_i[w_id]
                
                # Flat-earth distance in meters
                dlat = math.radians(w_lat - r_lat)
                dlon = math.radians(w_lon - r_lon)
                lat_mid = math.radians((w_lat + r_lat) / 2.0)
                dist_m = math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
                
                if dist_m <= 300.0:
                    if not crosses_river(w_lat, w_lon, r_lat, r_lon):
                        rows.append(w_idx)
                        cols.append(r_idx)
                        rows.append(r_idx)
                        cols.append(w_idx)
                    
    W_full = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_all, n_all))
    _, labels = connected_components(csgraph=W_full, directed=False)
    unique, counts = np.unique(labels, return_counts=True)
    giant_indices = np.where(labels == unique[np.argmax(counts)])[0]
    nodes_list = [all_stop_ids[i] for i in giant_indices]
    node_to_idx = {node: i for i, node in enumerate(nodes_list)}
    W = W_full[giant_indices, :][:, giant_indices]
    W.data[:] = 1.0
    
    coords_list = [[stops_info.get(nid, {'lat':0,'lon':0})['lat'], stops_info.get(nid, {'lat':0,'lon':0})['lon']] for nid in nodes_list]
    tree = KDTree(np.array(coords_list))
    state["graph_stats"]["centrality"] = np.array(W.sum(axis=1)).flatten().astype(float).tolist()
    state["graph_stats"]["active_nodes"] = len(nodes_list)
    state["graph_stats"]["active_edges"] = W.nnz
    
    # Export static stops list for UI autocomplete search
    static_dir = "static"
    if not os.path.exists(static_dir): os.makedirs(static_dir)
    stops_list = [{
        "id": nid,
        "name": stops_info.get(nid, {}).get('name', nid),
        "lat": float(stops_info.get(nid, {}).get('lat', 0.0)),
        "lon": float(stops_info.get(nid, {}).get('lon', 0.0))
    } for nid in nodes_list]
    stops_list.sort(key=lambda x: x['name'])
    with open(os.path.join(static_dir, "stops_list.json"), "w") as f_stops:
        json.dump(stops_list, f_stops)

    # Generate static/shapes.json if it does not exist
    shapes_json_path = os.path.join(static_dir, "shapes.json")
    if not os.path.exists(shapes_json_path):
        shapes_map = {}
        # Load WMATA shapes
        wmata_shapes_path = os.path.join(GTFS_DIR, "shapes.txt")
        if os.path.exists(wmata_shapes_path):
            try:
                df_sh = pd.read_csv(wmata_shapes_path, low_memory=False)
                df_sh = df_sh.sort_values(by=['shape_id', 'shape_pt_sequence'])
                for sh_id, group in df_sh.groupby('shape_id'):
                    coords = group[['shape_pt_lat', 'shape_pt_lon']].values.tolist()
                    downsampled = coords[::4]
                    if len(coords) > 0 and (len(coords) - 1) % 4 != 0:
                        downsampled.append(coords[-1])
                    shapes_map['wmata_' + str(sh_id)] = downsampled
            except Exception as e:
                print(f"Error parsing WMATA shapes.txt: {e}")
                
        # Load RideOn shapes
        rideon_shapes_path = os.path.join(GTFS_DIR, "rideon", "shapes.txt")
        if os.path.exists(rideon_shapes_path):
            try:
                df_sh = pd.read_csv(rideon_shapes_path, low_memory=False)
                df_sh = df_sh.sort_values(by=['shape_id', 'shape_pt_sequence'])
                for sh_id, group in df_sh.groupby('shape_id'):
                    coords = group[['shape_pt_lat', 'shape_pt_lon']].values.tolist()
                    downsampled = coords[::4]
                    if len(coords) > 0 and (len(coords) - 1) % 4 != 0:
                        downsampled.append(coords[-1])
                    shapes_map['rideon_' + str(sh_id)] = downsampled
            except Exception as e:
                print(f"Error parsing RideOn shapes.txt: {e}")
                
        with open(shapes_json_path, "w") as f_sh_json:
            json.dump(shapes_map, f_sh_json)
        
    return W, nodes_list, node_to_idx, stops_info, tree, route_to_stops

async def poll_weather_once(session):
    try:
        async with session.get(WEATHER_URL) as resp:
            data = await resp.json()
            w_code = data.get('current_weather', {}).get('weathercode', 0)
            if w_code > 50:
                state['weather_penalty'] = 0.85
                state['weather_desc'] = "Precipitation 🌨️"
            else:
                state['weather_penalty'] = 1.0
                state['weather_desc'] = "Clear ☀️"
            
            # Extract hourly precipitation rate matching the current time
            precip = 0.0
            hourly_data = data.get('hourly', {})
            if 'precipitation' in hourly_data:
                import datetime
                now_hour_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:00")
                if now_hour_str in hourly_data.get('time', []):
                    idx = hourly_data['time'].index(now_hour_str)
                    precip = float(hourly_data['precipitation'][idx])
                else:
                    precip = float(hourly_data['precipitation'][0]) if hourly_data['precipitation'] else 0.0
            state['precipitation_rate'] = precip
    except Exception as e:
        print(f"Error in poll_weather_once: {e}")

async def fetch_incident_layer(session, url, tree, region_key):
    if not url:
        state['incidents'][region_key] = []
        return
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.json()
                snapped = []
                for feat in data.get('features', []):
                    geom = feat.get('geometry')
                    if not geom: continue
                    coords = geom['coordinates']
                    if isinstance(coords[0], list): coords = coords[0]
                    dist, idx = tree.query([coords[1], coords[0]])
                    if dist < 0.008: snapped.append(int(idx))
                state['incidents'][region_key] = snapped
    except: pass

async def poll_incidents_once(session, tree):
    await asyncio.gather(
        fetch_incident_layer(session, INCIDENTS_DC, tree, "dc"),
        fetch_incident_layer(session, INCIDENTS_MD, tree, "md"),
        fetch_incident_layer(session, INCIDENTS_VA, tree, "va")
    )

async def poll_bikeshare_once(session, tree):
    try:
        async with session.get(BIKESHARE_INFO) as resp:
            info_data = await resp.json()
            stations = info_data['data']['stations']
            id_to_node = {s['station_id']: int(tree.query([s['lat'], s['lon']])[1]) for s in stations if tree.query([s['lat'], s['lon']])[0] < 0.0015}
            station_metadata = {s['station_id']: {"name": s['name'], "capacity": s['capacity']} for s in stations}
    except Exception as e:
        print(f"Error in poll_bikeshare_once (info): {e}")
        return

    try:
        async with session.get(BIKESHARE_STATUS) as resp:
            data = await resp.json()
            stats = data['data']['stations']
            total = 0
            depl_idx = []
            node_to_metadata = {}
            for s in stats:
                bikes = s['num_bikes_available'] + s['num_ebikes_available']
                total += bikes
                if bikes < 2 and s['station_id'] in id_to_node:
                    n_idx = id_to_node[s['station_id']]
                    depl_idx.append(n_idx)
                    meta = station_metadata.get(s['station_id'], {"name": "Capital Bikeshare", "capacity": 0})
                    node_to_metadata[n_idx] = {
                        "name": meta["name"],
                        "bikes": bikes,
                        "docks": s['num_docks_available'],
                        "capacity": meta["capacity"]
                    }
            state['bikeshare'].update({
                "total_bikes": total, 
                "active_stations": len(stats), 
                "depleted_stations": len(depl_idx), 
                "depleted_node_indices": depl_idx,
                "node_to_metadata": node_to_metadata
            })
    except Exception as e:
        print(f"Error in poll_bikeshare_once (status): {e}")

async def poll_gtfs_rt_once(session):
    try:
        async with session.get("https://api.wmata.com/gtfs/bus-gtfsrt-tripupdates.pb") as resp:
            if resp.status == 200:
                content = await resp.read()
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(content)
                delays = {}
                trip_delays = {}
                for entity in feed.entity:
                    if entity.HasField('trip_update'):
                        tu = entity.trip_update
                        t_id = 'wmata_' + str(tu.trip.trip_id)
                        for stu in tu.stop_time_update:
                            s_id = 'wmata_' + str(stu.stop_id)
                            d_val = stu.departure.delay if stu.HasField('departure') and stu.departure.HasField('delay') else (stu.arrival.delay if stu.HasField('arrival') and stu.arrival.HasField('delay') else 0)
                            delays[s_id] = d_val
                            if t_id not in trip_delays and d_val != 0:
                                trip_delays[t_id] = d_val
                
                # Merge delays safely
                for k in list(state['gtfs_delays'].keys()):
                    if k.startswith('wmata_'): del state['gtfs_delays'][k]
                for k in list(state['trip_delays'].keys()):
                    if k.startswith('wmata_'): del state['trip_delays'][k]
                state['gtfs_delays'].update(delays)
                state['trip_delays'].update(trip_delays)
                state['last_payload_ts'] = time.time()
    except Exception as e:
        print(f"Error in poll_gtfs_rt_once: {e}")

async def poll_rideon_vp_once(session, tree, nodes_list):
    api_key = os.environ.get("RIDEON_API_KEY", "")
    client_id = os.environ.get("RIDEON_CLIENT_ID", "")
    if not api_key or not client_id:
        return
    url_vp = f"http://rideon.app/json/GetGtfsRealtimeVehiclePositions?apiKey={api_key}&ClientId={client_id}"
    try:
        async with session.get(url_vp) as resp:
            if resp.status == 200:
                text = await resp.text()
                if "Access denied" not in text:
                    data = json.loads(text)
                    current_speeds_list = {}
                    current_buses = {}
                    rideon_bus_dict = {}
                    
                    for v in data:
                        trip_obj = v.get("Trip", {})
                        veh_obj = v.get("Vehicle", {})
                        pos_obj = v.get("Position", {})
                        
                        v_id = 'rideon_' + str(veh_obj.get("Id", ""))
                        r_id = 'rideon_' + str(trip_obj.get("RouteId", ""))
                        stop_id = 'rideon_' + str(v.get("StopId")) if v.get("StopId") else None
                        
                        lat = float(pos_obj.get("Latitude", 0.0)) if pos_obj.get("Latitude") else 0.0
                        lon = float(pos_obj.get("Longitude", 0.0)) if pos_obj.get("Longitude") else 0.0
                        bearing = float(pos_obj.get("Bearing", 0.0)) if pos_obj.get("Bearing") else 0.0
                        speed_mph = float(pos_obj.get("Speed", 0.0)) if pos_obj.get("Speed") else 0.0
                        speed_mps = speed_mph / 2.23694
                        
                        # Snap fallback
                        if not stop_id and lat != 0.0 and lon != 0.0:
                            dist, idx = tree.query([lat, lon])
                            if dist < 0.001:
                                snapped_id = nodes_list[idx]
                                if snapped_id.startswith('rideon_'):
                                    stop_id = snapped_id
                                    
                        if r_id not in current_buses:
                            current_buses[r_id] = []
                        current_buses[r_id].append(v_id)
                        
                        timestamp = int(v.get("Timestamp")) if v.get("Timestamp") else int(time.time())
                        t_id = 'rideon_' + str(trip_obj.get("TripId")) if trip_obj.get("TripId") else None
                        
                        delay_sec = 0
                        if t_id and t_id in state.get('trip_delays', {}):
                            delay_sec = int(state['trip_delays'][t_id])
                        elif stop_id and stop_id in state.get('gtfs_delays', {}):
                            delay_sec = int(state['gtfs_delays'][stop_id])
                            
                        if lat != 0.0 and lon != 0.0:
                            rideon_bus_dict[v_id] = {
                                "id": v_id,
                                "route": r_id,
                                "lat": lat,
                                "lon": lon,
                                "bearing": bearing,
                                "speed": speed_mps,
                                "timestamp": timestamp,
                                "trip_id": t_id,
                                "delay": delay_sec,
                                "shape_id": state.get('trip_to_shape', {}).get(t_id) if t_id else None
                            }
                            
                        if stop_id:
                            if stop_id not in current_speeds_list:
                                current_speeds_list[stop_id] = []
                            current_speeds_list[stop_id].append(speed_mph)
                            
                        # Canary tracking
                        if v_id in state['canary_buses']:
                            canary = state['canary_buses'][v_id]
                            if speed_mph < 2.0:
                                canary['status'] = "IMPACT CONFIRMED"
                                if r_id in state['active_predictions'] and state['active_predictions'][r_id]['t_physical'] is None:
                                    state['active_predictions'][r_id]['t_physical'] = time.time()
                            else:
                                canary['status'] = f"Tracking ({speed_mph:.1f}mph)"
                                
                    # Merge bus positions dictionary
                    for k in list(state['bus_positions_dict'].keys()):
                        if k.startswith('rideon_'): del state['bus_positions_dict'][k]
                    state['bus_positions_dict'].update(rideon_bus_dict)
                    
                    # Merge live speeds list
                    for k in list(state['live_speeds_list'].keys()):
                        if k.startswith('rideon_'): del state['live_speeds_list'][k]
                    state['live_speeds_list'].update(current_speeds_list)
                    
                    for k in list(state['live_speeds'].keys()):
                        if k.startswith('rideon_'): del state['live_speeds'][k]
                    for stop_id, speeds in current_speeds_list.items():
                        if speeds:
                            state['live_speeds'][stop_id] = sum(speeds) / len(speeds)
                            
                    # Merge live buses
                    for k in list(state['live_buses'].keys()):
                        if k.startswith('rideon_'): del state['live_buses'][k]
                    state['live_buses'].update(current_buses)
                    
                    state['last_payload_ts'] = time.time()
    except Exception as e:
        print(f"Error in poll_rideon_vp_once: {e}")

async def poll_rideon_tu_once(session):
    api_key = os.environ.get("RIDEON_API_KEY", "")
    client_id = os.environ.get("RIDEON_CLIENT_ID", "")
    if not api_key or not client_id:
        return
    url_tu = f"http://rideon.app/json/GetGtfsRealtimeTripUpdates?apiKey={api_key}&ClientId={client_id}"
    try:
        async with session.get(url_tu) as resp:
            if resp.status == 200:
                text = await resp.text()
                if "Access denied" not in text:
                    data = json.loads(text)
                    new_delays = {}
                    new_trip_delays = {}
                    for item in data:
                        trip_obj = item.get("Trip", {})
                        t_id = 'rideon_' + str(trip_obj.get("TripId", ""))
                        for stu in item.get("StopTimeUpdates", []):
                            s_id = 'rideon_' + str(stu.get("StopId", ""))
                            d_val = 0
                            new_delays[s_id] = d_val
                            if t_id not in new_trip_delays and d_val != 0:
                                new_trip_delays[t_id] = d_val
                                
                    # Merge delays
                    for k in list(state['gtfs_delays'].keys()):
                        if k.startswith('rideon_'): del state['gtfs_delays'][k]
                    for k in list(state['trip_delays'].keys()):
                        if k.startswith('rideon_'): del state['trip_delays'][k]
                    state['gtfs_delays'].update(new_delays)
                    state['trip_delays'].update(new_trip_delays)
    except Exception as e:
        print(f"Error in poll_rideon_tu_once: {e}")

async def poll_alerts_once(session):
    try:
        async with session.get("https://api.wmata.com/gtfs/bus-gtfsrt-alerts.pb") as resp:
            if resp.status == 200:
                content = await resp.read()
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(content)
                active_alerts = 0
                alerted_routes = set()
                for entity in feed.entity:
                    if entity.HasField('alert'):
                        active_alerts += 1
                        for informed in entity.alert.informed_entity:
                            if informed.HasField('route_id'):
                                alerted_routes.add('wmata_' + str(informed.route_id))
                state['wmata_official_alerts'] = alerted_routes
                state['graph_stats']['active_alerts'] = active_alerts
    except Exception as e:
        print(f"Error in poll_alerts_once: {e}")

async def poll_vehicle_positions_once(session):
    try:
        async with session.get("https://api.wmata.com/gtfs/bus-gtfsrt-vehiclepositions.pb") as resp:
            if resp.status == 200:
                content = await resp.read()
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(content)
                
                current_speeds_list = {}
                current_buses = {}
                wmata_bus_dict = {}
                for entity in feed.entity:
                    if entity.HasField('vehicle'):
                        v = entity.vehicle
                        v_id = 'wmata_' + str(v.vehicle.id)
                        r_id = 'wmata_' + str(v.trip.route_id)
                        stop_id = 'wmata_' + str(v.stop_id) if v.stop_id else None
                        
                        if r_id not in current_buses: current_buses[r_id] = []
                        current_buses[r_id].append(v_id)

                        speed_mph = (v.position.speed * 2.23694) if v.position.speed else 0
                        
                        lat = float(v.position.latitude) if v.position.latitude else 0.0
                        lon = float(v.position.longitude) if v.position.longitude else 0.0
                        bearing = float(v.position.bearing) if v.position.bearing else 0.0
                        speed_mps = float(v.position.speed) if v.position.speed else 0.0
                        timestamp = int(v.timestamp) if v.timestamp else int(time.time())
                        
                        t_id = 'wmata_' + str(v.trip.trip_id) if v.trip.trip_id else None
                        delay_sec = 0
                        if t_id and 'trip_delays' in state and t_id in state['trip_delays']:
                            delay_sec = int(state['trip_delays'][t_id])
                        elif stop_id and 'gtfs_delays' in state and stop_id in state['gtfs_delays']:
                            delay_sec = int(state['gtfs_delays'][stop_id])

                        if lat != 0.0 and lon != 0.0:
                            wmata_bus_dict[v_id] = {
                                "id": v_id,
                                "route": r_id,
                                "lat": lat,
                                "lon": lon,
                                "bearing": bearing,
                                "speed": speed_mps,
                                "timestamp": timestamp,
                                "trip_id": t_id,
                                "delay": delay_sec,
                                "shape_id": state.get('trip_to_shape', {}).get(t_id) if t_id else None
                            }

                        if stop_id:
                            if stop_id not in current_speeds_list:
                                current_speeds_list[stop_id] = []
                            current_speeds_list[stop_id].append(speed_mph)
                        
                        # Canary Stats
                        if v_id in state['canary_buses']:
                            canary = state['canary_buses'][v_id]
                            if speed_mph < 2.0:
                                canary['status'] = "IMPACT CONFIRMED"
                                if r_id in state['active_predictions'] and state['active_predictions'][r_id]['t_physical'] is None:
                                    state['active_predictions'][r_id]['t_physical'] = time.time()
                            else:
                                canary['status'] = f"Tracking ({speed_mph:.1f}mph)"
                
                # Merge wmata_bus_dict
                for k in list(state['bus_positions_dict'].keys()):
                    if k.startswith('wmata_'): del state['bus_positions_dict'][k]
                state['bus_positions_dict'].update(wmata_bus_dict)
                
                # Merge live speeds list
                for k in list(state['live_speeds_list'].keys()):
                    if k.startswith('wmata_'): del state['live_speeds_list'][k]
                state['live_speeds_list'].update(current_speeds_list)
                
                for k in list(state['live_speeds'].keys()):
                    if k.startswith('wmata_'): del state['live_speeds'][k]
                for stop_id, speeds in current_speeds_list.items():
                    if speeds:
                        state['live_speeds'][stop_id] = sum(speeds) / len(speeds)
                        
                # Merge live buses
                for k in list(state['live_buses'].keys()):
                    if k.startswith('wmata_'): del state['live_buses'][k]
                state['live_buses'].update(current_buses)
                
                state['last_payload_ts'] = time.time()
    except Exception as e:
        print(f"Error in poll_vehicle_positions_once: {e}")

async def poll_metrorail_rt_once(session):
    try:
        async with session.get("https://api.wmata.com/gtfs/rail-gtfsrt-alerts.pb") as resp:
            if resp.status == 200:
                content = await resp.read()
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(content)
                alerts = 0
                for entity in feed.entity:
                    if entity.HasField('alert'): alerts += 1
                state['rail_alerts'] = alerts
    except Exception as e:
        print(f"Error in poll_metrorail_rt_once: {e}")

def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8 # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlon/2.0)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

metro_stations = {}
try:
    if os.path.exists("static/metro_stations.json"):
        with open("static/metro_stations.json", "r") as f:
            metro_stations = json.load(f)
except Exception as e:
    print(f"Error loading metro_stations.json: {e}")

async def poll_rail_positions_once(session, tree, stops_info, nodes_list):
    try:
        async with session.get("https://api.wmata.com/gtfs/rail-gtfsrt-vehiclepositions.pb") as resp:
            if resp.status == 200:
                content = await resp.read()
                feed = gtfs_realtime_pb2.FeedMessage()
                feed.ParseFromString(content)
                surges = set()
                train_positions = []
                
                for entity in feed.entity:
                    if entity.HasField('vehicle'):
                        v = entity.vehicle
                        v_id = str(v.vehicle.id)
                        r_id = str(v.trip.route_id)
                        stop_id = str(v.stop_id) if v.stop_id else None
                        
                        lat = float(v.position.latitude) if v.position.latitude else 0.0
                        lon = float(v.position.longitude) if v.position.longitude else 0.0
                        bearing = float(v.position.bearing) if v.position.bearing else 0.0
                        timestamp = int(v.timestamp) if v.timestamp else int(time.time())
                        current_status = getattr(v, 'current_status', 2) # Default to IN_TRANSIT_TO
                        
                        # Map stop_id to station name
                        station_name = "Unknown Station"
                        if stop_id:
                            parts = stop_id.split('_')
                            if len(parts) >= 2:
                                code = parts[1]
                                station_name = metro_stations.get(code, "Unknown Station")
                        
                        # Fallback to KDTree snapping to get a nearby surface intersection if station is unknown
                        if station_name == "Unknown Station" and lat != 0.0 and lon != 0.0:
                            dist, idx = tree.query([lat, lon])
                            if dist < 0.005:
                                node_id = nodes_list[idx]
                                station_name = stops_info.get(node_id, {}).get('name', 'Unknown Station')
                        
                        # Speed estimation via coordinate caching & Haversine distance
                        speed_mph = 0.0
                        if lat != 0.0 and lon != 0.0:
                            if current_status == 1: # STOPPED_AT
                                speed_mph = 0.0
                            else:
                                prev = state['prev_trains'].get(v_id)
                                if prev:
                                    prev_lat, prev_lon, prev_ts, prev_speed = prev
                                    dt = timestamp - prev_ts
                                    if dt > 0:
                                        dist_miles = haversine(prev_lat, prev_lon, lat, lon)
                                        speed_mph = (dist_miles / dt) * 3600.0
                                        if speed_mph > 75.0: # Clamp excessive speed jumps due to GPS telemetry jumps
                                            speed_mph = prev_speed if prev_speed is not None else 0.0
                                    else:
                                        speed_mph = prev_speed if prev_speed is not None else 0.0
                                else:
                                    speed_mph = (v.position.speed * 2.23694) if (hasattr(v.position, 'speed') and v.position.speed) else 0.0
                                
                                state['prev_trains'][v_id] = (lat, lon, timestamp, speed_mph)
                                
                                # Convert speed back to m/s for GTFS-RT compliant output
                                speed_mps = speed_mph / 2.23694
                                
                                train_positions.append({
                                    "id": v_id,
                                    "route": r_id,
                                    "lat": lat,
                                    "lon": lon,
                                    "bearing": bearing,
                                    "speed": speed_mps,
                                    "speed_mph": speed_mph,
                                    "timestamp": timestamp,
                                    "trip_id": v.trip.trip_id,
                                    "station": station_name,
                                    "status": "STOPPED_AT" if current_status == 1 else "IN_TRANSIT_TO" if current_status == 2 else "INCOMING_AT" if current_status == 0 else "UNKNOWN"
                                })
                                
                                # Surge Candidate Logic (maintain [longitude, latitude] tree query order for active ground surges index compatibility)
                                if current_status == 1 and state['rail_alerts'] > 0:
                                    dist, idx = tree.query([v.position.latitude, v.position.longitude])
                                    if dist < 0.005:
                                        surges.add(int(idx))
                                    
                state['rail_surges'] = surges
                state['train_positions'] = train_positions
    except Exception as e:
        print(f"Error in poll_rail_positions_once: {e}")

class PollingTask:
    def __init__(self, task_id, interval, priority):
        self.task_id = task_id
        self.interval = interval
        self.priority = priority # 1 (High), 2 (Mid), 3 (Low)
        self.next_run = 0.0

    def __lt__(self, other):
        if self.next_run == other.next_run:
            return self.priority < other.priority
        return self.next_run < other.next_run

import heapq

async def priority_queue_scheduler(tasks, state, rate_limiter, tree, nodes_list, stops_info, route_to_stops):
    queue = []
    now = time.time()
    for task in tasks:
        task.next_run = now
        heapq.heappush(queue, task)
        
    async with aiohttp.ClientSession() as session:
        while True:
            if not queue:
                await asyncio.sleep(1.0)
                continue
                
            task = heapq.heappop(queue)
            now = time.time()
            
            if task.next_run > now:
                heapq.heappush(queue, task)
                await asyncio.sleep(min(1.0, task.next_run - now))
                continue
                
            url_map = {
                "weather": WEATHER_URL,
                "incidents": INCIDENTS_DC,
                "bikeshare": BIKESHARE_STATUS,
                "wmata_tu": "https://api.wmata.com/gtfs/bus-gtfsrt-tripupdates.pb",
                "rideon_vp": f"http://rideon.app/json/GetGtfsRealtimeVehiclePositions?apiKey={os.environ.get('RIDEON_API_KEY', '')}&ClientId={os.environ.get('RIDEON_CLIENT_ID', '')}",
                "rideon_tu": f"http://rideon.app/json/GetGtfsRealtimeTripUpdates?apiKey={os.environ.get('RIDEON_API_KEY', '')}&ClientId={os.environ.get('RIDEON_CLIENT_ID', '')}",
                "wmata_alerts": "https://api.wmata.com/gtfs/bus-gtfsrt-alerts.pb",
                "wmata_vp": "https://api.wmata.com/gtfs/bus-gtfsrt-vehiclepositions.pb",
                "metrorail_rt": "https://api.wmata.com/gtfs/rail-gtfsrt-alerts.pb",
                "rail_positions": "https://api.wmata.com/gtfs/rail-gtfsrt-vehiclepositions.pb"
            }
            
            url = url_map.get(task.task_id)
            if url and not rate_limiter.can_request(url, now):
                task.next_run = now + 1.0
                heapq.heappush(queue, task)
                await asyncio.sleep(0.1)
                continue
                
            if url:
                rate_limiter.record_request(url, now)
                
            try:
                if task.task_id == 'weather':
                    await poll_weather_once(session)
                elif task.task_id == 'incidents':
                    await poll_incidents_once(session, tree)
                elif task.task_id == 'bikeshare':
                    await poll_bikeshare_once(session, tree)
                elif task.task_id == 'wmata_tu':
                    await poll_gtfs_rt_once(session)
                elif task.task_id == 'rideon_vp':
                    await poll_rideon_vp_once(session, tree, nodes_list)
                elif task.task_id == 'rideon_tu':
                    await poll_rideon_tu_once(session)
                elif task.task_id == 'wmata_alerts':
                    await poll_alerts_once(session)
                elif task.task_id == 'wmata_vp':
                    await poll_vehicle_positions_once(session)
                elif task.task_id == 'metrorail_rt':
                    await poll_metrorail_rt_once(session)
                elif task.task_id == 'rail_positions':
                    await poll_rail_positions_once(session, tree, stops_info, nodes_list)
            except Exception as task_err:
                print(f"Error executing task {task.task_id}: {task_err}")
                
            task.next_run = now + task.interval
            heapq.heappush(queue, task)
            await asyncio.sleep(0.01)



def spectral_analysis(W, f=None, v0=None):
    n = W.shape[0]
    D_diag = np.array(W.sum(axis=1)).flatten()
    L = sp.diags(D_diag) - W
    
    # Calculate adaptive regularization shift sigma_t
    if f is not None and len(f) > 0:
        sigma_t = float(max(1e-5, 1e-3 * np.std(f)))
    else:
        sigma_t = 1e-5
        
    # Validate v0
    v0_param = None
    if v0 is not None and len(v0) == n:
        v0_param = v0
        
    try:
        evals, evecs = eigsh(L, k=3, which='LM', sigma=sigma_t, tol=1e-2, maxiter=500, v0=v0_param)
        idx = np.argsort(evals)
        return evals[idx[1]], evecs[:, idx[1]], evals[idx[2]] - evals[idx[1]]
    except:
        try:
            evals, evecs = eigsh(L, k=3, which='SM', tol=1e-1, v0=v0_param)
            idx = np.argsort(evals)
            return evals[idx[1]], evecs[:, idx[1]], evals[idx[2]] - evals[idx[1]]
        except:
            return 0.0, np.zeros(n), 0.0

async def main_loop(W, nodes_list, node_to_idx, stops_info, tree, route_to_stops):
    W_mask = W.copy(); W_mask.data[:] = 1.0
    
    # Pre-compute Surface Surge hubs (Top 2% most connected bus stops)
    degree = np.array(W_mask.sum(axis=1)).flatten()
    threshold = np.percentile(degree, 98)
    surge_hubs_idx = np.where(degree >= threshold)[0]
    
    # Pre-compute Degree Inverse for Spatial Friction Diffusion (Graph Heat Kernel)
    D_diag = np.array(W_mask.sum(axis=1)).flatten()
    D_inv = np.ones_like(D_diag, dtype=np.float32)
    non_zeros = D_diag > 0
    D_inv[non_zeros] = 1.0 / D_diag[non_zeros]
    
    # Pre-compute normalized centrality vector C (degree centrality normalized by max degree)
    max_deg = np.max(D_diag) if np.max(D_diag) > 0 else 1.0
    C = D_diag / max_deg
    
    # Pre-compute CSR row index array to support instantaneous vectorised edge row lookups
    rows = np.repeat(np.arange(W_mask.shape[0]), np.diff(W_mask.indptr))
    
    # Pre-compute static centrality-adjusted edge thresholds vector to eliminate runtime calculation overhead
    thresholds = 0.4 * (1.0 + GAMMA_CENTRALITY * (1.0 - 0.5 * (C[rows] + C[W_mask.indices])))
    
    last_v2 = None
    last_l2 = 0.05
    last_gap = 0.05
    last_f = None
    
    while True:
        loop_start = time.time(); state["total_cycles"] += 1
        f = np.ones(len(nodes_list), dtype=np.float32) * state['weather_penalty']
        
        # Mathematical Tau Wiggle: Break degenerate eigenvalue symmetry ties.
        # This microscopic deterministic noise ensures the Fiedler bisection is mathematically 
        # stable and unique, preventing the Z-axis topology from snapping between orthogonal bases.
        f += 1e-5 * np.sin(math.tau * np.arange(len(nodes_list)) / len(nodes_list))
        
        total_delay = 0
        
        # Apply Metrorail Surface Surge
        if state['rail_alerts'] > 0:
            for idx in surge_hubs_idx: f[idx] *= 0.7
            
        # Precision Ground Zero Surges
        for idx in state['rail_surges']:
            f[idx] *= 0.6
            
        # We don't use gtfs_delays anymore because WMATA doesn't populate it.
        # Instead, we use physical live speeds from VehiclePositions.
        for s_id, speeds in state.get('live_speeds_list', {}).items():
            if s_id in node_to_idx:
                idx = node_to_idx[s_id]
                n_severe = sum(1 for speed in speeds if speed < 5.0)
                m_heavy = sum(1 for speed in speeds if 5.0 <= speed < 15.0)
                if n_severe > 0 or m_heavy > 0:
                    penalty = (0.3 ** min(n_severe, 3)) * (0.6 ** min(m_heavy, 3))
                    f[idx] *= penalty
                    f[idx] = max(f[idx], 0.01)
        
        # HARDCORE REFEREE MODE: WMATA alerts DO NOT apply friction to the map anymore.
        # The Engine must catch gridlock purely through physical physics (speed < 5mph).
        
        if os.path.exists("sim_state.json"):
            try:
                with open("sim_state.json", "r") as f_sim: state['injections'] = json.load(f_sim).get("injected_nodes", [])
            except: pass
        for region in state['incidents'].values():
            for idx in region: f[idx] = 0.05
        for s_id in state['injections']:
            if s_id in node_to_idx: f[node_to_idx[s_id]] = 0.01
        
        # Apply Spatial Friction Diffusion (Graph Heat Kernel)
        # Penalties bleed onto topological neighbors, simulating physical tailbacks upstream of blockages.
        D_inv_W_f = D_inv * (W_mask @ f)
        f = (1.0 - ALPHA_DIFFUSION) * f + ALPHA_DIFFUSION * D_inv_W_f
        f = np.clip(smooth_floor(f), 0.01, 1.0)
        
        W = sp.diags(f) @ W_mask @ sp.diags(f)
        
        if last_f is not None:
            f_diff_l2 = float(np.linalg.norm(f - last_f))
        else:
            f_diff_l2 = 1.0
            
        if last_f is not None and f_diff_l2 < 1e-3:
            l2 = last_l2
            v2 = last_v2
            gap = last_gap
        else:
            try:
                l2, v2, gap = spectral_analysis(W, f, last_v2)
                if l2 == 0 and np.all(v2 == 0):
                    l2 = last_l2 if last_l2 is not None else 0.05
                    v2 = last_v2 if last_v2 is not None else np.zeros(len(nodes_list))
                    gap = last_gap if last_gap is not None else 0.05
                    print("⚠️ [Solver Warning] Spectral convergence failed. Using topological fallback values.")
            except Exception as solver_err:
                l2 = last_l2 if last_l2 is not None else 0.05
                v2 = last_v2 if last_v2 is not None else np.zeros(len(nodes_list))
                gap = last_gap if last_gap is not None else 0.05
                print(f"⚠️ [Solver Exception] Mathematical error: {solver_err}. Using topological fallback values.")

        last_f = f.copy()
        last_l2 = l2
        last_v2 = v2
        last_gap = gap

        lat = (time.time() - loop_start) * 1000
        # Safe v2 formatting for graph_stats
        v2_for_stats = v2 if v2 is not None else np.zeros(len(nodes_list))
        worst_idx = np.argmin(f)
        state["graph_stats"].update({
            "compute_latency": lat, 
            "peak_latency": max(state["graph_stats"]["peak_latency"], lat), 
            "avg_friction": np.mean(f), 
            "system_tension": np.std(f), 
            "spectral_gap": gap, 
            "total_delay_sec": total_delay, 
            "fiedler_max": v2_for_stats[np.argmax(np.abs(v2_for_stats))] if len(v2_for_stats) > 0 else 0.0, 
            "fiedler_pole_name": stops_info[nodes_list[np.argmax(np.abs(v2_for_stats))]]['name'] if len(v2_for_stats) > 0 else "Core", 
            "worst_node_name": stops_info[nodes_list[worst_idx]]['name'] if worst_idx < len(nodes_list) else "None"
        })
        
        # --- FRACTURE DIAGNOSTICS & TARGET ACQUISITION ---
        fractures = []
        stressed_idx = np.where(W.data < thresholds)[0]
        
        if len(stressed_idx) > 0:
            sorted_stress = stressed_idx[np.argsort(W.data[stressed_idx])]
            seen_edges = set()
            
            for idx in sorted_stress:
                weight = W.data[idx]
                r = rows[idx]
                c = W.indices[idx]
                
                edge_key = tuple(sorted([r, c]))
                if edge_key in seen_edges: continue
                seen_edges.add(edge_key)
                
                u_id, v_id = nodes_list[r], nodes_list[c]
                u_name, v_name = stops_info[u_id]['name'], stops_info[v_id]['name']
                
                cause = "Cascading Bus Delays"
                delay_sec = max(state['gtfs_delays'].get(u_id, 0), state['gtfs_delays'].get(v_id, 0))
                
                if u_id in state['injections'] or v_id in state['injections']: cause = "🛑 SIMULATED INJECTION"
                elif r in state['incidents']['dc'] or c in state['incidents']['dc'] or r in state['incidents']['md'] or c in state['incidents']['md'] or r in state['incidents']['va'] or c in state['incidents']['va']: cause = "🚓 MUNICIPAL INCIDENT (Crash/Closure)"
                else:
                    is_alerted = False
                    for r_id in state['wmata_official_alerts']:
                        if r_id in route_to_stops and (u_id in route_to_stops[r_id] or v_id in route_to_stops[r_id]):
                            is_alerted = True; break
                    if is_alerted: cause = "⚠️ WMATA SERVICE ALERT"
                    elif state['live_speeds'].get(u_id, 20) < 5 or state['live_speeds'].get(v_id, 20) < 5:
                        cause = "🚗 LOW VELOCITY SENSOR (< 5mph)"
                    elif delay_sec > 0: cause = f"🚌 SEVERE DELAY ({delay_sec}s reported)"
                
                fractures.append({"u": u_name, "v": v_name, "w": weight, "cause": cause, "u_id": u_id})
                
                # --- CANARY LOCK-ON ---
                # Reverse-map stop_id to route_ids (naive approach for demonstration)
                for r_id, stops in route_to_stops.items():
                    if u_id in stops and r_id not in state['active_predictions'] and len(state['canary_buses']) < 5:
                        # Grab a REAL bus currently on this route
                        available_buses = state.get('live_buses', {}).get(r_id, [])
                        if available_buses:
                            canary_vid = available_buses[0] # Pick the first real bus
                            state['canary_buses'][canary_vid] = {"route_id": r_id, "target": u_name, "status": "Acquiring Target..."}
                            
                            # If WMATA already has an alert out, they beat us to it.
                            initial_t_wmata = time.time() - 1 if r_id in state['wmata_official_alerts'] else None
                            state['active_predictions'][r_id] = {"t_engine": time.time(), "t_physical": None, "t_wmata": initial_t_wmata, "cause": cause, "resolved": False, "pre_existing": r_id in state['wmata_official_alerts']}
                
                if len(fractures) >= 15: break

        # --- ADJUDICATION ENGINE & HEALING PROTOCOL ---
        current_time = time.time()
        for r_id, pred in list(state['active_predictions'].items()):
            if not pred['resolved']:
                # Check if WMATA finally issued an alert
                if r_id in state['wmata_official_alerts'] and pred['t_wmata'] is None:
                    pred['t_wmata'] = current_time
                
                # Condition 1: Both physical confirmation and WMATA alert occurred
                if pred['t_physical'] is not None and pred['t_wmata'] is not None:
                    lead_time = pred['t_wmata'] - pred['t_engine']
                    
                    if pred.get('pre_existing', False):
                        winner = "WMATA"
                        lead_time = 0 # Don't skew the average with fake negative time
                    else:
                        if lead_time > 60: winner = "X-RAY"
                        elif lead_time > -60: winner = "TIE"
                        else: winner = "WMATA"
                    
                    if winner == "X-RAY": state['scoreboard_stats']['xray_wins'] += 1
                    elif winner == "WMATA": state['scoreboard_stats']['wmata_wins'] += 1
                    
                    # Only calculate average for actual races, not pre-existing conditions
                    if not pred.get('pre_existing', False):
                        state['scoreboard_stats']['total_races'] += 1
                        state['scoreboard_stats']['avg_lead_time_sec'] = ((state['scoreboard_stats']['avg_lead_time_sec'] * (state['scoreboard_stats']['total_races'] - 1)) + lead_time) / state['scoreboard_stats']['total_races']
                    
                    try:
                        with open(SCOREBOARD_FILE, 'a', newline='') as f_score:
                            csv.writer(f_score).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), r_id, pred['cause'], pred['t_engine'], pred['t_physical'], pred['t_wmata'], lead_time, winner])
                    except: pass
                    
                    pred['resolved'] = True
                    
                # Condition 2: Healing Protocol / Timeout
                # If 15 minutes (900s) have passed and WMATA never alerted, it was an Unreported Micro-Jam
                elif (current_time - pred['t_engine']) > 900 and pred['t_wmata'] is None:
                    # If we got physical confirmation, it was a real jam they missed. If not, it was just math noise.
                    final_cause = "UNREPORTED MICRO-JAM" if pred['t_physical'] else "GHOST FRACTURE (HEALED)"
                    
                    # We still award X-Ray the win if it was a real, physical micro-jam.
                    if pred['t_physical']:
                        state['scoreboard_stats']['xray_wins'] += 1
                        # We give a flat +15m lead time for catching something they entirely missed
                        lead_time = 900 
                        state['scoreboard_stats']['total_races'] += 1
                        state['scoreboard_stats']['avg_lead_time_sec'] = ((state['scoreboard_stats']['avg_lead_time_sec'] * (state['scoreboard_stats']['total_races'] - 1)) + lead_time) / state['scoreboard_stats']['total_races']
                    else:
                        lead_time = 0
                        
                    try:
                        with open(SCOREBOARD_FILE, 'a', newline='') as f_score:
                            csv.writer(f_score).writerow([time.strftime('%Y-%m-%d %H:%M:%S'), r_id, final_cause, pred['t_engine'], pred['t_physical'], "NEVER", lead_time, "X-RAY (UNREPORTED)"])
                    except: pass
                    
                    pred['resolved'] = True
                
                # Cleanup Canaries if resolved
                if pred['resolved']:
                    keys_to_delete = [vid for vid, info in state['canary_buses'].items() if info['route_id'] == r_id]
                    for k in keys_to_delete: del state['canary_buses'][k]
                    del state['active_predictions'][r_id]

        os.system('clear'); uptime = time.time() - state["engine_start"]; freshness = time.time() - state["last_payload_ts"] if state["last_payload_ts"] > 0 else 0
        print(f"🛰️  DMV GRIDLOCK X-RAY | {time.strftime('%X')} | CYCLE: {state['total_cycles']}\n{'='*80}\n🌍 TOPOLOGY: {state['graph_stats']['active_nodes']} Nodes | {state['graph_stats']['active_edges']} Edges\n🌤️  WEATHER:  {state['weather_desc']} ({state['weather_penalty']:.2f})\n📍 INCIDENTS: DC:{len(state['incidents']['dc'])} | MD:{len(state['incidents']['md'])} | VA:{len(state['incidents']['va'])}\n🚲 BIKESHARE: {state['bikeshare']['total_bikes']} Bikes | {state['bikeshare']['depleted_stations']} Empty Stations\n📢 ALERTS:    {state['graph_stats']['active_alerts']} Bus | {state['rail_alerts']} Metrorail\n{'-'*80}\n📊 SYSTEM PULSE:\n   Connectivity (λ2): {l2:.8f} [{'🟢 Stable' if l2 > 0.0001 else '🔴 Critical'}]\n   Spectral Gap:     {gap:.8f}\n   Avg Node Flow:    {state['graph_stats']['avg_friction']:.4f}\n   System Tension:   {state['graph_stats']['system_tension']:.4f}\n{'-'*80}")
        
        print("🏆 PREDICTION SCOREBOARD:")
        avg_lead_mins = state['scoreboard_stats']['avg_lead_time_sec'] / 60
        print(f"   X-Ray Wins: {state['scoreboard_stats']['xray_wins']} | WMATA Wins: {state['scoreboard_stats']['wmata_wins']} | Avg Lead Time: +{avg_lead_mins:.1f}m")
        
        print("\n🚌 CANARY TRACKER (Live GPS Target Lock):")
        if not state['canary_buses']: print("   [Standby] No active predictions. Scanning...")
        for vid, info in state['canary_buses'].items():
            print(f"   [LOCKED] Route {info['route_id']} approaching {info['target'][:15]}... | {info['status']}")

        print(f"\n🚨 ACTIVE FRACTURES (Top 15):")
        if not fractures: print("   ✅ Grid is flowing nominally.")
        else:
            for i, frac in enumerate(fractures, 1):
                print(f"   {i}. {frac['u']} ➔ {frac['v']}\n      └─ Friction: {frac['w']:.4f} | Cause: {frac['cause']}")
                
        print(f"{'-'*80}\n⚡ COMPUTE: {lat:.2f}ms | Uptime: {int(uptime//3600)}h {int((uptime%3600)//60)}m | Data: {int(freshness)}s\n{'='*80}")


        log_telemetry(time.strftime('%Y-%m-%d %H:%M:%S'), l2, gap, state["graph_stats"]["avg_friction"], len(state['incidents']['dc']), len(state['incidents']['md']), len(state['incidents']['va']), state['bikeshare']['depleted_stations'], lat)
        # --- STATE EXPORT FOR 3D VIZ (Atomic Write) ---
        try:
            target_path = "network_state.npz"; tmp_path = "network_state_tmp"; coords = np.array([[stops_info[nid]['lat'], stops_info[nid]['lon']] for nid in nodes_list])
            
            # Dynamic Spectral Bisection partitioning based on Fiedler median split
            partitions = np.where(v2 >= np.median(v2), 0, 1)
            
            np.savez_compressed(tmp_path, nodes=np.array(nodes_list), names=np.array([stops_info[nid]['name'] for nid in nodes_list]), v_2=v2, coords=coords, weights=W.data, indices=W.indices, indptr=W.indptr, node_friction=f, lambda_2=np.array([l2]), spectral_gap=np.array([gap]), weather_penalty=np.array([state['weather_penalty']]), weather_desc=np.array([state['weather_desc']]), incidents_dc=np.array([len(state['incidents']['dc'])]), incidents_md=np.array([len(state['incidents']['md'])]), incidents_va=np.array([len(state['incidents']['va'])]), bikeshare_depleted=np.array([state['bikeshare']['depleted_stations']]), depleted_indices=np.array(state['bikeshare']['depleted_node_indices']), centrality=np.array(state["graph_stats"]["centrality"]), active_alerts=np.array([state['graph_stats'].get('active_alerts', 0)]), partitions=partitions)
            if os.path.exists(tmp_path + ".npz"): os.replace(tmp_path + ".npz", target_path)
            
            # Export JSON for the isolated WebGL iframe
            z_vals = v2 * 150
            gx = coords[:, 1].tolist(); gy = coords[:, 0].tolist(); gz = [0] * len(nodes_list)

            # Extract boundary crossing edges (choke-points) crossing partitions
            diff_partition = partitions[rows] != partitions[W_mask.indices]
            boundary_mask = diff_partition & (rows < W_mask.indices)
            bx_bound, by_bound, bz_bound = [], [], []
            for b_idx in np.where(boundary_mask)[0]:
                r_n = rows[b_idx]
                c_n = W_mask.indices[b_idx]
                bx_bound.extend([float(coords[r_n, 1]), float(coords[c_n, 1]), None])
                by_bound.extend([float(coords[r_n, 0]), float(coords[c_n, 0]), None])
                bz_bound.extend([float(z_vals[r_n]), float(z_vals[c_n]), None])

            # Neural Web (Split into two traces with Bezier Arcs for Express Routes)
            wx1, wy1, wz1 = [], [], []
            wx2, wy2, wz2 = [], [], []
            midpoint = len(W.indptr) // 2
            
            def get_arc(x0, y0, z0, x2, y2, z2):
                dist = math.hypot(x2 - x0, y2 - y0)
                if dist < 0.04: # Less than ~4km, keep it a straight line (2 points)
                    return [x0, x2], [y0, y2], [z0, z2]
                # Long express route -> 5-point Bezier Arc
                x1 = (x0 + x2) / 2; y1 = (y0 + y2) / 2; z1 = max(z0, z2) + (dist * 250)
                xs, ys, zs = [], [], []
                for t in [0, 0.25, 0.5, 0.75, 1.0]:
                    xs.append((1-t)**2 * x0 + 2*(1-t)*t * x1 + t**2 * x2)
                    ys.append((1-t)**2 * y0 + 2*(1-t)*t * y1 + t**2 * y2)
                    zs.append((1-t)**2 * z0 + 2*(1-t)*t * z1 + t**2 * z2)
                return xs, ys, zs

            for i in range(0, len(W.indptr)-1, 1):
                for j in range(W.indptr[i], W.indptr[i+1]):
                    target = W.indices[j]
                    if i < target: # Deduplicate symmetric edges
                        ax, ay, az = get_arc(float(coords[i, 1]), float(coords[i, 0]), float(z_vals[i]), float(coords[target, 1]), float(coords[target, 0]), float(z_vals[target]))
                        if i < midpoint:
                            wx1.extend(ax + [None]); wy1.extend(ay + [None]); wz1.extend(az + [None])
                        else:
                            wx2.extend(ax + [None]); wy2.extend(ay + [None]); wz2.extend(az + [None])

            # Fractures
            stressed_idx = np.where(W.data < thresholds)[0]
            fx, fy, fz = [], [], []
            for idx in stressed_idx:
                r = rows[idx]; c = W.indices[idx]
                if r < c: # Deduplicate symmetric fractures
                    fx.extend([float(coords[r, 1]), float(coords[c, 1]), None])
                    fy.extend([float(coords[r, 0]), float(coords[c, 0]), None])
                    fz.extend([float(z_vals[r]), float(z_vals[c]), None])

            # Hubs
            centrality_np = np.array(state["graph_stats"]["centrality"])
            clear_idx = np.where(f >= 0.85)[0]; jam_idx = np.where(f < 0.85)[0]
            
            # Hover-Over Intelligence Generator
            def build_hover_text(idx):
                u_id = nodes_list[idx]
                name = stops_info[u_id]['name']
                f_score = f[idx]
                speed = state['live_speeds'].get(u_id, None)
                
                is_alerted = False
                for r_id in state['wmata_official_alerts']:
                    if r_id in route_to_stops and u_id in route_to_stops[r_id]:
                        is_alerted = True; break
                        
                speed_str = f"Live Speed: {speed:.1f} mph" if speed is not None else "No Live Buses"
                alert_str = "⚠️ WMATA Alert Active" if is_alerted else "✅ Normal Service"
                if u_id in state['injections']: alert_str = "🛑 SIMULATED INJECTION"
                if u_id in state['bikeshare']['depleted_node_indices']: alert_str += "<br>🚲 Bikes Depleted"
                if state['rail_alerts'] > 0 and idx in surge_hubs_idx: alert_str += "<br>🚇 METRO HUB SURGE (Network Delay)"
                if idx in state['rail_surges']: alert_str += "<br>🎯 PRECISION GROUND ZERO SURGE"
                
                return f"<b>{name}</b><br>Friction: {f_score:.2f}<br>{speed_str}<br>{alert_str}"

            clear_x = coords[clear_idx, 1].tolist() if len(clear_idx) > 0 else []
            clear_y = coords[clear_idx, 0].tolist() if len(clear_idx) > 0 else []
            clear_z = z_vals[clear_idx].tolist() if len(clear_idx) > 0 else []
            clear_txt = [build_hover_text(i) for i in clear_idx]
            
            jam_x = coords[jam_idx, 1].tolist() if len(jam_idx) > 0 else []
            jam_y = coords[jam_idx, 0].tolist() if len(jam_idx) > 0 else []
            jam_z = z_vals[jam_idx].tolist() if len(jam_idx) > 0 else []
            jam_txt = [build_hover_text(i) for i in jam_idx]
            jam_size = np.clip(centrality_np[jam_idx] * 1.5, 6, 25).tolist() if len(jam_idx) > 0 else []
            jam_col = f[jam_idx].tolist() if len(jam_idx) > 0 else []

            # Bikeshare
            depleted = state['bikeshare']['depleted_node_indices']
            bx = coords[depleted, 1].tolist() if len(depleted) > 0 else []
            by = coords[depleted, 0].tolist() if len(depleted) > 0 else []
            bz = z_vals[depleted].tolist() if len(depleted) > 0 else []
            
            node_to_meta = state['bikeshare'].get('node_to_metadata', {})
            btxt = []
            for i in depleted:
                u_id = nodes_list[i]
                stop_name = stops_info[u_id]['name']
                meta = node_to_meta.get(i, None)
                if meta:
                    btxt.append(f"🚲 <b>{meta['name']}</b> (Bikeshare)<br>Status: Depleted (Commuter Pressure)<br>Bikes Available: {meta['bikes']}/{meta['capacity']}<br>Empty Docks: {meta['docks']}<br>Snapped Stop: {stop_name}")
                else:
                    btxt.append(f"🚲 <b>Capital Bikeshare Station</b><br>Status: Depleted<br>Snapped Stop: {stop_name}")

            # Route Highlighting Mapping
            route_map = {}
            for r_id, stops in route_to_stops.items():
                indices = [node_to_idx[s] for s in stops if s in node_to_idx]
                if indices: route_map[r_id] = indices

            # --- PHASE 4: MULTIMODAL SYNERGY METRICS CALCULATIONS ---
            # Helper for spatial wavefront propagation
            def haversine_distance(lat1, lon1, lat2, lon2):
                R = 6371000.0 # Earth radius in meters
                phi1 = np.radians(lat1)
                phi2 = np.radians(lat2)
                dphi = np.radians(lat2 - lat1)
                dlambda = np.radians(lon2 - lon1)
                a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2.0)**2
                c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
                return R * c

            # A. Panic Shift Index (Bikeshare depletion vs Metrorail delays)
            curr_depl = set(state['bikeshare']['depleted_node_indices'])
            prev_depl = state.setdefault('history_bikes_set', set())
            new_depletions = curr_depl - prev_depl
            state['history_bikes_set'] = curr_depl
            
            rail_factor = float(state['rail_alerts'])
            panic_shift_val = len(new_depletions) * 15.0 * (1.0 + rail_factor)
            panic_shift_val += len(curr_depl) * 0.5 * (1.0 + rail_factor)
            panic_shift_idx = float(min(max(panic_shift_val, 0.0), 100.0))

            # B. Congestion Wavefront Velocity (Incidents spreading speed)
            incident_nodes = set(state['incidents']['dc'] + state['incidents']['md'] + state['incidents']['va'])
            jammed_nodes = np.where(f < 0.85)[0]
            velocities = []
            new_history = {}
            
            for k in incident_nodes:
                if len(jammed_nodes) > 0:
                    dists = haversine_distance(coords[k, 0], coords[k, 1], coords[jammed_nodes, 0], coords[jammed_nodes, 1])
                    max_dist = float(np.max(dists))
                else:
                    max_dist = 0.0
                    
                prev_max_dist = state['history_distances'].get(k, None)
                if prev_max_dist is not None:
                    v_k = (max_dist - prev_max_dist) / 30.0
                    if v_k > 0:
                        velocities.append(v_k)
                new_history[k] = max_dist
                
            state['history_distances'] = new_history
            wavefront_vel = max(velocities) if len(velocities) > 0 else 0.0
            wavefront_vel = float(min(max(wavefront_vel, 0.0), 25.0))

            # C. Frictional vs. Operational Latency Splits
            frictional_sum = 0.0
            operational_sum = 0.0
            
            for u_id in state['gtfs_delays'].keys():
                delay = float(state['gtfs_delays'][u_id])
                if delay <= 0: continue
                
                speed = state['live_speeds'].get(u_id, None)
                if speed is not None:
                    p_i = 1.0 - min(max(speed / 20.0, 0.01), 1.0)
                    f_delay = delay * p_i
                    o_delay = delay - f_delay
                    frictional_sum += f_delay
                    operational_sum += o_delay
                else:
                    idx = node_to_idx.get(u_id, None)
                    if idx is not None:
                        p_i = 1.0 - float(f[idx])
                        f_delay = delay * p_i
                        o_delay = delay - f_delay
                        frictional_sum += f_delay
                        operational_sum += o_delay
                    else:
                        operational_sum += delay
            
            total_latency = frictional_sum + operational_sum
            if total_latency > 0:
                gridlock_split_pct = float((frictional_sum / total_latency) * 100.0)
                ops_split_pct = float((operational_sum / total_latency) * 100.0)
            else:
                gridlock_split_pct = 50.0
                ops_split_pct = 50.0

            # D. Dynamic Weather Drag Coefficient
            V = np.mean(list(state['live_speeds'].values())) if state['live_speeds'] else 14.5
            P = float(state['precipitation_rate'])
            
            ema = state['weather_ema']
            alpha_ema = 0.95
            ema['V'] = alpha_ema * ema['V'] + (1.0 - alpha_ema) * V
            ema['P'] = alpha_ema * ema['P'] + (1.0 - alpha_ema) * P
            ema['VP'] = alpha_ema * ema['VP'] + (1.0 - alpha_ema) * (V * P)
            ema['P2'] = alpha_ema * ema['P2'] + (1.0 - alpha_ema) * (P * P)
            
            cov_VP = ema['VP'] - ema['V'] * ema['P']
            var_P = ema['P2'] - ema['P'] * ema['P']
            
            if var_P > 1e-4:
                weather_drag_coeff = -cov_VP / var_P
            else:
                weather_drag_coeff = 0.5
            weather_drag_coeff = float(min(max(weather_drag_coeff, 0.1), 5.0))

            # Assemble Live Data Payload
            live_data = {
                "timestamp": time.time(),
                "cycle": state['total_cycles'],
                "metrics": {
                    "l2": float(l2), "gap": float(gap), "weather": state['weather_desc'], "penalty": float(state['weather_penalty']),
                    "dc": len(state['incidents']['dc']), "md": len(state['incidents']['md']), "va": len(state['incidents']['va']),
                    "bike": state['bikeshare']['depleted_stations'], "alerts": state['graph_stats'].get('active_alerts', 0), "rail_alerts": state.get('rail_alerts', 0),
                    "xray_wins": state['scoreboard_stats']['xray_wins'], "wmata_wins": state['scoreboard_stats']['wmata_wins'], 
                    "avg_lead": round(state['scoreboard_stats']['avg_lead_time_sec'] / 60, 1),
                    "panic_shift": panic_shift_idx,
                    "wavefront_velocity": wavefront_vel,
                    "gridlock_split": gridlock_split_pct,
                    "ops_split": ops_split_pct,
                    "weather_drag": weather_drag_coeff
                },
                "gx": gx, "gy": gy, "gz": gz, "wx1": wx1, "wy1": wy1, "wz1": wz1, "wx2": wx2, "wy2": wy2, "wz2": wz2, "fx": fx, "fy": fy, "fz": fz,
                "cx": clear_x, "cy": clear_y, "cz": clear_z, "ctxt": clear_txt,
                "jx": jam_x, "jy": jam_y, "jz": jam_z, "jtxt": jam_txt, "jsiz": jam_size, "jcol": jam_col,
                "bx": bx, "by": by, "bz": bz, "btxt": btxt,
                "routes": route_map,
                "buses": list(state.get('bus_positions_dict', {}).values()),
                "trains": state.get('train_positions', []),
                "bisection": {
                    "partitions": partitions.tolist(),
                    "bx_bound": bx_bound,
                    "by_bound": by_bound,
                    "bz_bound": bz_bound
                }
            }
            
            # --- DVR BUFFER EXPORT ---
            # Save the current state to the live file
            with open("static/live_data.json", "w") as f_json: json.dump(live_data, f_json)
            
            # Save to historical buffer (keep last 30 cycles = ~15 mins)
            history_dir = "static/history"
            if not os.path.exists(history_dir): os.makedirs(history_dir)
            
            # We use modulo 30 to rotate files 0-29
            cycle_idx = state['total_cycles'] % 30
            with open(f"{history_dir}/frame_{cycle_idx}.json", "w") as f_hist: json.dump(live_data, f_hist)
            
            # Write manifest so UI knows the latest frame and total frames available
            with open("static/manifest.json", "w") as f_man: 
                json.dump({"latest_frame": cycle_idx, "total_cycles": state['total_cycles']}, f_man)
                
        except Exception as e:
            import traceback
            print(f"Export Error: {e}")
            traceback.print_exc()

        await asyncio.sleep(max(0, POLL_INTERVAL_GTFS - (time.time() - loop_start)))

async def main():
    init_logger()
    W, nodes_list, node_to_idx, stops_info, tree, route_to_stops = load_static_topology()
    
    # Instantiate task queue objects
    tasks = [
        PollingTask('wmata_vp', 15.0, 1),
        PollingTask('rail_positions', 15.0, 1),
        PollingTask('rideon_vp', 20.0, 1),
        PollingTask('wmata_tu', 30.0, 2),
        PollingTask('rideon_tu', 20.0, 2),
        PollingTask('bikeshare', 120.0, 3),
        PollingTask('metrorail_rt', 120.0, 3),
        PollingTask('incidents', 300.0, 3),
        PollingTask('wmata_alerts', 300.0, 3),
        PollingTask('weather', 900.0, 3)
    ]
    
    await asyncio.gather(
        priority_queue_scheduler(tasks, state, rate_limiter, tree, nodes_list, stops_info, route_to_stops),
        main_loop(W, nodes_list, node_to_idx, stops_info, tree, route_to_stops)
    )

if __name__ == "__main__": asyncio.run(main())
