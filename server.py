import http.server
import socketserver
import json
import os
import time
import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree

PORT = 8501
DIRECTORY = "."

# Cache for the augmented graph to prevent reloading if data hasn't changed
_cache = {
    "mtime": 0,
    "nodes": None,
    "coords": None,
    "names": None,
    "node_friction": None,
    "aug_num_nodes": 0,
    "aug_coords": None,
    "aug_names": None,
    "aug_modes": None,
    "aug_cost_matrix": None,
    "aug_clear_cost_matrix": None,
    "station_name_to_aug_idx": None,
    "unique_stations": None,
    "metro_tunnels": None
}

def get_line_code(name_u, name_v, metro_tunnels):
    for line_code, line_data in metro_tunnels.items():
        stations = [s['name'] for s in line_data['stations']]
        if name_u in stations and name_v in stations:
            idx_u = stations.index(name_u)
            idx_v = stations.index(name_v)
            if abs(idx_u - idx_v) == 1:
                return line_code
    return None

def lookup_cost(csr, u, v):
    row_start = csr.indptr[u]
    row_end = csr.indptr[u+1]
    cols = csr.indices[row_start:row_end]
    idx = np.where(cols == v)[0]
    if len(idx) > 0:
        return float(csr.data[row_start + idx[0]])
    return 0.0

def get_path_cost(path_indices, cost_matrix):
    cost = 0.0
    for idx in range(len(path_indices) - 1):
        u = path_indices[idx]
        v = path_indices[idx+1]
        cost += lookup_cost(cost_matrix, u, v)
    return cost

def load_and_augment_graph():
    path = "network_state.npz"
    tunnels_path = "static/metro_tunnels.json"
    if not os.path.exists(path) or not os.path.exists(tunnels_path):
        return None
    
    mtime = max(os.path.getmtime(path), os.path.getmtime(tunnels_path))
    if _cache["mtime"] == mtime:
        return _cache
    
    # Load base network state
    data = np.load(path, allow_pickle=True)
    nodes = data['nodes']
    coords = data['coords']
    names = data['names']
    node_friction = data['node_friction']
    weights = data['weights']
    indices = data['indices']
    indptr = data['indptr']
    
    # Optional load of bisection features
    partitions = data.get('partitions', None)
    centrality = data.get('centrality', None)
    lambda_2 = float(data.get('lambda_2', [0.001])[0])
    
    N_street = len(nodes)
    
    # Load metro tunnels
    with open(tunnels_path, "r") as f:
        metro_tunnels = json.load(f)
    
    # Identify unique station names
    unique_stations = {}
    for line_code, line_data in metro_tunnels.items():
        for s in line_data['stations']:
            name = s['name']
            if name not in unique_stations:
                unique_stations[name] = {
                    'lat': s['lat'],
                    'lon': s['lon'],
                    'lines': set()
                }
            unique_stations[name]['lines'].add(line_code)
            
    unique_station_names = sorted(list(unique_stations.keys()))
    N_rail = len(unique_station_names)
    
    station_name_to_aug_idx = {name: N_street + i for i, name in enumerate(unique_station_names)}
    
    # Coordinate array for the augmented graph
    aug_coords = np.zeros((N_street + N_rail, 2))
    aug_coords[:N_street] = coords
    for i, name in enumerate(unique_station_names):
        aug_coords[N_street + i] = [unique_stations[name]['lat'], unique_stations[name]['lon']]
        
    aug_names = list(names) + unique_station_names
    aug_modes = ['Road'] * N_street + ['Rail'] * N_rail
    
    edges_from = []
    edges_to = []
    edges_cost = []
    clear_edges_cost = []
    
    # Street edges
    rows = np.repeat(np.arange(N_street), np.diff(indptr))
    cols = indices
    
    lat_u = coords[rows, 0]
    lon_u = coords[rows, 1]
    lat_v = coords[cols, 0]
    lon_v = coords[cols, 1]
    
    dlat = np.radians(lat_v - lat_u)
    dlon = np.radians(lon_v - lon_u)
    lat_mid = np.radians((lat_u + lat_v) / 2.0)
    dist_meters = np.sqrt(dlat**2 + (np.cos(lat_mid) * dlon)**2) * 6371000.0
    
    f_u = np.clip(node_friction[rows], 0.01, 1.0)
    f_v = np.clip(node_friction[cols], 0.01, 1.0)
    street_costs = dist_meters / (13.4 * f_u * f_v)
    street_costs_clear = dist_meters / 13.4
    
    edges_from.extend(rows.tolist())
    edges_to.extend(cols.tolist())
    edges_cost.extend(street_costs.tolist())
    clear_edges_cost.extend(street_costs_clear.tolist())
    
    # Rail tunnel edges
    rail_speed = 18.0 
    for line_code, line_data in metro_tunnels.items():
        stations_list = line_data['stations']
        for i in range(len(stations_list) - 1):
            s_from = stations_list[i]
            s_to = stations_list[i+1]
            idx_from = station_name_to_aug_idx[s_from['name']]
            idx_to = station_name_to_aug_idx[s_to['name']]
            
            dlat = math.radians(s_to['lat'] - s_from['lat'])
            dlon = math.radians(s_to['lon'] - s_from['lon'])
            lat_mid = math.radians((s_from['lat'] + s_to['lat']) / 2.0)
            dist_meters = math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
            
            cost = dist_meters / rail_speed
            edges_from.extend([idx_from, idx_to])
            edges_to.extend([idx_to, idx_from])
            edges_cost.extend([cost, cost])
            clear_edges_cost.extend([cost, cost])
            
    # Virtual Transfer Edges (Dynamic Transfer Penalty)
    weather_penalty = float(data.get('weather_penalty', [1.0])[0])
    transfer_penalty_sec = 300.0 * weather_penalty
    
    transfer_penalty_clear = 300.0
    
    for name, s_info in unique_stations.items():
        rail_idx = station_name_to_aug_idx[name]
        lat_r, lon_r = s_info['lat'], s_info['lon']
        dlat = coords[:, 0] - lat_r
        dlon = coords[:, 1] - lon_r
        dist_sq = dlat**2 + (math.cos(math.radians(lat_r)) * dlon)**2
        closest_street_idx = int(np.argmin(dist_sq))
        
        edges_from.extend([rail_idx, closest_street_idx])
        edges_to.extend([closest_street_idx, rail_idx])
        edges_cost.extend([transfer_penalty_sec, transfer_penalty_sec])
        clear_edges_cost.extend([transfer_penalty_clear, transfer_penalty_clear])
        
    # Virtual Transfer Edges between WMATA and RideOn stops (within 300 meters)
    wmata_indices = [i for i, node in enumerate(nodes) if node.startswith('wmata_')]
    rideon_indices = [i for i, node in enumerate(nodes) if node.startswith('rideon_')]
    
    if wmata_indices and rideon_indices:
        wmata_coords = coords[wmata_indices]
        wmata_tree = KDTree(wmata_coords)
        
        # Max transfer distance 150m in degrees (approx 0.0018)
        max_dist_deg = 0.0018
        
        for r_idx in rideon_indices:
            r_lat, r_lon = coords[r_idx]
            neighbors = wmata_tree.query_ball_point([r_lat, r_lon], max_dist_deg)
            for neighbor_idx in neighbors:
                w_idx = wmata_indices[neighbor_idx]
                w_lat, w_lon = coords[w_idx]
                
                # Haversine distance
                dlat = math.radians(r_lat - w_lat)
                dlon = math.radians(r_lon - w_lon)
                lat_mid = math.radians((r_lat + w_lat) / 2.0)
                dist_m = math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
                
                if dist_m <= 150.0:
                    edges_from.extend([w_idx, r_idx])
                    edges_to.extend([r_idx, w_idx])
                    edges_cost.extend([transfer_penalty_sec, transfer_penalty_sec])
                    clear_edges_cost.extend([transfer_penalty_clear, transfer_penalty_clear])
                    
    aug_num_nodes = N_street + N_rail
    aug_cost_matrix = sp.csr_matrix((edges_cost, (edges_from, edges_to)), shape=(aug_num_nodes, aug_num_nodes))
    aug_clear_cost_matrix = sp.csr_matrix((clear_edges_cost, (edges_from, edges_to)), shape=(aug_num_nodes, aug_num_nodes))
    
    _cache.update({
        "mtime": mtime,
        "nodes": nodes,
        "coords": coords,
        "names": names,
        "node_friction": node_friction,
        "aug_num_nodes": aug_num_nodes,
        "aug_coords": aug_coords,
        "aug_names": aug_names,
        "aug_modes": aug_modes,
        "aug_cost_matrix": aug_cost_matrix,
        "aug_clear_cost_matrix": aug_clear_cost_matrix,
        "station_name_to_aug_idx": station_name_to_aug_idx,
        "unique_stations": unique_stations,
        "metro_tunnels": metro_tunnels,
        "partitions": partitions if partitions is not None else np.zeros(N_street, dtype=np.int32),
        "centrality": centrality if centrality is not None else np.zeros(N_street),
        "lambda_2": lambda_2
    })
    return _cache

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
        
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_POST(self):
        import json
        import os

        if self.path == '/api/inject':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                payload = json.loads(post_data.decode('utf-8'))
                node_id = str(payload.get('node_id', ''))
                
                if node_id:
                    sim_file = "sim_state.json"
                    injected = []
                    if os.path.exists(sim_file):
                        try:
                            with open(sim_file, "r") as f:
                                injected = json.load(f).get("injected_nodes", [])
                        except: pass
                    if node_id not in injected:
                        injected.append(node_id)
                        with open(sim_file, "w") as f:
                            json.dump({"injected_nodes": injected}, f)
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "injected": injected}).encode('utf-8'))
                    return
                else:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Missing node_id"}).encode('utf-8'))
                    return
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
                return
                
        elif self.path == '/api/clear':
            try:
                sim_file = "sim_state.json"
                if os.path.exists(sim_file):
                    os.remove(sim_file)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success", "message": "Simulations cleared"}).encode('utf-8'))
                return
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
                return

        elif self.path == '/api/route':
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length)
                payload = json.loads(post_data.decode('utf-8'))
                
                start_lat = float(payload.get('start_lat', 0.0))
                start_lon = float(payload.get('start_lon', 0.0))
                end_lat = float(payload.get('end_lat', 0.0))
                end_lon = float(payload.get('end_lon', 0.0))
                
                if not (start_lat and start_lon and end_lat and end_lon):
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Missing coordinates"}).encode('utf-8'))
                    return
                
                graph = load_and_augment_graph()
                if not graph:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Graph data not available yet"}).encode('utf-8'))
                    return
                
                N_street = len(graph["nodes"])
                street_coords = graph["aug_coords"][:N_street]
                
                dlat_s = street_coords[:, 0] - start_lat
                dlon_s = street_coords[:, 1] - start_lon
                dist_sq_s = dlat_s**2 + (np.cos(np.radians(start_lat)) * dlon_s)**2
                start_idx = int(np.argmin(dist_sq_s))
                
                dlat_e = street_coords[:, 0] - end_lat
                dlon_e = street_coords[:, 1] - end_lon
                dist_sq_e = dlat_e**2 + (np.cos(np.radians(end_lat)) * dlon_e)**2
                end_idx = int(np.argmin(dist_sq_e))
                
                # Dijkstra Stressed Cost Routing
                dist_matrix, predecessors = dijkstra(
                    graph["aug_cost_matrix"],
                    directed=False,
                    indices=start_idx,
                    return_predecessors=True
                )
                
                if dist_matrix[end_idx] == np.inf:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "No path found between selected coordinates."}).encode('utf-8'))
                    return
                
                # Path reconstruction
                path = []
                curr = end_idx
                while curr != start_idx and curr >= 0:
                    path.append(curr)
                    curr = predecessors[curr]
                if curr == start_idx:
                    path.append(start_idx)
                    path.reverse()
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Reconstruction failed."}).encode('utf-8'))
                    return
                
                # Baseline Clear Dijkstra
                clear_dist_matrix, clear_predecessors = dijkstra(
                    graph["aug_clear_cost_matrix"],
                    directed=False,
                    indices=start_idx,
                    return_predecessors=True
                )
                
                clear_path = []
                curr_c = end_idx
                while curr_c != start_idx and curr_c >= 0:
                    clear_path.append(curr_c)
                    curr_c = clear_predecessors[curr_c]
                if curr_c == start_idx:
                    clear_path.append(start_idx)
                    clear_path.reverse()
                
                friction_time_sec = float(dist_matrix[end_idx])
                baseline_time_sec = float(clear_dist_matrix[end_idx])
                
                stressed_baseline_cost = get_path_cost(clear_path, graph["aug_cost_matrix"])
                stress_avoided_sec = max(0.0, stressed_baseline_cost - friction_time_sec)
                
                num_nodes_path = len(path)
                jammed_nodes_count = sum(1 for node in path if node < N_street and graph["node_friction"][node] < 0.5)
                delay_exposure_pct = float((jammed_nodes_count / max(1, num_nodes_path)) * 100.0)
                
                route_points = []
                directions = []
                legs = []
                curr_leg = None
                
                for idx in path:
                    mode = graph["aug_modes"][idx]
                    lat_c, lon_c = graph["aug_coords"][idx]
                    name_c = graph["aug_names"][idx]
                    
                    point_info = {
                        "lat": float(lat_c),
                        "lon": float(lon_c),
                        "name": name_c,
                        "mode": mode,
                        "node_idx": int(idx)
                    }
                    route_points.append(point_info)
                    
                    node_id = graph["nodes"][idx] if idx < N_street else ""
                    provider = "WMATA" if node_id.startswith("wmata_") else "RideOn" if node_id.startswith("rideon_") else "Rail"
                    
                    if not curr_leg:
                        curr_leg = {"mode": mode, "provider": provider, "indices": [idx]}
                    else:
                        if curr_leg["mode"] == mode and curr_leg["provider"] == provider:
                            curr_leg["indices"].append(idx)
                        else:
                            legs.append(curr_leg)
                            curr_leg = {"mode": mode, "provider": provider, "indices": [idx]}
                if curr_leg:
                    legs.append(curr_leg)
                    
                for idx_leg, leg in enumerate(legs):
                    mode = leg["mode"]
                    provider = leg["provider"]
                    indices = leg["indices"]
                    
                    if idx_leg > 0:
                        prev_leg = legs[idx_leg-1]
                        u = prev_leg["indices"][-1]
                        v = indices[0]
                        mode_u = graph["aug_modes"][u]
                        mode_v = graph["aug_modes"][v]
                        name_u = graph["aug_names"][u]
                        name_v = graph["aug_names"][v]
                        provider_u = prev_leg["provider"]
                        provider_v = provider
                        
                        # Calculate geodesic distance
                        lat_u, lon_u = graph["aug_coords"][u]
                        lat_v, lon_v = graph["aug_coords"][v]
                        dlat = math.radians(lat_v - lat_u)
                        dlon = math.radians(lon_v - lon_u)
                        lat_mid = math.radians((lat_u + lat_v) / 2.0)
                        dist_m = math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
                        
                        # Lookup transfer time
                        transfer_time_sec = lookup_cost(graph["aug_cost_matrix"], u, v)
                        
                        if provider_u == "WMATA" and provider_v == "RideOn":
                            directions.append({
                                "instruction": f"Walk from WMATA Stop ({name_u}) to RideOn Stop ({name_v}) (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        elif provider_u == "RideOn" and provider_v == "WMATA":
                            directions.append({
                                "instruction": f"Walk from RideOn Stop ({name_u}) to WMATA Stop ({name_v}) (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        elif mode_u == "Road" and mode_v == "Rail":
                            directions.append({
                                "instruction": f"Walk into {name_v} Station and board the Metrorail (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        elif mode_u == "Rail" and mode_v == "Road":
                            directions.append({
                                "instruction": f"Exit {name_u} Station and proceed on foot (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        else:
                            directions.append({
                                "instruction": f"Transfer at {name_u} Station hub (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                    
                    if mode == "Road":
                        start_name = graph["aug_names"][indices[0]]
                        end_name = graph["aug_names"][indices[-1]]
                        
                        leg_dist = 0.0
                        for i in range(len(indices)-1):
                            u, v = indices[i], indices[i+1]
                            lat_u, lon_u = graph["aug_coords"][u]
                            lat_v, lon_v = graph["aug_coords"][v]
                            dlat = math.radians(lat_v - lat_u)
                            dlon = math.radians(lon_v - lon_u)
                            lat_mid = math.radians((lat_u + lat_v) / 2.0)
                            leg_dist += math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
                        
                        leg_dist_miles = round(leg_dist / 1609.34, 2)
                        
                        if leg_dist_miles > 0.05:
                            directions.append({
                                "instruction": f"Travel along street network ({start_name.split(' & ')[0]} ➔ {end_name.split(' & ')[0]}) for {leg_dist_miles} miles.",
                                "mode": "Road"
                            })
                        else:
                            directions.append({
                                "instruction": "Proceed along local street grid.",
                                "mode": "Road"
                            })
                            
                    elif mode == "Rail":
                        start_name = graph["aug_names"][indices[0]]
                        end_name = graph["aug_names"][indices[-1]]
                        
                        line_code = get_line_code(start_name, graph["aug_names"][indices[1]] if len(indices) > 1 else start_name, graph["metro_tunnels"])
                        line_display = line_code if line_code else "Metrorail"
                        
                        num_stations = len(indices) - 1
                        directions.append({
                            "instruction": f"Ride the {line_display} Line {num_stations} stations from {start_name} to {end_name}.",
                            "mode": "Rail",
                            "line": line_code
                        })
                                
                modes_breakdown = {"Road": 0.0, "Rail": 0.0, "Transfer": 0.0}
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    m_u = graph["aug_modes"][u]
                    m_v = graph["aug_modes"][v]
                    cost_val = lookup_cost(graph["aug_cost_matrix"], u, v)
                    if m_u != m_v:
                        modes_breakdown["Transfer"] += cost_val
                    else:
                        modes_breakdown[m_u] += cost_val
                    
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                resp_payload = {
                  "status": "success",
                  "route": route_points,
                  "telemetry": {
                    "estimated_time_sec": round(friction_time_sec, 1),
                    "baseline_time_sec": round(baseline_time_sec, 1),
                    "delay_exposure_pct": round(delay_exposure_pct, 1),
                    "stress_avoided_sec": round(stress_avoided_sec, 1),
                    "modes": {
                      "Road": round(modes_breakdown["Road"], 1),
                      "Rail": round(modes_breakdown["Rail"], 1),
                      "Transfer": round(modes_breakdown["Transfer"], 1)
                    }
                  },
                  "directions": directions
                }
                
                self.wfile.write(json.dumps(resp_payload).encode('utf-8'))
                return
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
                return
                
        elif self.path == '/api/bridge':
            try:
                graph = load_and_augment_graph()
                if not graph:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Graph data not available yet"}).encode('utf-8'))
                    return
                
                N_street = len(graph["nodes"])
                partitions = graph.get("partitions", None)
                centrality = graph.get("centrality", None)
                lambda_2 = graph.get("lambda_2", 0.001)
                
                if partitions is None or centrality is None:
                    path = "network_state.npz"
                    data = np.load(path, allow_pickle=True)
                    partitions = data.get('partitions', np.zeros(N_street, dtype=np.int32))
                    centrality = data.get('centrality', np.zeros(N_street))
                    lambda_2 = float(data.get('lambda_2', [0.001])[0])
                
                is_fractured = lambda_2 < 0.0001
                
                mask_v1 = (partitions == 0)
                mask_v2 = (partitions == 1)
                
                centrality_threshold = np.percentile(centrality, 90)
                high_traffic_mask = (centrality >= centrality_threshold)
                
                v1_hubs_idx = np.where(mask_v1 & high_traffic_mask)[0]
                v2_hubs_idx = np.where(mask_v2 & high_traffic_mask)[0]
                
                if len(v1_hubs_idx) == 0:
                    v1_hubs_idx = np.where(mask_v1)[0]
                if len(v2_hubs_idx) == 0:
                    v2_hubs_idx = np.where(mask_v2)[0]
                
                if len(v1_hubs_idx) == 0 or len(v2_hubs_idx) == 0:
                    self.send_response(400)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Degenerate graph: partitions are empty."}).encode('utf-8'))
                    return
                
                coords = graph["coords"]
                lat_1 = coords[v1_hubs_idx, 0]
                lon_1 = coords[v1_hubs_idx, 1]
                lat_2 = coords[v2_hubs_idx, 0]
                lon_2 = coords[v2_hubs_idx, 1]
                
                phi_1 = np.radians(lat_1)[:, np.newaxis]
                phi_2 = np.radians(lat_2)[np.newaxis, :]
                dphi = np.radians(lat_2)[np.newaxis, :] - np.radians(lat_1)[:, np.newaxis]
                dlambda = np.radians(lon_2)[np.newaxis, :] - np.radians(lon_1)[:, np.newaxis]
                
                R = 6371000.0
                a = np.sin(dphi/2.0)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(dlambda/2.0)**2
                c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
                dist_matrix = R * c
                
                min_idx = np.argmin(dist_matrix)
                u_idx_local, v_idx_local = np.unravel_index(min_idx, dist_matrix.shape)
                
                u_star = int(v1_hubs_idx[u_idx_local])
                v_star = int(v2_hubs_idx[v_idx_local])
                geodesic_dist = float(dist_matrix[u_idx_local, v_idx_local])
                
                dist_matrix_dijkstra, predecessors = dijkstra(
                    graph["aug_cost_matrix"],
                    directed=False,
                    indices=u_star,
                    return_predecessors=True
                )
                
                if dist_matrix_dijkstra[v_star] == np.inf:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "No routing path found between Snapped Hubs"}).encode('utf-8'))
                    return
                
                path = []
                curr = v_star
                while curr != u_star and curr >= 0:
                    path.append(curr)
                    curr = predecessors[curr]
                if curr == u_star:
                    path.append(u_star)
                    path.reverse()
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Path reconstruction failed."}).encode('utf-8'))
                    return
                
                clear_dist_matrix, clear_predecessors = dijkstra(
                    graph["aug_clear_cost_matrix"],
                    directed=False,
                    indices=u_star,
                    return_predecessors=True
                )
                
                clear_path = []
                curr_c = v_star
                while curr_c != u_star and curr_c >= 0:
                    clear_path.append(curr_c)
                    curr_c = clear_predecessors[curr_c]
                if curr_c == u_star:
                    clear_path.append(u_star)
                    clear_path.reverse()
                
                friction_time_sec = float(dist_matrix_dijkstra[v_star])
                baseline_time_sec = float(clear_dist_matrix[v_star])
                
                stressed_baseline_cost = get_path_cost(clear_path, graph["aug_cost_matrix"])
                stress_avoided_sec = max(0.0, stressed_baseline_cost - friction_time_sec)
                
                num_nodes_path = len(path)
                jammed_nodes_count = sum(1 for node in path if node < N_street and graph["node_friction"][node] < 0.5)
                delay_exposure_pct = float((jammed_nodes_count / max(1, num_nodes_path)) * 100.0)
                
                route_points = []
                directions = []
                legs = []
                curr_leg = None
                
                for idx in path:
                    mode = graph["aug_modes"][idx]
                    lat_c, lon_c = graph["aug_coords"][idx]
                    name_c = graph["aug_names"][idx]
                    
                    point_info = {
                        "lat": float(lat_c),
                        "lon": float(lon_c),
                        "name": name_c,
                        "mode": mode,
                        "node_idx": int(idx)
                    }
                    route_points.append(point_info)
                    
                    node_id = graph["nodes"][idx] if idx < N_street else ""
                    provider = "WMATA" if node_id.startswith("wmata_") else "RideOn" if node_id.startswith("rideon_") else "Rail"
                    
                    if not curr_leg:
                        curr_leg = {"mode": mode, "provider": provider, "indices": [idx]}
                    else:
                        if curr_leg["mode"] == mode and curr_leg["provider"] == provider:
                            curr_leg["indices"].append(idx)
                        else:
                            legs.append(curr_leg)
                            curr_leg = {"mode": mode, "provider": provider, "indices": [idx]}
                if curr_leg:
                    legs.append(curr_leg)
                
                directions.append({
                    "instruction": f"ESTABLISHED EMERGENCY SHUTTLE BRIDGE between partition hubs: {graph['aug_names'][u_star]} and {graph['aug_names'][v_star]}.",
                    "mode": "Shuttle"
                })
                
                for idx_leg, leg in enumerate(legs):
                    mode = leg["mode"]
                    provider = leg["provider"]
                    indices = leg["indices"]
                    
                    if idx_leg > 0:
                        prev_leg = legs[idx_leg-1]
                        u = prev_leg["indices"][-1]
                        v = indices[0]
                        mode_u = graph["aug_modes"][u]
                        mode_v = graph["aug_modes"][v]
                        name_u = graph["aug_names"][u]
                        name_v = graph["aug_names"][v]
                        provider_u = prev_leg["provider"]
                        provider_v = provider
                        
                        # Calculate geodesic distance
                        lat_u, lon_u = graph["aug_coords"][u]
                        lat_v, lon_v = graph["aug_coords"][v]
                        dlat = math.radians(lat_v - lat_u)
                        dlon = math.radians(lon_v - lon_u)
                        lat_mid = math.radians((lat_u + lat_v) / 2.0)
                        dist_m = math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
                        
                        # Lookup transfer time
                        transfer_time_sec = lookup_cost(graph["aug_cost_matrix"], u, v)
                        
                        if provider_u == "WMATA" and provider_v == "RideOn":
                            directions.append({
                                "instruction": f"Walk from WMATA Stop ({name_u}) to RideOn Stop ({name_v}) (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        elif provider_u == "RideOn" and provider_v == "WMATA":
                            directions.append({
                                "instruction": f"Walk from RideOn Stop ({name_u}) to WMATA Stop ({name_v}) (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        elif mode_u == "Road" and mode_v == "Rail":
                            directions.append({
                                "instruction": f"Walk into {name_v} Station and board the Metrorail (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        elif mode_u == "Rail" and mode_v == "Road":
                            directions.append({
                                "instruction": f"Exit {name_u} Station and proceed on foot (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                        else:
                            directions.append({
                                "instruction": f"Transfer at {name_u} Station hub (Walk Transfer: {dist_m:.1f}m, {transfer_time_sec:.1f}s).",
                                "mode": "Transfer"
                            })
                    
                    start_name = graph["aug_names"][indices[0]]
                    end_name = graph["aug_names"][indices[-1]]
                    
                    leg_dist = 0.0
                    for i in range(len(indices)-1):
                        u, v = indices[i], indices[i+1]
                        lat_u, lon_u = graph["aug_coords"][u]
                        lat_v, lon_v = graph["aug_coords"][v]
                        dlat = math.radians(lat_v - lat_u)
                        dlon = math.radians(lon_v - lon_u)
                        lat_mid = math.radians((lat_u + lat_v) / 2.0)
                        leg_dist += math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
                    
                    leg_dist_miles = round(leg_dist / 1609.34, 2)
                    
                    if mode == "Road":
                        directions.append({
                            "instruction": f"Dispatch shuttle vehicle along optimized street corridor ({start_name.split(' & ')[0]} ➔ {end_name.split(' & ')[0]}) for {leg_dist_miles} miles.",
                            "mode": "Road"
                        })
                    elif mode == "Rail":
                        directions.append({
                            "instruction": f"Leverage clear subterranean rail path between {start_name} and {end_name} for {leg_dist_miles} miles.",
                            "mode": "Rail"
                        })
                
                modes_breakdown = {"Road": 0.0, "Rail": 0.0, "Transfer": 0.0}
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    m_u = graph["aug_modes"][u]
                    m_v = graph["aug_modes"][v]
                    cost_val = lookup_cost(graph["aug_cost_matrix"], u, v)
                    if m_u != m_v:
                        modes_breakdown["Transfer"] += cost_val
                    else:
                        modes_breakdown[m_u] += cost_val
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                resp_payload = {
                  "status": "success",
                  "is_fractured": is_fractured,
                  "lambda_2": lambda_2,
                  "u_star": {
                      "node_idx": u_star,
                      "name": graph["aug_names"][u_star],
                      "lat": float(graph["aug_coords"][u_star][0]),
                      "lon": float(graph["aug_coords"][u_star][1])
                  },
                  "v_star": {
                      "node_idx": v_star,
                      "name": graph["aug_names"][v_star],
                      "lat": float(graph["aug_coords"][v_star][0]),
                      "lon": float(graph["aug_coords"][v_star][1])
                  },
                  "geodesic_dist_meters": round(geodesic_dist, 1),
                  "route": route_points,
                  "telemetry": {
                    "estimated_time_sec": round(friction_time_sec, 1),
                    "baseline_time_sec": round(baseline_time_sec, 1),
                    "delay_exposure_pct": round(delay_exposure_pct, 1),
                    "stress_avoided_sec": round(stress_avoided_sec, 1),
                    "modes": {
                      "Road": round(modes_breakdown["Road"], 1),
                      "Rail": round(modes_breakdown["Rail"], 1),
                      "Transfer": round(modes_breakdown["Transfer"], 1)
                    }
                  },
                  "directions": directions
                }
                
                self.wfile.write(json.dumps(resp_payload).encode('utf-8'))
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
                return
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"🚀 DMV X-Ray Dashboard live at http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server.")
        httpd.serve_forever()
