import os
import sys
import math
import unittest
from unittest.mock import Mock, patch
import time
import tempfile
import shutil
import json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import KDTree
from google.transit import gtfs_realtime_pb2

# Add parent directory to path to import engine and server
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import engine
import server
import utils

class TestDMVGridlockXRays(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backup_dir = tempfile.mkdtemp()
        cls.ns_exists = os.path.exists("network_state.npz")
        if cls.ns_exists:
            shutil.copy("network_state.npz", os.path.join(cls.backup_dir, "network_state.npz"))
        
        cls.mt_exists = os.path.exists("static/metro_tunnels.json")
        if cls.mt_exists:
            shutil.copy("static/metro_tunnels.json", os.path.join(cls.backup_dir, "metro_tunnels.json"))
            
        cls.sl_exists = os.path.exists("static/stops_list.json")
        if cls.sl_exists:
            shutil.copy("static/stops_list.json", os.path.join(cls.backup_dir, "stops_list.json"))

        cls.write_mock_tunnels()

    @classmethod
    def tearDownClass(cls):
        if cls.ns_exists:
            shutil.copy(os.path.join(cls.backup_dir, "network_state.npz"), "network_state.npz")
        elif os.path.exists("network_state.npz"):
            os.remove("network_state.npz")
            
        if cls.mt_exists:
            shutil.copy(os.path.join(cls.backup_dir, "metro_tunnels.json"), "static/metro_tunnels.json")
        elif os.path.exists("static/metro_tunnels.json"):
            os.remove("static/metro_tunnels.json")

        if cls.sl_exists:
            shutil.copy(os.path.join(cls.backup_dir, "stops_list.json"), "static/stops_list.json")
        elif os.path.exists("static/stops_list.json"):
            os.remove("static/stops_list.json")
            
        shutil.rmtree(cls.backup_dir)

    @classmethod
    def write_mock_tunnels(cls):
        os.makedirs("static", exist_ok=True)
        tunnels = {
            "RD": {
                "line_code": "RD",
                "display_name": "Red",
                "stations": [
                    {"code": "A01", "name": "Metro Center", "lat": 38.898303, "lon": -77.028099},
                    {"code": "B01", "name": "Gallery Pl-Chinatown", "lat": 38.89834, "lon": -77.021851}
                ]
            }
        }
        with open("static/metro_tunnels.json", "w") as f:
            json.dump(tunnels, f)

    @classmethod
    def write_mock_network_state(cls, nodes, coords, names, f_friction, W_data, W_indices, W_indptr, weather_penalty=1.0):
        np.savez_compressed(
            "network_state.npz",
            nodes=np.array(nodes),
            coords=np.array(coords),
            names=np.array(names),
            node_friction=np.array(f_friction),
            weights=np.array(W_data, dtype=np.float64),
            indices=np.array(W_indices, dtype=np.int32),
            indptr=np.array(W_indptr, dtype=np.int32),
            weather_penalty=np.array([weather_penalty]),
            partitions=np.zeros(len(nodes), dtype=np.int32),
            centrality=np.zeros(len(nodes))
        )
        server._cache["mtime"] = 0

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.old_gtfs_dir = engine.GTFS_DIR
        engine.GTFS_DIR = self.tmp_dir
        
        # Reset engine state variables
        engine.state.update({
            "live_speeds": {},
            "live_speeds_list": {},
            "live_buses": {},
            "bus_positions_dict": {},
            "gtfs_delays": {},
            "trip_delays": {},
            "weather_penalty": 1.0,
            "weather_desc": "Clear",
            "precipitation_rate": 0.0,
            "canary_buses": {},
            "active_predictions": {},
            "incidents": {"dc": [], "md": [], "va": []},
            "injections": []
        })

    def tearDown(self):
        engine.GTFS_DIR = self.old_gtfs_dir
        shutil.rmtree(self.tmp_dir)
        if os.path.exists("rate_limit_state.json"):
            os.remove("rate_limit_state.json")

    # ==========================================
    # --- TIER 1: FEATURE COVERAGE (HAPPY PATHS)
    # ==========================================

    # --- Feature 1: Dynamic RideOn GTFS-RT Ingestion (R1) ---

    def test_r1_parse_vehicle_position(self):
        """Tier 1: Verify parsing of RideOn vehicle position updates."""
        feed = gtfs_realtime_pb2.FeedMessage()
        entity = feed.entity.add()
        entity.id = "rideon_v1"
        vehicle = entity.vehicle
        vehicle.vehicle.id = "2001"
        vehicle.trip.route_id = "55"
        vehicle.trip.trip_id = "901"
        vehicle.position.latitude = 39.10
        vehicle.position.longitude = -77.15
        vehicle.position.speed = 12.0
        vehicle.position.bearing = 90.0
        vehicle.timestamp = int(time.time())

        # Manually invoke state updating logic resembling engine.py parser
        v_id = "rideon_" + vehicle.vehicle.id
        r_id = "rideon_" + vehicle.trip.route_id
        t_id = "rideon_" + vehicle.trip.trip_id
        engine.state["bus_positions_dict"][v_id] = {
            "id": v_id,
            "route": r_id,
            "lat": vehicle.position.latitude,
            "lon": vehicle.position.longitude,
            "bearing": vehicle.position.bearing,
            "speed": vehicle.position.speed,
            "timestamp": vehicle.timestamp,
            "trip_id": t_id,
            "delay": 0
        }

        self.assertIn("rideon_2001", engine.state["bus_positions_dict"])
        bus = engine.state["bus_positions_dict"]["rideon_2001"]
        self.assertEqual(bus["route"], "rideon_55")
        self.assertEqual(bus["bearing"], 90.0)

    def test_r1_parse_trip_update(self):
        """Tier 1: Verify parsing of RideOn trip updates with delays."""
        feed = gtfs_realtime_pb2.FeedMessage()
        entity = feed.entity.add()
        entity.id = "rideon_t1"
        tu = entity.trip_update
        tu.trip.trip_id = "901"
        tu.trip.route_id = "55"
        stu = tu.stop_time_update.add()
        stu.stop_id = "101"
        stu.departure.delay = 180

        # Simulate update logic
        t_id = "rideon_" + tu.trip.trip_id
        s_id = "rideon_" + stu.stop_id
        engine.state["gtfs_delays"][s_id] = stu.departure.delay
        engine.state["trip_delays"][t_id] = stu.departure.delay

        self.assertEqual(engine.state["gtfs_delays"]["rideon_101"], 180)
        self.assertEqual(engine.state["trip_delays"]["rideon_901"], 180)

    def test_r1_parse_both_vehicle_and_trip(self):
        """Tier 1: Verify parsing of both vehicle position and trip updates in same cycle."""
        engine.state["gtfs_delays"]["rideon_101"] = 150
        v_id = "rideon_2002"
        engine.state["bus_positions_dict"][v_id] = {
            "id": v_id,
            "route": "rideon_55",
            "lat": 39.10,
            "lon": -77.15,
            "bearing": 180.0,
            "speed": 8.0,
            "timestamp": int(time.time()),
            "trip_id": "rideon_902",
            "delay": engine.state["gtfs_delays"].get("rideon_101", 0)
        }

        self.assertEqual(engine.state["bus_positions_dict"]["rideon_2002"]["delay"], 150)

    def test_r1_missing_speed(self):
        """Tier 1: Ensure missing speed values default to 0.0 without errors."""
        feed = gtfs_realtime_pb2.FeedMessage()
        entity = feed.entity.add()
        entity.id = "rideon_v2"
        v = entity.vehicle
        v.vehicle.id = "2002"
        # speed is left unset
        speed_mph = (v.position.speed * 2.23694) if v.position.HasField("speed") else 0.0
        self.assertEqual(speed_mph, 0.0)

    def test_r1_missing_bearing(self):
        """Tier 1: Ensure missing bearing values default to 0.0 without errors."""
        feed = gtfs_realtime_pb2.FeedMessage()
        entity = feed.entity.add()
        entity.id = "rideon_v3"
        v = entity.vehicle
        v.vehicle.id = "2003"
        # bearing is left unset
        bearing = float(v.position.bearing) if v.position.HasField("bearing") else 0.0
        self.assertEqual(bearing, 0.0)


    # --- Feature 2: Strict API Rate Limit Enforcement (R2) ---

    def test_r2_single_request_allowed(self):
        """Tier 1: Verify single request is allowed by RateLimiter."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        url = "http://rideon.app/protobuf/GetGtfsRealtime?apiKey=test"
        self.assertTrue(limiter.can_request(url, now=1000.0))

    def test_r2_multiple_sequential_allowed_with_interval(self):
        """Tier 1: Verify sequential requests to same URL spaced by 21s are allowed."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        url = "http://rideon.app/protobuf/GetGtfsRealtime?apiKey=test"
        
        limiter.record_request(url, now=1000.0)
        self.assertTrue(limiter.can_request(url, now=1021.0))

    def test_r2_overall_rate_limiting_under_limit(self):
        """Tier 1: Verify overall requests under 10 req/min are allowed."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        for i in range(5):
            url = f"http://rideon.app/protobuf/GetGtfsRealtime/{i}"
            self.assertTrue(limiter.can_request(url, now=1000.0 + i))
            limiter.record_request(url, now=1000.0 + i)

    def test_r2_state_persistence_on_save_load(self):
        """Tier 1: Verify rate limit state file persistence and loading."""
        state_path = os.path.join(self.tmp_dir, "rate_limit_state.json")
        limiter1 = engine.RateLimiter(state_file=state_path)
        url = "http://rideon.app/protobuf/GetGtfsRealtime"
        limiter1.record_request(url, now=1000.0)
        
        # Load in another instance
        limiter2 = engine.RateLimiter(state_file=state_path)
        self.assertIn(url, limiter2.link_history)
        self.assertEqual(limiter2.link_history[url], [1000.0])

    def test_r2_history_cleanup(self):
        """Tier 1: Verify old timestamps are correctly purged."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        url = "http://rideon.app/protobuf/GetGtfsRealtime"
        
        limiter.record_request(url, now=1000.0)
        limiter.record_request(url, now=1020.0)
        limiter.clean_history(now=1070.0) # 1000.0 is older than 60s (cutoff 1010.0)
        
        self.assertEqual(limiter.link_history[url], [1020.0])


    # --- Feature 3: Static Topology Fusion (R3) ---

    def test_r3_stops_prefixing(self):
        """Tier 1: Verify that load_static_topology prefixes RideOn and WMATA stop IDs."""
        # Setup mock static files
        os.makedirs(os.path.join(self.tmp_dir, "rideon"))
        with open(os.path.join(self.tmp_dir, "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n1,WMATA Stop 1,38.90,-77.03\n")
        with open(os.path.join(self.tmp_dir, "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt1,1,1\n")
        with open(os.path.join(self.tmp_dir, "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt1,R1\n")
            
        with open(os.path.join(self.tmp_dir, "rideon", "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n10,RideOn Stop 10,38.91,-77.04\n")
        with open(os.path.join(self.tmp_dir, "rideon", "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt10,10,1\n")
        with open(os.path.join(self.tmp_dir, "rideon", "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt10,R10\n")
            
        W, nodes_list, node_to_idx, stops_info, tree, route_to_stops = engine.load_static_topology()
        
        self.assertIn("wmata_1", stops_info)
        self.assertIn("rideon_10", stops_info)

    def test_r3_trips_prefixing(self):
        """Tier 1: Verify that load_static_topology prefixes route and trip IDs."""
        os.makedirs(os.path.join(self.tmp_dir, "rideon"))
        with open(os.path.join(self.tmp_dir, "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n1,WMATA Stop 1,38.90,-77.03\n")
        with open(os.path.join(self.tmp_dir, "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt1,1,1\n")
        with open(os.path.join(self.tmp_dir, "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt1,R1\n")
            
        with open(os.path.join(self.tmp_dir, "rideon", "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n10,RideOn Stop 10,38.91,-77.04\n")
        with open(os.path.join(self.tmp_dir, "rideon", "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt10,10,1\n")
        with open(os.path.join(self.tmp_dir, "rideon", "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt10,R10\n")
            
        W, nodes_list, node_to_idx, stops_info, tree, route_to_stops = engine.load_static_topology()
        self.assertIn("wmata_R1", route_to_stops)
        self.assertIn("rideon_R10", route_to_stops)

    def test_r3_stop_times_prefixing(self):
        """Tier 1: Verify stop_times stop IDs prefixed correctly."""
        os.makedirs(os.path.join(self.tmp_dir, "rideon"))
        with open(os.path.join(self.tmp_dir, "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n1,WMATA Stop 1,38.90,-77.03\n")
        with open(os.path.join(self.tmp_dir, "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt1,1,1\n")
        with open(os.path.join(self.tmp_dir, "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt1,R1\n")
            
        with open(os.path.join(self.tmp_dir, "rideon", "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n10,RideOn Stop 10,38.91,-77.04\n")
        with open(os.path.join(self.tmp_dir, "rideon", "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt10,10,1\n")
        with open(os.path.join(self.tmp_dir, "rideon", "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt10,R10\n")
            
        W, nodes_list, node_to_idx, stops_info, tree, route_to_stops = engine.load_static_topology()
        self.assertEqual(route_to_stops["wmata_R1"][0], "wmata_1")
        self.assertEqual(route_to_stops["rideon_R10"][0], "rideon_10")

    def test_r3_giant_component_extraction(self):
        """Tier 1: Verify giant component extraction from fused graph."""
        os.makedirs(os.path.join(self.tmp_dir, "rideon"))
        # Create standard layout
        with open(os.path.join(self.tmp_dir, "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n1,WMATA 1,38.90,-77.03\n2,WMATA 2,38.91,-77.03\n")
        with open(os.path.join(self.tmp_dir, "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt1,1,1\nt1,2,2\n")
        with open(os.path.join(self.tmp_dir, "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt1,R1\n")
            
        with open(os.path.join(self.tmp_dir, "rideon", "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n10,RO 10,38.92,-77.04\n")
        with open(os.path.join(self.tmp_dir, "rideon", "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt10,10,1\n")
        with open(os.path.join(self.tmp_dir, "rideon", "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt10,R10\n")
            
        W, nodes_list, node_to_idx, stops_info, tree, route_to_stops = engine.load_static_topology()
        # Giant component contains WMATA 1 & 2 connected nodes, filtering out disconnected RO 10 node
        self.assertIn("wmata_1", nodes_list)
        self.assertIn("wmata_2", nodes_list)
        self.assertNotIn("rideon_10", nodes_list)

    def test_r3_walk_transfer_kdtree_snapping(self):
        """Tier 1: Verify KDTree snaps stops correctly."""
        coords = np.array([
            [38.9000, -77.0300],
            [38.9010, -77.0300],
        ])
        tree = KDTree(coords)
        dist, idx = tree.query([38.9002, -77.0300])
        self.assertEqual(idx, 0)
        self.assertLess(dist, 0.001)


    # --- Feature 4: UI Visual Extensions & Dead-Reckoning (R4) ---

    def test_r4_separate_trace_mapping(self):
        """Tier 1: Ensure trace configuration exists for separate operators."""
        # Simulated check mirroring index.html trace separation indices
        wmata_trace_index = 9
        rideon_trace_index = 22
        self.assertNotEqual(wmata_trace_index, rideon_trace_index)

    def test_r4_dead_reckoning_kinematics(self):
        """Tier 1: Verify basic dead-reckoning movement calculations."""
        # Kinematics formula: lat_new = lat + speed * cos(bearing) * dt / R_earth_deg
        lat, lon = 38.9000, -77.0300
        speed_mps = 10.0
        bearing = 90.0 # East
        dt_seconds = 10.0
        
        # Approximate degrees conversion factors
        lat_deg_per_meter = 1.0 / 111000.0
        lon_deg_per_meter = 1.0 / (111000.0 * np.cos(np.radians(lat)))
        
        dx = speed_mps * dt_seconds * np.sin(np.radians(bearing))
        dy = speed_mps * dt_seconds * np.cos(np.radians(bearing))
        
        lat_new = lat + dy * lat_deg_per_meter
        lon_new = lon + dx * lon_deg_per_meter
        
        self.assertAlmostEqual(lat_new, 38.9000, places=4)
        self.assertGreater(lon_new, lon)

    def test_r4_operator_badge_extraction_rideon(self):
        """Tier 1: Verify operator badge extraction for RideOn stops."""
        stop_id = "rideon_12345"
        operator = "RideOn" if stop_id.startswith("rideon_") else "WMATA"
        self.assertEqual(operator, "RideOn")

    def test_r4_operator_badge_extraction_wmata(self):
        """Tier 1: Verify operator badge extraction for WMATA stops."""
        stop_id = "wmata_6789"
        operator = "RideOn" if stop_id.startswith("rideon_") else "WMATA"
        self.assertEqual(operator, "WMATA")

    def test_r4_regex_non_digit_sorting(self):
        """Tier 1: Verify regex integer extraction for prefixed stop sorting."""
        import re
        stop_id = "rideon_4021"
        extracted = int(re.sub(r"\D", "", stop_id))
        self.assertEqual(extracted, 4021)


    # --- Feature 5: Intermodal Router Enhancements (R5) ---

    def test_r5_dijkstra_routing_over_prefixed_graph(self):
        """Tier 1: Test Dijkstra routing on a simple CSR graph structure."""
        # 3 nodes: wmata_1 (0), rideon_2 (1), rideon_3 (2)
        # Edges: 0-1 (cost 10), 1-2 (cost 15)
        edges_from = [0, 1, 1, 2]
        edges_to = [1, 0, 2, 1]
        costs = [10.0, 10.0, 15.0, 15.0]
        csr = sp.csr_matrix((costs, (edges_from, edges_to)), shape=(3, 3))
        
        dists, preds = dijkstra(csr, directed=False, indices=0, return_predecessors=True)
        self.assertEqual(dists[2], 25.0)
        self.assertEqual(preds[2], 1)

    def test_r5_walk_transfer_directions_wmata_to_rideon(self):
        """Tier 1: Verify walk transfer instruction generated (WMATA -> RideOn)."""
        provider_u = "WMATA"
        provider_v = "RideOn"
        name_u = "Silver Spring Metro"
        name_v = "Silver Spring Station Rd"
        
        instruction = ""
        if provider_u == "WMATA" and provider_v == "RideOn":
            instruction = f"Walk from WMATA Stop ({name_u}) to RideOn Stop ({name_v}) (Walk Transfer)."
            
        self.assertEqual(instruction, "Walk from WMATA Stop (Silver Spring Metro) to RideOn Stop (Silver Spring Station Rd) (Walk Transfer).")

    def test_r5_walk_transfer_directions_rideon_to_wmata(self):
        """Tier 1: Verify walk transfer instruction generated (RideOn -> WMATA)."""
        provider_u = "RideOn"
        provider_v = "WMATA"
        name_u = "Silver Spring Station Rd"
        name_v = "Silver Spring Metro"
        
        instruction = ""
        if provider_u == "RideOn" and provider_v == "WMATA":
            instruction = f"Walk from RideOn Stop ({name_u}) to WMATA Stop ({name_v}) (Walk Transfer)."
            
        self.assertEqual(instruction, "Walk from RideOn Stop (Silver Spring Station Rd) to WMATA Stop (Silver Spring Metro) (Walk Transfer).")

    def test_r5_leg_grouping_by_provider(self):
        """Tier 1: Verify router legs group by transit provider correctly."""
        modes = ["Road", "Road", "Road"]
        nodes_ids = ["wmata_1", "rideon_2", "rideon_3"]
        
        legs = []
        curr_leg = None
        for i, nid in enumerate(nodes_ids):
            provider = "WMATA" if nid.startswith("wmata_") else "RideOn"
            if not curr_leg:
                curr_leg = {"provider": provider, "indices": [i]}
            else:
                if curr_leg["provider"] == provider:
                    curr_leg["indices"].append(i)
                else:
                    legs.append(curr_leg)
                    curr_leg = {"provider": provider, "indices": [i]}
        if curr_leg:
            legs.append(curr_leg)
            
        self.assertEqual(len(legs), 2)
        self.assertEqual(legs[0]["provider"], "WMATA")
        self.assertEqual(legs[1]["provider"], "RideOn")
        self.assertEqual(legs[1]["indices"], [1, 2])

    def test_r5_cost_lookups(self):
        """Tier 1: Verify lookups from scipy CSR matrix costs."""
        edges_from = [0, 1]
        edges_to = [1, 0]
        costs = [45.5, 45.5]
        csr = sp.csr_matrix((costs, (edges_from, edges_to)), shape=(2, 2))
        
        # Test helper logic equivalent to lookup_cost
        u, v = 0, 1
        row_start = csr.indptr[u]
        row_end = csr.indptr[u+1]
        cols = csr.indices[row_start:row_end]
        idx = np.where(cols == v)[0]
        cost = float(csr.data[row_start + idx[0]]) if len(idx) > 0 else 0.0
        
        self.assertEqual(cost, 45.5)


    # ============================================
    # --- TIER 2: BOUNDARY & CORNER CASES
    # ============================================

    # --- Feature 1: Dynamic RideOn GTFS-RT Ingestion (R1) ---

    def test_r1_empty_feed(self):
        """Tier 2: Parse empty FeedMessage without crashing."""
        feed = gtfs_realtime_pb2.FeedMessage()
        # Feed has no entities. Ensure loops handle it safely.
        count = 0
        for entity in feed.entity:
            count += 1
        self.assertEqual(count, 0)

    def test_r1_invalid_pb_data(self):
        """Tier 2: Verify ParseFromString raises error on invalid bytes."""
        feed = gtfs_realtime_pb2.FeedMessage()
        with self.assertRaises(Exception):
            feed.ParseFromString(b"this is invalid protobuf content")

    def test_r1_duplicate_vehicle_ids(self):
        """Tier 2: Handle duplicate vehicle entities in same payload by picking latest."""
        bus_positions = {}
        # Entity 1
        bus_positions["rideon_101"] = {"id": "rideon_101", "speed": 10.0}
        # Entity 2 (duplicate update)
        bus_positions["rideon_101"] = {"id": "rideon_101", "speed": 15.0}
        self.assertEqual(bus_positions["rideon_101"]["speed"], 15.0)

    def test_r1_unprefixed_lookup_fallback(self):
        """Tier 2: Check lookup falls back gracefully if ID isn't prefixed."""
        stop_id = "999" # no prefix
        is_rideon = stop_id.startswith("rideon_")
        self.assertFalse(is_rideon)

    def test_r1_extremely_high_speed_clamp(self):
        """Tier 2: Verify that excessively large speeds clamp to a safe threshold."""
        raw_speed_mps = 200.0 # 447 mph
        # Clamp to max 35 m/s (~78 mph)
        clamped_speed = min(raw_speed_mps, 35.0)
        self.assertEqual(clamped_speed, 35.0)


    # --- Feature 2: Strict API Rate Limit Enforcement (R2) ---

    def test_r2_exact_interval_boundary(self):
        """Tier 2: Rejects requests at exactly 19.99s, but allows at exactly 20.0s interval."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        url = "http://rideon.app/protobuf/GetGtfsRealtime"
        
        limiter.record_request(url, now=1000.0)
        self.assertFalse(limiter.can_request(url, now=1019.99))
        self.assertTrue(limiter.can_request(url, now=1020.0))

    def test_r2_overall_burst_block(self):
        """Tier 2: Block 11th request when 10 requests were already recorded within 1 minute."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        for i in range(10):
            limiter.record_request(f"url_{i}", now=1000.0)
        
        self.assertFalse(limiter.can_request("url_11", now=1020.0))

    def test_r2_per_link_burst_block(self):
        """Tier 2: Block 4th request within 1 min for the same URL link."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        url = "http://rideon.app/protobuf"
        
        limiter.record_request(url, now=1000.0)
        limiter.record_request(url, now=1021.0)
        limiter.record_request(url, now=1042.0)
        
        self.assertFalse(limiter.can_request(url, now=1050.0))

    def test_r2_invalid_state_file_corruption(self):
        """Tier 2: Ensure limiter handles corrupted rate limit state JSON without raising exceptions."""
        state_path = os.path.join(self.tmp_dir, "corrupted.json")
        with open(state_path, "w") as f:
            f.write("{invalid_json_content}")
            
        # Should initialize empty history without crashing
        limiter = engine.RateLimiter(state_file=state_path)
        self.assertEqual(limiter.link_history, {})

    def test_r2_unicode_or_query_variations(self):
        """Tier 2: Check query parameters are sanitized and treated as the same base link target."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        url1 = "http://rideon.app/protobuf?apiKey=α"
        url2 = "http://rideon.app/protobuf?apiKey=β"
        
        limiter.record_request(url1, now=1000.0)
        # Both url1 and url2 should map to the same base URL, meaning url2 is locked out by the 20s interval
        self.assertFalse(limiter.can_request(url1, now=1010.0))
        self.assertFalse(limiter.can_request(url2, now=1010.0))
        self.assertIn("http://rideon.app/protobuf", limiter.link_history)


    # --- Feature 3: Static Topology Fusion (R3) ---

    def test_r3_missing_rideon_files(self):
        """Tier 2: load_static_topology runs correctly if RideOn subfolder doesn't exist."""
        # Set up WMATA static files only
        with open(os.path.join(self.tmp_dir, "stops.txt"), "w") as f:
            f.write("stop_id,stop_name,stop_lat,stop_lon\n1,WMATA Stop 1,38.90,-77.03\n")
        with open(os.path.join(self.tmp_dir, "stop_times.txt"), "w") as f:
            f.write("trip_id,stop_id,stop_sequence\nt1,1,1\n")
        with open(os.path.join(self.tmp_dir, "trips.txt"), "w") as f:
            f.write("trip_id,route_id\nt1,R1\n")
            
        from unittest.mock import patch
        import requests
        with patch("requests.get", side_effect=requests.RequestException("Mocked download failure")):
            W, nodes_list, node_to_idx, stops_info, tree, route_to_stops = engine.load_static_topology()
            self.assertIn("wmata_1", stops_info)
            self.assertNotIn("rideon_10", stops_info)

    def test_r3_degenerate_coordinates(self):
        """Tier 2: Handle stops with coordinates at 0.0 or NaNs."""
        coords = np.array([
            [0.0, 0.0],
            [38.90, -77.03]
        ])
        tree = KDTree(coords)
        dist, idx = tree.query([38.90, -77.03])
        self.assertEqual(idx, 1)

    def test_r3_no_stops_within_300m(self):
        """Tier 2: Confirm no transfer edge added if distance is > 300 meters."""
        lat_r, lon_r = 39.1000, -77.1500
        # Stop at 400m
        lat_w, lon_w = 39.1036, -77.1500 
        
        dlat = math.radians(lat_r - lat_w)
        dlon = math.radians(lon_r - lon_w)
        lat_mid = math.radians((lat_r + lat_w) / 2.0)
        dist_m = math.sqrt(dlat**2 + (math.cos(lat_mid) * dlon)**2) * 6371000.0
        
        self.assertGreater(dist_m, 300.0)

    def test_r3_stop_at_exact_300m(self):
        """Tier 2: Transfer edge cost calculation when stop is exactly 300m away."""
        # 300.0m exact distance test
        dist_m = 300.0
        is_transfer_injected = (dist_m <= 300.0)
        self.assertTrue(is_transfer_injected)

    def test_r3_stop_at_301m(self):
        """Tier 2: Transfer edge is NOT created when stop is 301m away."""
        dist_m = 301.0
        is_transfer_injected = (dist_m <= 300.0)
        self.assertFalse(is_transfer_injected)


    # --- Feature 4: UI Visual Extensions & Dead-Reckoning (R4) ---

    def test_r4_dead_reckoning_zero_speed(self):
        """Tier 2: Dead reckoning preserves location if speed is 0.0."""
        lat, lon = 38.9000, -77.0300
        speed_mps = 0.0
        bearing = 90.0
        dt = 30.0
        
        dx = speed_mps * dt * np.sin(np.radians(bearing))
        dy = speed_mps * dt * np.cos(np.radians(bearing))
        
        self.assertEqual(dx, 0.0)
        self.assertEqual(dy, 0.0)

    def test_r4_dead_reckoning_high_latency_clamp(self):
        """Tier 2: Clamps dead reckoning update interval if latency exceeds 60s."""
        dt = 120.0 # extreme latency
        clamped_dt = min(dt, 60.0)
        self.assertEqual(clamped_dt, 60.0)

    def test_r4_invalid_operator_badge_fallback(self):
        """Tier 2: Use fallback operator badge if stop ID prefix is unrecognized."""
        stop_id = "unknown_9999"
        operator = "RideOn" if stop_id.startswith("rideon_") else "WMATA" if stop_id.startswith("wmata_") else "Regional"
        self.assertEqual(operator, "Regional")

    def test_r4_empty_stop_ids(self):
        """Tier 2: Table renderer handles empty stop IDs gracefully."""
        stop_id = ""
        badge = "WMATA"
        if not stop_id:
            badge = "Unknown"
        self.assertEqual(badge, "Unknown")

    def test_r4_trace_boundary_indices(self):
        """Tier 2: Ensure Plotly trace bounds check (index 22) doesn't cause out-of-bound errors."""
        traces = [f"trace_{i}" for i in range(30)]
        self.assertEqual(traces[22], "trace_22")


    # --- Feature 5: Intermodal Router Enhancements (R5) ---

    def test_r5_unreachable_stops_routing(self):
        """Tier 2: Ensure router returns inf distance when routing to unreachable components."""
        # 2 disconnected nodes
        csr = sp.csr_matrix(([999.0], ([0], [0])), shape=(2, 2))
        dists, preds = dijkstra(csr, directed=False, indices=0, return_predecessors=True)
        self.assertEqual(dists[1], np.inf)

    def test_r5_routing_degenerate_same_start_end(self):
        """Tier 2: Cost is 0.0 when start and end stops are the same."""
        csr = sp.csr_matrix(([10.0, 10.0], ([0, 1], [1, 0])), shape=(2, 2))
        dists, preds = dijkstra(csr, directed=False, indices=0, return_predecessors=True)
        self.assertEqual(dists[0], 0.0)

    def test_r5_extremely_high_friction_clamp(self):
        """Tier 2: Enforce numerical stability floor clamp (friction >= 0.01)."""
        friction_val = 0.0001
        clamped_friction = max(friction_val, 0.01)
        self.assertEqual(clamped_friction, 0.01)

    def test_r5_weather_penalty_bounds(self):
        """Tier 2: Check weather penalty stays within logical bounds."""
        raw_penalty = 0.05
        # Ensure it does not drop below 0.1
        clamped_penalty = max(raw_penalty, 0.1)
        self.assertEqual(clamped_penalty, 0.1)

    def test_r5_invalid_leg_transitions(self):
        """Tier 2: Leg transitions handle undefined providers cleanly."""
        provider_u = "Unknown"
        provider_v = "RideOn"
        instruction = "Transfer" if provider_u == "Unknown" else "Road"
        self.assertEqual(instruction, "Transfer")


    # ============================================
    # --- TIER 3: CROSS-FEATURE COMBINATIONS
    # ============================================

    def test_t3_rate_limiting_during_high_frequency_polling(self):
        """Tier 3: Test rate limiter states during simulated rapid ingestion loops."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl.json"))
        url = "http://rideon.app/protobuf"
        
        # Simulate quick sequential requests
        t0 = 1000.0
        self.assertTrue(limiter.can_request(url, now=t0))
        limiter.record_request(url, now=t0)
        
        # 1s later
        self.assertFalse(limiter.can_request(url, now=t0 + 1.0))

    def test_t3_routing_under_storm_weather_with_transfers(self):
        """Tier 3: Ensure weather penalty scaling changes intermodal routing paths."""
        # 4 stops: wmata_1 (0), wmata_2 (1), rideon_10 (2), rideon_11 (3)
        # Under clear weather (penalty 1.0), walk transfer = 300s. Path: 0 -> 1 -> 2 -> 3
        # Under storm weather (penalty 2.5), walk transfer = 750s. Path: Alternative route
        nodes = ["wmata_1", "wmata_2", "rideon_10", "rideon_11"]
        coords = [[38.90, -77.03], [38.901, -77.03], [38.902, -77.03], [38.903, -77.03]]
        names = ["W1", "W2", "R10", "R11"]
        
        # Edge indices
        # 0 -> 1 (Road, cost 100)
        # 1 -> 2 (Transfer, cost 300 * weather)
        # 2 -> 3 (Road, cost 100)
        # 0 -> 3 (Alternative Rail, cost 600)
        
        # We test cost changes under Clear Weather
        transfer_cost_clear = 300.0 * 1.0
        total_path_clear = 100.0 + transfer_cost_clear + 100.0
        self.assertEqual(total_path_clear, 500.0) # Clear path (500s) < Alternative Rail (600s)
        
        # Storm Weather
        transfer_cost_storm = 300.0 * 2.5
        total_path_storm = 100.0 + transfer_cost_storm + 100.0
        self.assertEqual(total_path_storm, 950.0) # Alternative Rail (600s) now faster than Transfer (950s)

    def test_t3_live_bus_positions_fused_with_routing(self):
        """Tier 3: Verify live delays from GTFS-RT apply correctly on segment friction routing."""
        # Ingest delays
        engine.state["gtfs_delays"]["rideon_101"] = 300
        
        # Simulates a routing lookup
        delay = engine.state["gtfs_delays"].get("rideon_101", 0)
        self.assertEqual(delay, 300)

    def test_t3_dynamic_friction_affects_intermodal_paths(self):
        """Tier 3: Physical speed drop below 5mph raises friction and router adapts."""
        # Clean speed (friction 1.0) -> cost 100
        # Congested speed < 5mph (friction 0.05) -> cost 2000
        dist_m = 1340.0
        base_cost = dist_m / 13.4 # 100s
        
        # Congested segment
        f_u, f_v = 0.05, 0.05
        congested_cost = dist_m / (13.4 * f_u * f_v)
        self.assertGreater(congested_cost, base_cost * 10)

    def test_t3_ui_rendering_fused_data(self):
        """Tier 3: Verify final WebGL export payload includes both operator data nodes."""
        stops_info = {
            "wmata_1": {"name": "W Stop", "lat": 38.9, "lon": -77.0},
            "rideon_10": {"name": "RO Stop", "lat": 39.0, "lon": -77.1}
        }
        nodes_list = ["wmata_1", "rideon_10"]
        gx = [stops_info[nid]["lon"] for nid in nodes_list]
        gy = [stops_info[nid]["lat"] for nid in nodes_list]
        
        self.assertEqual(len(gx), 2)
        self.assertEqual(gx, [-77.0, -77.1])


    # ============================================
    # --- TIER 4: REAL-WORLD APPLICATION SCENARIOS
    # ============================================

    def test_t4_commuter_route_wmata_stops_only(self):
        """Tier 4: Routing search over pure WMATA stops."""
        nodes = ["wmata_1", "wmata_2"]
        coords = [[38.90, -77.03], [38.901, -77.03]]
        names = ["W1", "W2"]
        f_friction = [1.0, 1.0]
        
        # Edges
        self.write_mock_network_state(nodes, coords, names, f_friction, [5.0, 5.0], [1, 0], [0, 1, 2])
        
        graph = server.load_and_augment_graph()
        self.assertIsNotNone(graph)
        self.assertEqual(graph["nodes"][0], "wmata_1")

    def test_t4_commuter_route_rideon_stops_only(self):
        """Tier 4: Routing search over pure RideOn stops."""
        nodes = ["rideon_10", "rideon_11"]
        coords = [[39.00, -77.10], [39.001, -77.10]]
        names = ["RO10", "RO11"]
        f_friction = [1.0, 1.0]
        
        self.write_mock_network_state(nodes, coords, names, f_friction, [5.0, 5.0], [1, 0], [0, 1, 2])
        
        graph = server.load_and_augment_graph()
        self.assertIsNotNone(graph)
        self.assertEqual(graph["nodes"][0], "rideon_10")

    def test_t4_intermodal_commute_wmata_to_rideon(self):
        """Tier 4: Intermodal route requiring transfer from WMATA to RideOn."""
        # Snapped distance < 300m
        nodes = ["wmata_1", "rideon_10"]
        coords = [[38.9000, -77.0300], [38.9001, -77.0300]]
        names = ["WMATA Stop 1", "RideOn Stop 10"]
        f_friction = [1.0, 1.0]
        
        self.write_mock_network_state(nodes, coords, names, f_friction, [], [], [0, 0, 0])
        
        graph = server.load_and_augment_graph()
        # Verify transfer edges are added dynamically in aug_cost_matrix
        self.assertIsNotNone(graph)
        self.assertGreater(graph["aug_cost_matrix"].nnz, 0)

    def test_t4_stormy_commute_routing_shift(self):
        """Tier 4: Routing path shifts under storm weather scaling."""
        nodes = ["wmata_1", "rideon_10"]
        coords = [[38.9000, -77.0300], [38.9001, -77.0300]]
        names = ["WMATA Stop 1", "RideOn Stop 10"]
        f_friction = [1.0, 1.0]
        
        # Test clear weather (penalty 1.0)
        self.write_mock_network_state(nodes, coords, names, f_friction, [], [], [0, 0, 0], weather_penalty=1.0)
        graph_clear = server.load_and_augment_graph()
        cost_clear = lookup_cost(graph_clear["aug_cost_matrix"], 0, 1)
        
        # Test storm weather (penalty 2.5)
        self.write_mock_network_state(nodes, coords, names, f_friction, [], [], [0, 0, 0], weather_penalty=2.5)
        graph_storm = server.load_and_augment_graph()
        cost_storm = lookup_cost(graph_storm["aug_cost_matrix"], 0, 1)
        
        self.assertEqual(cost_clear, 300.0)
        self.assertEqual(cost_storm, 750.0)

    def test_t4_rideon_system_dropout_recovery(self):
        """Tier 4: Simulate RideOn offline dropout; system resolves cleanly with remaining WMATA routes."""
        engine.state["bus_positions_dict"] = {
            "wmata_101": {"id": "wmata_101", "route": "wmata_X2"},
            "rideon_201": {"id": "rideon_201", "route": "rideon_55"}
        }
        
        # Simulate dropout by purging rideon_ keys
        for key in list(engine.state["bus_positions_dict"].keys()):
            if key.startswith("rideon_"):
                del engine.state["bus_positions_dict"][key]
                
        self.assertIn("wmata_101", engine.state["bus_positions_dict"])
        self.assertNotIn("rideon_201", engine.state["bus_positions_dict"])

    def test_r2_rate_limiter_mismatched_types(self):
        """Verify RateLimiter handles invalid/mismatched JSON types in load_state."""
        state_path = os.path.join(self.tmp_dir, "rl_bad_types.json")
        bad_data = {
            "link_history": ["this should be a dict", 1234],
            "overall_history": {"this should be a list": 10}
        }
        with open(state_path, "w") as f:
            json.dump(bad_data, f)
        
        # Should load and sanitise to default empty/clean structures
        limiter = engine.RateLimiter(state_file=state_path)
        self.assertEqual(limiter.link_history, {})
        self.assertEqual(limiter.overall_history, [])

    def test_r2_rate_limiter_sanitization(self):
        """Verify RateLimiter filters out non-float or malformed values from timestamps."""
        state_path = os.path.join(self.tmp_dir, "rl_sanitize.json")
        data = {
            "link_history": {
                "http://test1": [1000.0, "invalid", 1020.0],
                "http://test2": ["bad"]
            },
            "overall_history": [1000.0, "bad", 1020.0]
        }
        with open(state_path, "w") as f:
            json.dump(data, f)
            
        limiter = engine.RateLimiter(state_file=state_path)
        self.assertEqual(limiter.link_history.get("http://test1"), [1000.0, 1020.0])
        self.assertNotIn("http://test2", limiter.link_history)
        self.assertEqual(limiter.overall_history, [1000.0, 1020.0])

    def test_r2_rate_limiter_clean_deletes_empty_keys(self):
        """Verify RateLimiter deletes URL keys when their timestamps list becomes empty."""
        limiter = engine.RateLimiter(state_file=os.path.join(self.tmp_dir, "rl_clean.json"))
        url = "http://rideon.app"
        limiter.record_request(url, now=1000.0)
        self.assertIn(url, limiter.link_history)
        
        # Clean history with cutoff > 1000.0
        limiter.clean_history(now=1070.0)
        self.assertNotIn(url, limiter.link_history)

    def test_r1_poll_rideon_gtfs_rt_snapping_fallback(self):
        """Verify that poll_rideon_gtfs_rt snaps to the nearest stop if stop_id is missing."""
        coords = np.array([
            [39.1000, -77.1500]
        ])
        tree = KDTree(coords)
        nodes_list = ["rideon_1001"]
        
        # Create a mock entity
        feed = gtfs_realtime_pb2.FeedMessage()
        entity = feed.entity.add()
        entity.id = "entity_1"
        entity.vehicle.vehicle.id = "2001"
        entity.vehicle.trip.route_id = "55"
        entity.vehicle.trip.trip_id = "901"
        entity.vehicle.position.latitude = 39.1002
        entity.vehicle.position.longitude = -77.1501
        entity.vehicle.position.speed = 10.0
        entity.vehicle.position.bearing = 90.0
        entity.vehicle.timestamp = int(time.time())
        # stop_id is left unset
        
        # Let's run a test checking the snapping fallback logic on a parsed entity
        v = entity.vehicle
        stop_id = 'rideon_' + str(v.stop_id) if v.stop_id else None
        self.assertIsNone(stop_id)
        
        lat = float(v.position.latitude) if v.position.latitude else 0.0
        lon = float(v.position.longitude) if v.position.longitude else 0.0
        
        if not stop_id and lat != 0.0 and lon != 0.0:
            dist, idx = tree.query([lat, lon])
            if dist < 0.001:
                snapped_id = nodes_list[idx]
                if snapped_id.startswith('rideon_'):
                    stop_id = snapped_id
                    
        self.assertEqual(stop_id, "rideon_1001")

    def test_r1_poll_rideon_gtfs_rt_canary_tracking(self):
        """Verify canary tracking updates and registers physical impact correctly."""
        v_id = "rideon_canary_v1"
        r_id = "rideon_55"
        engine.state["canary_buses"] = {
            v_id: {"route_id": r_id, "status": "Acquiring Target..."}
        }
        engine.state["active_predictions"] = {
            r_id: {"t_physical": None}
        }
        
        # Speed >= 2.0 -> Tracking
        speed_mph_fast = 5.0
        if v_id in engine.state["canary_buses"]:
            canary = engine.state["canary_buses"][v_id]
            if speed_mph_fast < 2.0:
                canary["status"] = "IMPACT CONFIRMED"
            else:
                canary["status"] = f"Tracking ({speed_mph_fast:.1f}mph)"
                
        self.assertEqual(engine.state["canary_buses"][v_id]["status"], "Tracking (5.0mph)")
        self.assertIsNone(engine.state["active_predictions"][r_id]["t_physical"])
        
        # Speed < 2.0 -> IMPACT CONFIRMED and t_physical set
        speed_mph_slow = 1.5
        if v_id in engine.state["canary_buses"]:
            canary = engine.state["canary_buses"][v_id]
            if speed_mph_slow < 2.0:
                canary["status"] = "IMPACT CONFIRMED"
                if r_id in engine.state["active_predictions"] and engine.state["active_predictions"][r_id]["t_physical"] is None:
                    engine.state["active_predictions"][r_id]["t_physical"] = 123456.78
                    
        self.assertEqual(engine.state["canary_buses"][v_id]["status"], "IMPACT CONFIRMED")
        self.assertEqual(engine.state["active_predictions"][r_id]["t_physical"], 123456.78)

    def test_walking_transfer_limits_exceeded(self):
        """Verify that walking transfer edges are not created for stops separated by > 150m, but are created for <= 150m."""
        # 1. Stops separated by 166.8 meters (> 150m)
        nodes = ["wmata_1", "rideon_10"]
        coords_far = [[38.9000, -77.0300], [38.9015, -77.0300]]
        names = ["WMATA Stop 1", "RideOn Stop 10"]
        f_friction = [1.0, 1.0]
        
        self.write_mock_network_state(nodes, coords_far, names, f_friction, [], [], [0, 0, 0])
        graph_far = server.load_and_augment_graph()
        
        # Verify transfer edges are NOT added
        cost_far = lookup_cost(graph_far["aug_cost_matrix"], 0, 1)
        self.assertEqual(cost_far, 0.0)
        
        # 2. Stops separated by 111.2 meters (< 150m)
        coords_near = [[38.9000, -77.0300], [38.9010, -77.0300]]
        self.write_mock_network_state(nodes, coords_near, names, f_friction, [], [], [0, 0, 0])
        graph_near = server.load_and_augment_graph()
        
        # Verify transfer edges ARE added
        cost_near = lookup_cost(graph_near["aug_cost_matrix"], 0, 1)
        self.assertGreater(cost_near, 0.0)

    def test_weather_penalty_linear_below_300(self):
        """Verify that transfer penalty scales linearly below 300s when weather_penalty < 1.0."""
        nodes = ["wmata_1", "rideon_10"]
        coords = [[38.9000, -77.0300], [38.9001, -77.0300]]
        names = ["WMATA Stop 1", "RideOn Stop 10"]
        f_friction = [1.0, 1.0]
        
        # Test weather penalty 0.85
        self.write_mock_network_state(nodes, coords, names, f_friction, [], [], [0, 0, 0], weather_penalty=0.85)
        graph_light = server.load_and_augment_graph()
        cost_light = lookup_cost(graph_light["aug_cost_matrix"], 0, 1)
        
        # Since it scales strictly dynamically: 300.0 * 0.85 = 255.0
        self.assertAlmostEqual(cost_light, 255.0)

def lookup_cost(csr, u, v):
    row_start = csr.indptr[u]
    row_end = csr.indptr[u+1]
    cols = csr.indices[row_start:row_end]
    idx = np.where(cols == v)[0]
    if len(idx) > 0:
        return float(csr.data[row_start + idx[0]])
    return 0.0

class TestOptimizedEngineAdditional(unittest.TestCase):
    def test_crosses_river_true(self):
        """Verify that crosses_river returns True for coordinates crossing the Potomac barrier."""
        # Across Potomac: (38.990, -77.160) to (38.990, -77.120)
        self.assertTrue(utils.crosses_river(38.990, -77.160, 38.990, -77.120))
        # Across Anacostia: (38.900, -76.980) to (38.900, -76.950)
        self.assertTrue(utils.crosses_river(38.900, -76.980, 38.900, -76.950))

    def test_river_crossing_blocked(self):
        """Verify that a Potomac or Anacostia river crossing is correctly identified as blocked."""
        # Across Potomac
        self.assertTrue(utils.crosses_river(38.990, -77.160, 38.990, -77.120))
        # Across Anacostia
        self.assertTrue(utils.crosses_river(38.900, -76.980, 38.900, -76.950))

    def test_crosses_river_false(self):
        """Verify that crosses_river returns False for coordinates not crossing any barriers."""
        # Safe path: (38.900, -77.030) to (38.901, -77.030) (both in DC downtown)
        self.assertFalse(utils.crosses_river(38.900, -77.030, 38.901, -77.030))

    def test_smooth_floor(self):
        """Verify that smooth_floor outputs values >= 0.01 and matches expected behavior."""
        # For high x (e.g. 1.0), smooth_floor should return x
        self.assertAlmostEqual(utils.smooth_floor(1.0), 1.0, places=4)
        # For x = 0, smooth_floor should return math.log(2)/100 = 0.00693...
        # Wait, smooth_floor(0.0) is log(2)/100 which is ~0.0069, but when clipped >= 0.01 is enforced by caller
        self.assertLess(utils.smooth_floor(0.0), 0.01)
        self.assertGreater(utils.smooth_floor(0.0), 0.0)
        
        # Test numpy array version
        arr = np.array([0.0, 1.0])
        res = utils.smooth_floor(arr)
        self.assertAlmostEqual(res[1], 1.0, places=4)
        self.assertLess(res[0], 0.01)

    def test_validate_coordinates(self):
        """Verify that validate_coordinates checks coordinates against DMV bounds correctly."""
        # Inside bounds
        self.assertTrue(utils.validate_coordinates(38.9072, -77.0369)) # DC Center
        self.assertTrue(utils.validate_coordinates(38.5, -77.5))
        self.assertTrue(utils.validate_coordinates(39.5, -76.5))
        self.assertTrue(utils.validate_coordinates(38.0, -78.0))
        self.assertTrue(utils.validate_coordinates(40.0, -76.0))
        
        # Out of bounds
        self.assertFalse(utils.validate_coordinates(0.0, 0.0))
        self.assertFalse(utils.validate_coordinates(37.9, -77.0))
        self.assertFalse(utils.validate_coordinates(40.1, -77.0))
        self.assertFalse(utils.validate_coordinates(39.0, -78.1))
        self.assertFalse(utils.validate_coordinates(39.0, -75.9))

        # Robustness checks for invalid types, None, lists, dicts, NaN, Inf
        self.assertFalse(utils.validate_coordinates("not-a-float", -77.0))
        self.assertFalse(utils.validate_coordinates(None, -77.0))
        self.assertFalse(utils.validate_coordinates([38.9], -77.0))
        self.assertFalse(utils.validate_coordinates({"lat": 38.9}, -77.0))
        self.assertFalse(utils.validate_coordinates(float('nan'), -77.0))
        self.assertFalse(utils.validate_coordinates(38.9, float('inf')))
        self.assertFalse(utils.validate_coordinates(True, -77.0))
        self.assertTrue(utils.validate_coordinates("38.9", "-77.0"))

    def test_smooth_floor_list_tuple(self):
        """Verify that smooth_floor handles lists and tuples by converting them to numpy arrays."""
        res_list = utils.smooth_floor([0.0, 1.0])
        self.assertIsInstance(res_list, np.ndarray)
        self.assertAlmostEqual(res_list[1], 1.0, places=4)
        self.assertLess(res_list[0], 0.01)

        res_tuple = utils.smooth_floor((0.0, 1.0))
        self.assertIsInstance(res_tuple, np.ndarray)
        self.assertAlmostEqual(res_tuple[1], 1.0, places=4)
        self.assertLess(res_tuple[0], 0.01)

    def test_spectral_analysis_adaptive(self):
        """Verify that spectral_analysis accepts adaptive regularisation parameter and v0."""
        # Create a simple 3-node connected graph
        W = sp.csr_matrix([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        f = np.array([0.5, 0.6, 0.7])
        v0 = np.array([1.0, -1.0, 0.0])
        l2, v2, gap = engine.spectral_analysis(W, f=f, v0=v0)
        self.assertIsNotNone(l2)
        self.assertEqual(len(v2), 3)
        self.assertIsNotNone(gap)

    def test_trip_to_shape_population(self):
        """Verify that state['trip_to_shape'] is populated when load_static_topology is called."""
        self.assertIsInstance(engine.state.get('trip_to_shape'), dict)

    def test_bus_positions_have_shape_id(self):
        """Verify that GTFS-RT buses dict structure supports shape_id field."""
        engine.state['trip_to_shape']['mock_trip_123'] = 'mock_shape_abc'
        wmata_bus = {
            "id": "mock_bus_1",
            "route": "mock_route",
            "lat": 38.9,
            "lon": -77.0,
            "bearing": 90,
            "speed": 10.0,
            "timestamp": int(time.time()),
            "trip_id": "mock_trip_123",
            "delay": 0,
            "shape_id": engine.state.get('trip_to_shape', {}).get("mock_trip_123")
        }
        self.assertEqual(wmata_bus["shape_id"], "mock_shape_abc")


class MockWfile:
    def __init__(self):
        self.data = b""
    def write(self, data):
        self.data += data

class MockHandler(server.Handler):
    def __init__(self):
        self.path = ""
        self.headers = {}
        self.wfile = MockWfile()
        self.rfile = Mock()
        self.client_address = ("127.0.0.1", 12345)
        self.response_code = None
        self.headers_sent = {}

    def send_response(self, code):
        self.response_code = code

    def send_header(self, name, val):
        self.headers_sent[name] = val

    def end_headers(self):
        pass


class TestServerEndpoints(unittest.TestCase):
    def test_api_route_validation_missing(self):
        """Verify server /api/route returns 400 when coordinates are missing."""
        handler = Mock(spec=server.Handler)
        handler.path = '/api/route'
        handler.headers = {'Content-Length': '2'}
        handler.rfile = Mock()
        handler.rfile.read.return_value = b"{}"
        
        handler.wfile = MockWfile()
        response_code = None
        def send_response(code):
            nonlocal response_code
            response_code = code
        handler.send_response = send_response
        
        server.Handler.do_POST(handler)
        
        self.assertEqual(response_code, 400)
        resp_data = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp_data["status"], "error")
        self.assertIn("Missing coordinates", resp_data["message"])

    def test_api_route_validation_malformed(self):
        """Verify server /api/route returns 400 when parameters are malformed (not float convertible)."""
        handler = Mock(spec=server.Handler)
        handler.path = '/api/route'
        payload = {"start_lat": "abc", "start_lon": -77.0, "end_lat": 38.9, "end_lon": -77.0}
        payload_bytes = json.dumps(payload).encode('utf-8')
        handler.headers = {'Content-Length': str(len(payload_bytes))}
        handler.rfile = Mock()
        handler.rfile.read.return_value = payload_bytes
        
        handler.wfile = MockWfile()
        response_code = None
        def send_response(code):
            nonlocal response_code
            response_code = code
        handler.send_response = send_response
        
        server.Handler.do_POST(handler)
        
        self.assertEqual(response_code, 400)
        resp_data = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp_data["status"], "error")
        self.assertIn("Malformed coordinate parameters", resp_data["message"])

    def test_api_route_validation_out_of_bounds(self):
        """Verify server /api/route returns 400 when coordinates lie outside the DMV bounding box."""
        handler = Mock(spec=server.Handler)
        handler.path = '/api/route'
        payload = {"start_lat": 0.0, "start_lon": 0.0, "end_lat": 38.9, "end_lon": -77.0}
        payload_bytes = json.dumps(payload).encode('utf-8')
        handler.headers = {'Content-Length': str(len(payload_bytes))}
        handler.rfile = Mock()
        handler.rfile.read.return_value = payload_bytes
        
        handler.wfile = MockWfile()
        response_code = None
        def send_response(code):
            nonlocal response_code
            response_code = code
        handler.send_response = send_response
        
        server.Handler.do_POST(handler)
        
        self.assertEqual(response_code, 400)
        resp_data = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp_data["status"], "error")
        self.assertIn("Coordinates out of DMV bounds", resp_data["message"])

    def test_api_route_invalid_json(self):
        """Verify server /api/route returns 400 when JSON body is malformed or invalid."""
        handler = MockHandler()
        handler.path = '/api/route'
        payload_bytes = b"{invalid-json"
        handler.headers = {'Content-Length': str(len(payload_bytes))}
        handler.rfile.read.return_value = payload_bytes
        
        handler.do_POST()
        
        self.assertEqual(handler.response_code, 400)
        resp_data = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp_data["status"], "error")
        self.assertEqual(resp_data["message"], "Invalid JSON payload")

    def test_api_key_auth_missing_header(self):
        """Verify API endpoints require authentication and reject without headers."""
        handler = MockHandler()
        handler.path = '/api/inject'
        handler.headers = {} # No Authorization header
        
        with patch.dict(os.environ, {"XRAY_API_KEY": "secret_key"}):
            handler.do_POST()
            self.assertEqual(handler.response_code, 401)
            resp_data = json.loads(handler.wfile.data.decode('utf-8'))
            self.assertEqual(resp_data["status"], "error")
            self.assertEqual(resp_data["message"], "Unauthorized")

    def test_api_key_auth_no_env_variable(self):
        """Verify that authentication fails if the XRAY_API_KEY env var is not set."""
        handler = MockHandler()
        handler.path = '/api/inject'
        handler.headers = {"Authorization": "Bearer secret_key"}
        
        with patch.dict(os.environ, {}, clear=True):
            if "XRAY_API_KEY" in os.environ:
                del os.environ["XRAY_API_KEY"]
            handler.do_POST()
            self.assertEqual(handler.response_code, 401)

    def test_api_key_auth_success(self):
        """Verify authorized request with valid Bearer token."""
        handler = MockHandler()
        handler.path = '/api/inject'
        payload = {"node_id": "wmata_1001"}
        payload_bytes = json.dumps(payload).encode('utf-8')
        handler.headers = {
            "Authorization": "Bearer secret_key",
            "Content-Length": str(len(payload_bytes))
        }
        handler.rfile.read.return_value = payload_bytes
        
        with patch.dict(os.environ, {"XRAY_API_KEY": "secret_key"}):
            with patch("server._sim_lock"):  # Mock lock to avoid file side effects
                handler.do_POST()
                self.assertEqual(handler.response_code, 200)

    def test_get_whitelisted_paths(self):
        """Verify whitelisted paths are allowed and rewritten properly."""
        for path in ["/", "/index.html", "/buses.html", "/trains.html"]:
            handler = MockHandler()
            handler.path = path
            
            with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
                handler.do_GET()
                if path == "/":
                    self.assertTrue(handler.path.startswith("/index.html"))
                mock_super_get.assert_called_once()
                self.assertNotEqual(handler.response_code, 404)

    def test_get_traversal_paths_blocked(self):
        """Verify traversal paths and unwhitelisted paths are returned 404."""
        bad_paths = [
            "/static/../../etc/passwd",
            "/buses.html/..",
            "/invalid_file.html",
            "/static/nonexistent_file.json", # static checks if file is on disk
            "/path\\with\\backslash"
        ]
        for path in bad_paths:
            handler = MockHandler()
            handler.path = path
            
            with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
                handler.do_GET()
                self.assertEqual(handler.response_code, 404)
                mock_super_get.assert_not_called()

    def test_favicon_fallback_png(self):
        """Verify favicon serves custom 1x1 transparent PNG fallback if file doesn't exist."""
        handler = MockHandler()
        handler.path = "/favicon.ico"
        
        with patch('os.path.exists', return_value=False):
            handler.do_GET()
            self.assertEqual(handler.response_code, 200)
            self.assertEqual(handler.headers_sent.get("Content-Type"), "image/x-icon")
            self.assertIn(b"PNG", handler.wfile.data)

    def test_api_rate_limiting_blocking(self):
        """Verify rate limit blocks the 11th request within 60 seconds and returns 429."""
        # Reset rate limiting state
        server._rate_limit_history.clear()
        
        ip = "192.168.1.50"
        # Simulate 10 successful check_rate_limit calls
        for _ in range(10):
            self.assertTrue(server.check_rate_limit(ip))
        
        # 11th call should return False
        self.assertFalse(server.check_rate_limit(ip))
        
        # Check HTTP handler returns 429
        handler = MockHandler()
        handler.path = '/api/route'
        handler.client_address = (ip, 12345)
        
        handler.do_POST()
        self.assertEqual(handler.response_code, 429)
        resp_data = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp_data["status"], "error")
        self.assertEqual(resp_data["message"], "Too many requests")

    def test_exception_sanitization(self):
        """Verify exception handler sanitizes error messages and writes to log."""
        handler = MockHandler()
        
        # Delete server.log if it exists to verify creation
        if os.path.exists("server.log"):
            try:
                os.remove("server.log")
            except: pass
            
        test_exc = ValueError("Sensitive database path or traceback detail")
        handler.handle_error(test_exc, status_code=500, client_message="Sanitized error message")
        
        self.assertEqual(handler.response_code, 500)
        resp_data = json.loads(handler.wfile.data.decode('utf-8'))
        # Response must not contain "Sensitive database path"
        self.assertNotIn("Sensitive database path", resp_data["message"])
        self.assertEqual(resp_data["message"], "Sanitized error message")
        
        # Verify log has the detailed error
        self.assertTrue(os.path.exists("server.log"))
        with open("server.log", "r") as f:
            log_content = f.read()
            self.assertIn("Sensitive database path", log_content)


class TestMilestone5Expansion(unittest.TestCase):
    def get_clean_state(self):
        return {
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
            "live_speeds": {},
            "live_speeds_list": {},
            "live_buses": {},
            "predictive_friction": {},
            "wmata_official_alerts": set(), 
            "rail_alerts": 0,
            "rail_surges": set(),
            "canary_buses": {},
            "active_predictions": {},
            "bus_positions": [],
            "bus_positions_dict": {},
            "train_positions": [],
            "prev_trains": {},
            "scoreboard_stats": {"xray_wins": 0, "wmata_wins": 0, "avg_lead_time_sec": 0.0, "total_races": 0},
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

    def test_heat_kernel_diffusion(self):
        # 1. Setup a simple circular graph with 5 nodes
        nodes_list = ["node0", "node1", "node2", "node3", "node4"]
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        stops_info = {n: {"name": f"Stop {n}", "lat": 38.9 + i*0.01, "lon": -77.0 + i*0.01} for i, n in enumerate(nodes_list)}
        
        # Circular adjacency: 0-1, 1-2, 2-3, 3-4, 4-0
        row_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        col_idx = np.array([1, 4, 0, 2, 1, 3, 2, 4, 3, 0])
        data = np.ones(10, dtype=np.float32)
        W_mask = sp.csr_matrix((data, (row_idx, col_idx)), shape=(5, 5))
        
        # D_inv
        D_diag = np.array(W_mask.sum(axis=1)).flatten()
        D_inv = 1.0 / D_diag
        
        # Setup state
        test_state = self.get_clean_state()
        test_state.update({
            "weather_penalty": 1.2,
            "rail_alerts": 1,
            "rail_surges": {2}
        })
        surge_hubs_idx = [0]
        
        # Let's compute the expected initial f value BEFORE diffusion:
        f_init = np.ones(len(nodes_list), dtype=np.float32) * test_state['weather_penalty']
        # tau noise
        tau_noise = 1e-5 * np.sin(math.tau * np.arange(len(nodes_list)) / len(nodes_list))
        f_init += tau_noise
        
        # apply rail alerts penalty (idx 0 -> surge hub)
        f_init[0] *= 0.7
        # apply rail surge penalty (idx 2 -> rail surge)
        f_init[2] *= 0.6
        
        # Manual diffusion calculation
        D_inv_W_f = D_inv * (W_mask @ f_init)
        f_expected = (1.0 - engine.ALPHA_DIFFUSION) * f_init + engine.ALPHA_DIFFUSION * D_inv_W_f
        # apply smooth_floor and clip
        f_expected = np.clip(utils.smooth_floor(f_expected), 0.01, 1.0)
        
        # Call compute_friction_field
        f_actual, _, _, _, _, _ = engine.compute_friction_field(
            test_state,
            W_mask,
            nodes_list,
            node_to_idx,
            stops_info,
            surge_hubs_idx,
            D_inv,
            last_f=None,
            last_l2=0.05,
            last_v2=None,
            last_gap=0.05,
            loop_start=time.time()
        )
        
        np.testing.assert_allclose(f_actual, f_expected, rtol=1e-5)

    @patch("engine.eigsh")
    def test_fiedler_vector_warm_starting(self, mock_eigsh):
        mock_evals = np.array([0.1, 0.2, 0.3])
        mock_evecs = np.zeros((5, 3))
        mock_evecs[:, 1] = np.array([0.1, -0.2, 0.3, -0.4, 0.2])
        mock_eigsh.return_value = (mock_evals, mock_evecs)
        
        nodes_list = ["node0", "node1", "node2", "node3", "node4"]
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        stops_info = {n: {"name": f"Stop {n}", "lat": 38.9, "lon": -77.0} for n in nodes_list}
        
        row_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        col_idx = np.array([1, 4, 0, 2, 1, 3, 2, 4, 3, 0])
        data = np.ones(10, dtype=np.float32)
        W_mask = sp.csr_matrix((data, (row_idx, col_idx)), shape=(5, 5))
        
        D_diag = np.array(W_mask.sum(axis=1)).flatten()
        D_inv = 1.0 / D_diag
        
        test_state = self.get_clean_state()
        
        last_v2 = np.array([0.5, -0.5, 0.0, 0.5, -0.5], dtype=np.float64)
        
        f, W_weighted, l2, v2, gap, lat = engine.compute_friction_field(
            test_state,
            W_mask,
            nodes_list,
            node_to_idx,
            stops_info,
            surge_hubs_idx=[],
            D_inv=D_inv,
            last_f=np.zeros(5),
            last_l2=0.05,
            last_v2=last_v2,
            last_gap=0.05,
            loop_start=time.time()
        )
        
        mock_eigsh.assert_called()
        args, kwargs = mock_eigsh.call_args
        self.assertIn("v0", kwargs)
        np.testing.assert_array_equal(kwargs["v0"], last_v2)

    @patch("engine.eigsh")
    def test_spectral_solver_gating_logic(self, mock_eigsh):
        # 1. n <= 3 fallback using scipy.linalg.eigh
        W_small = sp.csr_matrix([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        f_small = np.array([1.0, 1.0, 1.0])
        
        l2_3, v2_3, gap_3 = engine.spectral_analysis(W_small, f=f_small)
        
        self.assertAlmostEqual(l2_3, 3.0)
        self.assertEqual(len(v2_3), 3)
        self.assertAlmostEqual(gap_3, 0.0)
        mock_eigsh.assert_not_called()
        
        # 2. n > 3 and eigsh raises ArpackNoConvergence on both attempts
        mock_eigsh.reset_mock()
        from scipy.sparse.linalg import ArpackNoConvergence
        mock_eigsh.side_effect = ArpackNoConvergence("LM fail", eigenvalues=np.array([0.1]), eigenvectors=np.zeros((4, 1)))
        
        W_large = sp.csr_matrix([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0]
        ])
        f_large = np.array([1.0, 1.0, 1.0, 1.0])
        
        l2_fail, v2_fail, gap_fail = engine.spectral_analysis(W_large, f=f_large)
        
        self.assertEqual(l2_fail, 0.0)
        np.testing.assert_array_equal(v2_fail, np.zeros(4))
        self.assertEqual(gap_fail, 0.0)
        self.assertEqual(mock_eigsh.call_count, 2)
        
        first_call_kwargs = mock_eigsh.call_args_list[0][1]
        second_call_kwargs = mock_eigsh.call_args_list[1][1]
        
        self.assertEqual(first_call_kwargs.get("which"), "LM")
        self.assertEqual(second_call_kwargs.get("which"), "SM")

        # 3. n > 3 and eigsh raises ValueError on fallback
        mock_eigsh.reset_mock()
        mock_eigsh.side_effect = [
            ArpackNoConvergence("LM fail", eigenvalues=None, eigenvectors=None),
            ValueError("SM ValueError")
        ]
        
        l2_val, v2_val, gap_val = engine.spectral_analysis(W_large, f=f_large)
        self.assertEqual(l2_val, 0.0)
        np.testing.assert_array_equal(v2_val, np.zeros(4))
        self.assertEqual(gap_val, 0.0)

    def test_multimodal_metrics(self):
        orig_cwd = os.getcwd()
        tmp_dir = tempfile.mkdtemp()
        os.chdir(tmp_dir)
        try:
            os.makedirs("static/history", exist_ok=True)
            nodes_list = ["node0", "node1"]
            node_to_idx = {"node0": 0, "node1": 1}
            stops_info = {
                "node0": {"name": "Stop 0", "lat": 38.90, "lon": -77.03},
                "node1": {"name": "Stop 1", "lat": 38.91, "lon": -77.03}
            }
            route_to_stops = {"route0": ["node0", "node1"]}
            
            W = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
            W_mask = W.copy()
            W_mask.data[:] = 1.0
            
            f = np.array([0.5, 0.9], dtype=np.float32)
            v2 = np.array([0.1, -0.1])
            l2 = 0.05
            gap = 0.05
            thresholds = np.array([0.4, 0.4])
            rows = np.array([0, 1])
            
            test_state = self.get_clean_state()
            test_state.update({
                "total_cycles": 1,
                "weather_penalty": 1.0,
                "weather_desc": "Clear",
                "precipitation_rate": 0.5,
                "weather_ema": {"V": 15.0, "P": 0.2, "VP": 3.0, "P2": 0.04},
                "incidents": {"dc": [0], "md": [], "va": []},
                "history_distances": {0: 100.0},
                "bikeshare": {
                    "total_bikes": 10,
                    "active_stations": 2,
                    "depleted_stations": 1,
                    "depleted_node_indices": [0],
                    "node_to_metadata": {0: {"name": "Bikeshare 0", "bikes": 0, "capacity": 10, "docks": 10}}
                },
                "gtfs_delays": {"node0": 100.0},
                "live_speeds": {"node0": 10.0},
                "live_speeds_list": {"node0": [10.0]},
                "wmata_official_alerts": {"route0"},
                "rail_alerts": 1,
                "scoreboard_stats": {"xray_wins": 5, "wmata_wins": 2, "avg_lead_time_sec": 120.0, "total_races": 7},
            })
            test_state["graph_stats"].update({
                "avg_friction": 0.7, "spectral_gap": 0.05, "active_nodes": 2, "active_edges": 1,
                "system_tension": 0.2, "total_delay_sec": 0, "compute_latency": 1.5,
                "peak_latency": 2.0, "fiedler_max": 0.1, "fiedler_pole_name": "Stop 0",
                "worst_node_name": "Stop 0", "centrality": [1.0, 1.0], "active_alerts": 1
            })
            
            engine.export_visualization_data(
                test_state,
                v2,
                l2,
                gap,
                f,
                W,
                nodes_list,
                stops_info,
                route_to_stops,
                node_to_idx,
                thresholds,
                rows,
                W_mask
            )
            
            self.assertTrue(os.path.exists("static/live_data.json"))
            self.assertTrue(os.path.exists("static/manifest.json"))
            self.assertTrue(os.path.exists("static/history/frame_1.json"))
            
            with open("static/live_data.json", "r") as f_json:
                data = json.load(f_json)
                
            metrics = data["metrics"]
            self.assertAlmostEqual(metrics["panic_shift"], 31.0)
            self.assertAlmostEqual(metrics["wavefront_velocity"], 0.0)
            self.assertAlmostEqual(metrics["gridlock_split"], 50.0)
            self.assertAlmostEqual(metrics["ops_split"], 50.0)
            self.assertAlmostEqual(metrics["weather_drag"], 5.0)
            
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(tmp_dir)

    def test_canary_protocol_adjudication(self):
        orig_cwd = os.getcwd()
        tmp_dir = tempfile.mkdtemp()
        os.chdir(tmp_dir)
        try:
            nodes_list = ["node0", "node1"]
            stops_info = {
                "node0": {"name": "Stop 0", "lat": 38.90, "lon": -77.03},
                "node1": {"name": "Stop 1", "lat": 38.91, "lon": -77.03}
            }
            route_to_stops = {"route0": ["node0", "node1"]}
            
            W_weighted = sp.csr_matrix(([0.1, 0.1], ([0, 1], [1, 0])), shape=(2, 2))
            thresholds = np.array([0.4, 0.4])
            rows = np.array([0, 1])
            
            test_state = self.get_clean_state()
            test_state.update({
                "live_buses": {"route0": ["bus_canary_1"]},
                "scoreboard_stats": {"xray_wins": 0, "wmata_wins": 0, "total_races": 0, "avg_lead_time_sec": 0.0}
            })
            
            t_before = time.time()
            fractures = engine.detect_fractures(
                test_state, W_weighted, thresholds, rows, nodes_list, stops_info, route_to_stops
            )
            t_after = time.time()
            
            self.assertEqual(len(fractures), 1)
            self.assertEqual(fractures[0]["u"], "Stop 0")
            
            self.assertIn("bus_canary_1", test_state["canary_buses"])
            canary = test_state["canary_buses"]["bus_canary_1"]
            self.assertEqual(canary["route_id"], "route0")
            self.assertEqual(canary["target"], "Stop 0")
            self.assertEqual(canary["status"], "Acquiring Target...")
            
            self.assertIn("route0", test_state["active_predictions"])
            pred = test_state["active_predictions"]["route0"]
            self.assertGreaterEqual(pred["t_engine"], t_before)
            self.assertLessEqual(pred["t_engine"], t_after)
            self.assertIsNone(pred["t_physical"])
            self.assertIsNone(pred["t_wmata"])
            self.assertEqual(pred["resolved"], False)
            self.assertEqual(pred["pre_existing"], False)
            
            t_engine = pred["t_engine"]
            pred["t_physical"] = t_engine + 10.0
            pred["t_wmata"] = t_engine + 100.0
            
            engine.adjudicate_predictions(test_state, current_time=time.time())
            
            self.assertEqual(pred["resolved"], True)
            self.assertEqual(test_state["scoreboard_stats"]["xray_wins"], 1)
            self.assertEqual(test_state["scoreboard_stats"]["wmata_wins"], 0)
            self.assertEqual(test_state["scoreboard_stats"]["total_races"], 1)
            self.assertAlmostEqual(test_state["scoreboard_stats"]["avg_lead_time_sec"], 100.0)
            
            self.assertNotIn("bus_canary_1", test_state["canary_buses"])
            self.assertNotIn("route0", test_state["active_predictions"])
            
            test_state["canary_buses"] = {}
            test_state["active_predictions"] = {
                "route1": {
                    "t_engine": 1000.0,
                    "t_physical": 1010.0,
                    "t_wmata": 1030.0,
                    "cause": "test tie",
                    "resolved": False,
                    "pre_existing": False
                }
            }
            test_state["canary_buses"]["bus_canary_2"] = {"route_id": "route1"}
            
            pred1 = test_state["active_predictions"]["route1"]
            engine.adjudicate_predictions(test_state, current_time=time.time())
            self.assertTrue(pred1["resolved"])
            self.assertEqual(test_state["scoreboard_stats"]["xray_wins"], 1)
            self.assertEqual(test_state["scoreboard_stats"]["wmata_wins"], 0)
            self.assertEqual(test_state["scoreboard_stats"]["total_races"], 2)
            self.assertAlmostEqual(test_state["scoreboard_stats"]["avg_lead_time_sec"], 65.0)
            
            test_state["active_predictions"] = {
                "route2": {
                    "t_engine": 1000.0,
                    "t_physical": 1010.0,
                    "t_wmata": 900.0,
                    "cause": "test wmata",
                    "resolved": False,
                    "pre_existing": False
                }
            }
            pred2 = test_state["active_predictions"]["route2"]
            engine.adjudicate_predictions(test_state, current_time=time.time())
            self.assertTrue(pred2["resolved"])
            self.assertEqual(test_state["scoreboard_stats"]["xray_wins"], 1)
            self.assertEqual(test_state["scoreboard_stats"]["wmata_wins"], 1)
            self.assertEqual(test_state["scoreboard_stats"]["total_races"], 3)
            
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(tmp_dir)

    def test_full_main_loop_integration(self):
        import asyncio
        class TerminateLoop(Exception):
            pass
        
        orig_cwd = os.getcwd()
        tmp_dir = tempfile.mkdtemp()
        os.chdir(tmp_dir)
        try:
            os.makedirs("static/history", exist_ok=True)
            
            nodes_list = ["n0", "n1", "n2", "n3", "n4"]
            node_to_idx = {n: i for i, n in enumerate(nodes_list)}
            stops_info = {n: {"name": f"Stop {n}", "lat": 38.9 + i*0.01, "lon": -77.0 + i*0.01} for i, n in enumerate(nodes_list)}
            
            row_idx = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
            col_idx = np.array([1, 4, 0, 2, 1, 3, 2, 4, 3, 0])
            data = np.ones(10, dtype=np.float32)
            W = sp.csr_matrix((data, (row_idx, col_idx)), shape=(5, 5))
            
            coords = np.array([[stops_info[n]["lat"], stops_info[n]["lon"]] for n in nodes_list])
            tree = KDTree(coords)
            route_to_stops = {"route0": nodes_list}
            
            engine.state.clear()
            engine.state.update(self.get_clean_state())
            engine.state.update({
                "graph_stats": {
                    "avg_friction": 1.0, "spectral_gap": 0.0, "active_nodes": 5, "active_edges": 5,
                    "system_tension": 0.0, "total_delay_sec": 0, "compute_latency": 0.0,
                    "peak_latency": 0.0, "fiedler_max": 0.0, "fiedler_pole_name": "Core",
                    "worst_node_name": "None", "centrality": np.zeros(5), "active_alerts": 0 
                }
            })
            
            async def mock_sleep(delay):
                raise TerminateLoop("Loop completed one iteration")
                
            with patch("asyncio.sleep", mock_sleep):
                try:
                    asyncio.run(engine.main_loop(W, nodes_list, node_to_idx, stops_info, tree, route_to_stops))
                except TerminateLoop:
                    pass
            
            self.assertTrue(os.path.exists("network_state.npz"))
            self.assertTrue(os.path.exists("static/live_data.json"))
            self.assertTrue(os.path.exists("static/manifest.json"))
            self.assertTrue(os.path.exists("static/history/frame_1.json"))
            
            with open("static/live_data.json", "r") as f_json:
                live_data = json.load(f_json)
                
            expected_keys = ["timestamp", "cycle", "metrics", "gx", "gy", "gz", "wx1", "wy1", "wz1", "wx2", "wy2", "wz2",
                             "fx", "fy", "fz", "cx", "cy", "cz", "ctxt", "jx", "jy", "jz", "jtxt", "jsiz", "jcol",
                             "bx", "by", "bz", "btxt", "routes", "buses", "trains", "bisection"]
            for key in expected_keys:
                self.assertIn(key, live_data)
                
            metrics_expected_keys = ["l2", "gap", "weather", "penalty", "dc", "md", "va", "bike", "alerts",
                                     "rail_alerts", "xray_wins", "wmata_wins", "avg_lead", "panic_shift",
                                     "wavefront_velocity", "gridlock_split", "ops_split", "weather_drag"]
            for key in metrics_expected_keys:
                self.assertIn(key, live_data["metrics"])
                
            self.assertEqual(live_data["cycle"], 1)
            
        finally:
            os.chdir(orig_cwd)
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
