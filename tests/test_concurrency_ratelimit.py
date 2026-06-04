import os
import sys
import time
import json
import socket
import threading
import unittest
import urllib.request
import urllib.error
import numpy as np
import socketserver

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import server

class TestConcurrencyAndRateLimit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Find a free port
        s = socket.socket()
        s.bind(('127.0.0.1', 0))
        cls.port = s.getsockname()[1]
        s.close()
        
        # Configure and start server
        socketserver.ThreadingTCPServer.allow_reuse_address = True
        cls.httpd = socketserver.ThreadingTCPServer(("127.0.0.1", cls.port), server.Handler)
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.server_thread.start()
        
        # Base url
        cls.base_url = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        # Reset server state before each test
        server._rate_limit_history.clear()
        # Restore original time.time if modified
        server.time.time = time.time
        # Restore original load_and_augment_graph and _compute_route
        self.original_load_graph = server.load_and_augment_graph
        self.original_compute_route = server._compute_route

    def tearDown(self):
        server.load_and_augment_graph = self.original_load_graph
        server._compute_route = self.original_compute_route
        server.time.time = time.time

    def test_concurrency_non_blocking(self):
        """1. Verify that ThreadingTCPServer processes requests concurrently and does not block."""
        # Mock graph and route computation to sleep
        mock_graph = {
            "nodes": ["node_1", "node_2"],
            "aug_coords": np.array([[38.9072, -77.0369], [38.9073, -77.0370]]),
            "coords": np.array([[38.9072, -77.0369], [38.9073, -77.0370]]),
            "aug_names": ["Stop 1", "Stop 2"],
            "partitions": np.array([0, 1]),
            "centrality": np.array([0.5, 0.5]),
            "lambda_2": 0.00005,
        }
        
        server.load_and_augment_graph = lambda: mock_graph
        
        def slow_compute_route(graph, start_idx, end_idx, is_bridge=False):
            time.sleep(0.4)  # force request to take 0.4s
            return {
                "route": [{"lat": 38.9072, "lon": -77.0369, "name": "Stop 1", "mode": "Road", "node_idx": 0}],
                "telemetry": {"estimated_time_sec": 10.0, "baseline_time_sec": 10.0, "delay_exposure_pct": 0.0, "stress_avoided_sec": 0.0},
                "directions": []
            }
        
        server._compute_route = slow_compute_route
        
        # Payloads
        route_payload = {
            "start_lat": 38.9072,
            "start_lon": -77.0369,
            "end_lat": 38.9073,
            "end_lon": -77.0370
        }
        
        # Helper to make request
        results = []
        def make_request(endpoint, payload=None):
            url = f"{self.base_url}{endpoint}"
            data = json.dumps(payload).encode('utf-8') if payload else b""
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            try:
                start = time.time()
                with urllib.request.urlopen(req, timeout=5) as response:
                    body = response.read().decode('utf-8')
                    resp_data = json.loads(body)
                    results.append((response.status, resp_data, time.time() - start))
            except Exception as e:
                results.append((getattr(e, 'code', 500), str(e), 0.0))

        # We will fire 5 concurrent requests: 3 to /api/route, 2 to /api/bridge
        threads = []
        # Clear rate limit so we don't trigger 429
        server._rate_limit_history.clear()
        
        start_time = time.time()
        for i in range(3):
            t = threading.Thread(target=make_request, args=('/api/route', route_payload))
            threads.append(t)
            t.start()
        for i in range(2):
            t = threading.Thread(target=make_request, args=('/api/bridge', None))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        total_duration = time.time() - start_time
        
        # Verify all succeeded
        for status, resp, duration in results:
            self.assertEqual(status, 200, f"Expected 200, got {status}: {resp}")
            self.assertEqual(resp["status"], "success")
            
        # If blocking/sequential, total time would be >= 5 * 0.4s = 2.0s
        # If concurrent, total time should be close to 0.4s (we check if it's less than 0.8s)
        self.assertLess(total_duration, 0.8, f"Total duration was {total_duration:.3f}s, which suggests blocking behavior")
        print(f"Concurrency non-blocking test succeeded: 5 overlapping requests finished in {total_duration:.3f}s")

    def test_rate_limiter_enforcement_and_reset(self):
        """2. Stress test the rate limiter by sending more than 10 requests, verifying 429, and verifying reset after expiration."""
        mock_graph = {
            "nodes": ["node_1", "node_2"],
            "aug_coords": np.array([[38.9072, -77.0369], [38.9073, -77.0370]]),
            "coords": np.array([[38.9072, -77.0369], [38.9073, -77.0370]]),
            "aug_names": ["Stop 1", "Stop 2"],
            "partitions": np.array([0, 1]),
            "centrality": np.array([0.5, 0.5]),
            "lambda_2": 0.00005,
        }
        server.load_and_augment_graph = lambda: mock_graph
        server._compute_route = lambda *args, **kwargs: {
            "route": [], "telemetry": {}, "directions": []
        }
        
        # Setup mock time
        current_mock_time = 1000.0
        def get_mock_time():
            return current_mock_time
        server.time.time = get_mock_time
        
        route_payload = {
            "start_lat": 38.9072,
            "start_lon": -77.0369,
            "end_lat": 38.9073,
            "end_lon": -77.0370
        }
        
        # Send 10 requests from localhost
        url = f"{self.base_url}/api/route"
        data = json.dumps(route_payload).encode('utf-8')
        
        for i in range(10):
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=2) as response:
                self.assertEqual(response.status, 200)
                
        # The 11th request should receive HTTP 429
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=2)
        self.assertEqual(ctx.exception.code, 429)
        
        # Now advance time by 60.1 seconds (sliding window expires)
        current_mock_time = 1060.1
        
        # Request should succeed now
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=2) as response:
            self.assertEqual(response.status, 200)
            
        print("Rate limiter HTTP 429 enforcement and reset test succeeded.")

    def test_rate_limiter_thread_safety_stress(self):
        """3. Verify that the rate limit implementation is thread-safe and does not suffer from race conditions or data corruption."""
        # We will call check_rate_limit concurrently from many threads using a barrier
        num_threads = 50
        barrier = threading.Barrier(num_threads)
        
        results = []
        def worker(ip):
            barrier.wait()  # synchronize start
            allowed = server.check_rate_limit(ip)
            results.append(allowed)
            
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker, args=("mock_ip_threadsafe",))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Verify exactly 10 requests were allowed and 40 were rejected
        num_allowed = sum(1 for r in results if r is True)
        num_rejected = sum(1 for r in results if r is False)
        
        self.assertEqual(num_allowed, 10, f"Expected exactly 10 allowed, got {num_allowed}")
        self.assertEqual(num_rejected, 40, f"Expected exactly 40 rejected, got {num_rejected}")
        
        # Verify no data corruption by checking that the history has length 10
        history = server._rate_limit_history.get("mock_ip_threadsafe", [])
        self.assertEqual(len(history), 10)
        
        # Test concurrent requests with multiple distinct IPs
        server._rate_limit_history.clear()
        ips = [f"ip_{i%5}" for i in range(num_threads)]  # 5 distinct IPs, 10 threads per IP
        results.clear()
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(ips[i],))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        # Each IP should have exactly 10 requests allowed
        for i in range(5):
            ip = f"ip_{i}"
            ip_history = server._rate_limit_history.get(ip, [])
            self.assertEqual(len(ip_history), 10, f"Expected history size 10 for {ip}, got {len(ip_history)}")
            
        print("Rate limiter thread safety and concurrency stress test succeeded.")

    def test_rate_limiter_pruning_unbounded_growth_prevention(self):
        """Verify that the rate limiter prunes empty histories and old timestamps to prevent memory leaks."""
        now = time.time()
        server._rate_limit_history.clear()
        
        server._rate_limit_history["ip_inactive"] = [now - 70.0]
        server._rate_limit_history["ip_active"] = [now - 70.0, now - 10.0]
        server._rate_limit_history["ip_empty"] = []
        
        # Reset counter to 99 so the next request triggers pruning
        server._rate_limit_request_counter = 99
        
        # Check rate limit, which will increment counter to 100 and trigger pruning
        allowed = server.check_rate_limit("ip_new")
        self.assertTrue(allowed)
        self.assertEqual(server._rate_limit_request_counter, 100)
        
        # Inactive IP should be pruned (deleted) because its filtered list is empty
        self.assertNotIn("ip_inactive", server._rate_limit_history)
        self.assertNotIn("ip_empty", server._rate_limit_history)
        
        # Active IP should be updated (the timestamp older than 60s should be pruned)
        self.assertIn("ip_active", server._rate_limit_history)
        self.assertEqual(server._rate_limit_history["ip_active"], [now - 10.0])
        
        # New IP should be present with the new timestamp
        self.assertIn("ip_new", server._rate_limit_history)
        print("Rate limiter pruning test succeeded.")

if __name__ == "__main__":
    unittest.main()
