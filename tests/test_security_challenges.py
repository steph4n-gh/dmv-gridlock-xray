import os
import sys
import json
import unittest
from unittest.mock import Mock, patch
import hmac

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import server

class MockWfile:
    def __init__(self):
        self.data = b""
    def write(self, data):
        self.data += data

class SecurityChallengeHandler(server.Handler):
    def __init__(self, path, headers=None, client_address=("127.0.0.1", 12345)):
        self.path = path
        self.headers = headers or {}
        self.wfile = MockWfile()
        self.rfile = Mock()
        self.client_address = client_address
        self.response_code = None
        self.headers_sent = {}

    def send_response(self, code):
        self.response_code = code

    def send_header(self, name, val):
        self.headers_sent[name] = val

    def end_headers(self):
        pass

class TestSecurityAndAuthChallenges(unittest.TestCase):
    
    # ----------------------------------------------------
    # 1. Directory Traversal and Path Whitelisting Tests
    # ----------------------------------------------------
    
    def test_directory_traversal_standard_passwd(self):
        """Test '/static/../../etc/passwd' returns 404."""
        handler = SecurityChallengeHandler("/static/../../etc/passwd")
        with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
            handler.do_GET()
            self.assertEqual(handler.response_code, 404)
            mock_super_get.assert_not_called()

    def test_directory_traversal_dotdot_suffix(self):
        """Test '/buses.html/..' returns 404."""
        handler = SecurityChallengeHandler("/buses.html/..")
        with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
            handler.do_GET()
            self.assertEqual(handler.response_code, 404)
            mock_super_get.assert_not_called()

    def test_directory_traversal_url_encoded(self):
        """Test '/buses.html%2f..%2f..' returns 404."""
        handler = SecurityChallengeHandler("/buses.html%2f..%2f..")
        with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
            handler.do_GET()
            self.assertEqual(handler.response_code, 404)
            mock_super_get.assert_not_called()

    def test_directory_traversal_backslash(self):
        """Test paths containing backslashes return 404."""
        handler = SecurityChallengeHandler("/static\\..\\..\\etc\\passwd")
        with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
            handler.do_GET()
            self.assertEqual(handler.response_code, 404)
            mock_super_get.assert_not_called()

    def test_directory_traversal_null_byte_traversal(self):
        """Test paths containing URL-encoded null bytes with traversal return 404."""
        handler = SecurityChallengeHandler("/static/%00/../../etc/passwd")
        with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
            handler.do_GET()
            self.assertEqual(handler.response_code, 404)
            mock_super_get.assert_not_called()

    def test_directory_traversal_url_encoded_dots(self):
        """Test paths containing %2e%2e return 404."""
        handler = SecurityChallengeHandler("/static/%2e%2e/%2e%2e/etc/passwd")
        with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
            handler.do_GET()
            self.assertEqual(handler.response_code, 404)
            mock_super_get.assert_not_called()

    def test_null_byte_in_static_path_direct(self):
        """Test how a direct null byte under /static (e.g. /static/foo%00bar) is handled."""
        handler = SecurityChallengeHandler("/static/foo%00bar")
        handler.do_GET()
        self.assertEqual(handler.response_code, 404)

    def test_null_byte_in_whitelisted_path(self):
        """Test that index.html%00 cleanly returns 404 (blocked by whitelisting)."""
        handler = SecurityChallengeHandler("/index.html%00")
        with patch('http.server.SimpleHTTPRequestHandler.do_GET') as mock_super_get:
            handler.do_GET()
            self.assertEqual(handler.response_code, 404)
            mock_super_get.assert_not_called()

    # ----------------------------------------------------
    # 2. Bearer Token Verification Tests
    # ----------------------------------------------------

    def test_bearer_token_missing_header(self):
        """Test API endpoints return 401 when Authorization header is missing."""
        handler = SecurityChallengeHandler("/api/inject")
        with patch.dict(os.environ, {"XRAY_API_KEY": "valid_key"}):
            handler.do_POST()
            self.assertEqual(handler.response_code, 401)
            resp = json.loads(handler.wfile.data.decode('utf-8'))
            self.assertEqual(resp["status"], "error")
            self.assertEqual(resp["message"], "Unauthorized")

    def test_bearer_token_empty_header(self):
        """Test API endpoints return 401 when Authorization header is empty or malformed."""
        handler = SecurityChallengeHandler("/api/inject", headers={"Authorization": ""})
        with patch.dict(os.environ, {"XRAY_API_KEY": "valid_key"}):
            handler.do_POST()
            self.assertEqual(handler.response_code, 401)

    def test_bearer_token_invalid(self):
        """Test API endpoints return 401 when token is invalid."""
        handler = SecurityChallengeHandler("/api/inject", headers={"Authorization": "Bearer invalid_key"})
        with patch.dict(os.environ, {"XRAY_API_KEY": "valid_key"}):
            handler.do_POST()
            self.assertEqual(handler.response_code, 401)

    def test_bearer_token_trailing_spaces(self):
        """Test API endpoints return 401 when token has trailing spaces."""
        handler = SecurityChallengeHandler("/api/inject", headers={"Authorization": "Bearer valid_key "})
        with patch.dict(os.environ, {"XRAY_API_KEY": "valid_key"}):
            handler.do_POST()
            self.assertEqual(handler.response_code, 401)

    def test_bearer_token_constant_time_comparison(self):
        """Verify that hmac.compare_digest is used for constant-time comparison in is_authorized."""
        import inspect
        source = inspect.getsource(server.Handler.is_authorized)
        self.assertIn("hmac.compare_digest", source, "hmac.compare_digest must be used for token comparison")

    # ----------------------------------------------------
    # 3. Exception Handler Challenge Tests
    # ----------------------------------------------------

    def test_exception_handler_sanitized_output_do_GET(self):
        """Test that general exception in do_GET doesn't leak stack trace or internal paths."""
        handler = SecurityChallengeHandler("/")
        with patch('urllib.parse.urlparse', side_effect=Exception("/Users/sarrington/sensitive/path/database_failed.py: Line 42 error")):
            handler.do_GET()
            self.assertEqual(handler.response_code, 500)
            resp = json.loads(handler.wfile.data.decode('utf-8'))
            self.assertEqual(resp["status"], "error")
            self.assertEqual(resp["message"], "Internal server error")
            self.assertNotIn("sarrington", resp["message"])
            self.assertNotIn("traceback", resp["message"])
            self.assertNotIn("Line 42", resp["message"])

    def test_exception_handler_sanitized_output_do_POST_route(self):
        """Test that general exception in do_POST /api/route doesn't leak stack trace or internal paths."""
        handler = SecurityChallengeHandler("/api/route")
        handler.headers = {"Content-Length": "10"}
        handler.rfile.read.return_value = b"{invalid"
        
        handler.do_POST()
        self.assertEqual(handler.response_code, 400)
        resp = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["message"], "Invalid JSON payload")
        self.assertNotIn("traceback", resp["message"])
        self.assertNotIn("Line", resp["message"])

    def test_exception_handler_sanitized_output_do_POST_inject(self):
        """Test that internal exceptions in /api/inject return sanitized response."""
        handler = SecurityChallengeHandler("/api/inject", headers={"Authorization": "Bearer valid_key", "Content-Length": "10"})
        handler.rfile.read.return_value = b'{"node_id": "wmata_1"}'
        # Mock json.loads to raise an exception, but exit the mock before parsing the test's own response
        with patch.dict(os.environ, {"XRAY_API_KEY": "valid_key"}):
            with patch('json.loads', side_effect=Exception("/Volumes/Storage/sensitive_config.py: error")):
                handler.do_POST()
        
        self.assertEqual(handler.response_code, 500)
        resp = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["message"], "Internal server error")
        self.assertNotIn("sensitive_config.py", resp["message"])

    def test_exception_handler_sanitized_output_do_POST_clear(self):
        """Test that internal exceptions in /api/clear return sanitized response."""
        handler = SecurityChallengeHandler("/api/clear", headers={"Authorization": "Bearer valid_key"})
        with patch.dict(os.environ, {"XRAY_API_KEY": "valid_key"}):
            with patch('os.path.exists', side_effect=RuntimeError("filesystem failure on /root/secret")):
                handler.do_POST()
        
        self.assertEqual(handler.response_code, 500)
        resp = json.loads(handler.wfile.data.decode('utf-8'))
        self.assertEqual(resp["status"], "error")
        self.assertEqual(resp["message"], "Internal server error")
        self.assertNotIn("/root/secret", resp["message"])

if __name__ == "__main__":
    unittest.main()
