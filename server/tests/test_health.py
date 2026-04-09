import unittest
from pathlib import Path
import sys

from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from main import app


class HealthEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload.get("status"), "healthy")
        self.assertEqual(payload.get("language_model", {}).get("active_provider"), "gemini")
        self.assertEqual(payload.get("image_to_3d", {}).get("mode"), "triposg")


if __name__ == "__main__":
    unittest.main()
