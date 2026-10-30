import os
import unittest
from unittest.mock import patch
try:
    from fastapi.testclient import TestClient
    from api import app,controls
except ImportError:
    app=None

@unittest.skipIf(app is None, 'Optional API dependencies not installed')
class OperationsTests(unittest.TestCase):
    def test_authentication_liveness_and_tracing(self):
        with patch.dict(os.environ,{'PATHDSP_API_KEY':'test-only-value'}):
            with TestClient(app) as client:
                self.assertEqual(client.get('/health/live').status_code,200)
                self.assertEqual(client.get('/v1/model-card').status_code,401)
                response=client.get('/v1/model-card',headers={'Authorization':'Bearer test-only-value'})
                self.assertEqual(response.status_code,200)
                self.assertIn('X-Request-ID',response.headers)
                self.assertIn('Server-Timing',response.headers)
                self.assertEqual(response.headers['Cache-Control'],'no-store')
                self.assertIn('Unavailable',response.json()['prediction_interval_status'])

    def test_rate_limit_and_bounded_client_memory(self):
        from pathdsp.operations import ApiControls
        control=ApiControls()
        self.assertTrue(control.allow('a',1)[0]);self.assertFalse(control.allow('a',1)[0])
        for i in range(1100):control.allow(str(i),1)
        self.assertLessEqual(len(control.buckets),1024)
        with patch.dict(os.environ,{'PATHDSP_API_KEY':'','PATHDSP_REQUESTS_PER_MINUTE':'1'}):
            controls.buckets.clear()
            with TestClient(app) as client:
                self.assertEqual(client.get('/v1/model-card').status_code,200)
                response=client.get('/v1/model-card')
                self.assertEqual(response.status_code,429)
                self.assertIn('Retry-After',response.headers)
        controls.buckets.clear()
