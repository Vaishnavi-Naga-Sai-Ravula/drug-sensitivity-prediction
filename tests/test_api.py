import unittest
try:
    from fastapi.testclient import TestClient
    from api import app
except ImportError:
    app = None

@unittest.skipIf(app is None, 'Optional API dependencies are not installed')
class ApiTests(unittest.TestCase):
    def test_readiness_and_request_contracts(self):
        from pathdsp.inference import dataset_resource
        bundle = dataset_resource('GDSC')
        with TestClient(app) as client:
            self.assertEqual(client.get('/health/ready').status_code, 200)
            payload = {'dataset':'GDSC', 'pairs':[{'drug':bundle.drug_ids[0], 'cell':bundle.cell_ids[0]}]}
            response = client.post('/v1/predict', json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()['results']), 1)
            for invalid in ({'pairs':[]}, {'dataset':'bad','pairs':[{'drug':'x','cell':'y'}]}, {'pairs':[{'drug':'x','cell':'y'}]}, {'pairs':[{'drug':'x','cell':'y'}] * 257}):
                self.assertEqual(client.post('/v1/predict', json=invalid).status_code, 422)
