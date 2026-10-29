import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pathdsp.validation import read_feature_upload, validate_feature_frame, validate_metadata
from pathdsp.inference import model_resource, dataset_resource, predict_pairs, infer_features
from pathdsp.model import load_model_bundle, save_model_bundle

class BundledDataTests(unittest.TestCase):
    def test_untracked_processed_data_cannot_override_bundle(self):
        import zipfile
        from pathdsp import data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'processed_data').mkdir()
            (root / 'processed_data/table.txt').write_text('stale local data')
            archive = root / 'bundle.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('processed_data/table.txt', 'versioned bundled data')
            with patch.object(data, 'ROOT', root), patch.object(data, 'CORE_ARCHIVE', archive):
                with data._open_table('table.txt') as stream:
                    self.assertEqual(stream.read(), b'versioned bundled data')

class UploadTests(unittest.TestCase):
    def test_blank_imputation_is_distinct_from_bad_text(self):
        frame = pd.DataFrame([[1, 2, 3, 4, None]], columns=list('abcde'))
        self.assertTrue(np.isnan(validate_feature_frame(frame, list('abcde'))[0, 4]))
        frame = frame.astype(object)
        frame.iloc[0, 0] = 'not-a-number'
        with self.assertRaisesRegex(ValueError, 'nonnumeric'):
            validate_feature_frame(frame, list('abcde'))

    def test_malformed_batches_are_rejected(self):
        for payload in (b'', b'a,a\n1,2', b'a\ninf', b'a\ntext', b'wrong\n1', b'\xff'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                read_feature_upload(payload, ',', ['a'])
        with self.assertRaises(ValueError):
            validate_feature_frame(pd.DataFrame({'a': [None], 'b': [None]}), ['a', 'b'])
        with self.assertRaises(ValueError):
            validate_feature_frame(pd.DataFrame({'a': [1] * 2049}), ['a'])

class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metadata, _ = model_resource()

    def test_prediction_is_independent_of_observation_lookup(self):
        bundle = dataset_resource('GDSC')
        pair = [(bundle.drug_ids[0].upper(), bundle.cell_ids[0].upper())]
        original = predict_pairs('GDSC', pair)['results'][0]
        with patch.object(type(bundle), 'observed_response', return_value=99.0):
            changed = predict_pairs('GDSC', pair)['results'][0]
        self.assertEqual(original['predicted_neglog10_ic50'], changed['predicted_neglog10_ic50'])
        self.assertNotEqual(original['observed_neglog10_ic50'], changed['observed_neglog10_ic50'])

    def test_csv_feature_order_matches_catalogue_prediction(self):
        bundle = dataset_resource('GDSC')
        drug, cell = bundle.drug_ids[0], bundle.cell_ids[0]
        raw = bundle.pair_features(drug, cell)
        frame = pd.DataFrame([raw], columns=bundle.feature_names)
        csv = frame[frame.columns[::-1]].to_csv(index=False).encode()
        _, parsed = read_feature_upload(csv, ',', bundle.feature_names)
        prediction, _, _ = infer_features(parsed, self.model, self.metadata)
        catalogue = predict_pairs('GDSC', [(drug, cell)])['results'][0]
        self.assertAlmostEqual(float(prediction[0]), catalogue['predicted_neglog10_ic50'], places=5)

    def test_invalid_pair_and_batch_limits(self):
        for dataset, pairs in [('UNKNOWN', [('x','y')]), ('GDSC', []), ('GDSC', [('invalid','invalid')]), ('GDSC', [('x','y')] * 257)]:
            with self.subTest(dataset=dataset, size=len(pairs)), self.assertRaises(ValueError):
                predict_pairs(dataset, pairs)
        with self.assertRaises(ValueError):
            infer_features(np.full((1, len(self.metadata['feature_names'])), np.inf), self.model, self.metadata)

    def test_scaler_and_artifact_corruption_fail_closed(self):
        metadata = dict(self.metadata, scaler_scales=[0] * len(self.metadata['feature_names']))
        with self.assertRaisesRegex(ValueError, 'positive'):
            validate_metadata(metadata)
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ('model.pt', 'metadata.json', 'manifest.json'):
                shutil.copyfile(root / 'artifacts' / name, path / name)
            meta = json.loads((path / 'metadata.json').read_text())
            meta['scaler_means'][0] += 1
            (path / 'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_model_bundle(path)

    def test_saved_model_roundtrip_keeps_predictions_and_manifest(self):
        bundle = dataset_resource('GDSC')
        raw = bundle.pair_features(bundle.drug_ids[0], bundle.cell_ids[0]).reshape(1, -1)
        expected, _, _ = infer_features(raw, self.model, self.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_model_bundle(self.model, self.metadata['feature_names'], self.metadata['feature_groups'],
                              np.array(self.metadata['imputation_medians']), np.array(self.metadata['scaler_means']),
                              np.array(self.metadata['scaler_scales']), self.metadata['threshold_neglog10_ic50'], path)
            restored, metadata = load_model_bundle(path, bundle.feature_names)
            actual, _, _ = infer_features(raw, restored, metadata)
            np.testing.assert_allclose(actual, expected)
            (path / 'manifest.json').unlink()
            with self.assertRaisesRegex(ValueError, 'manifest is missing'):
                load_model_bundle(path)

    def test_windows_line_endings_do_not_break_artifact_integrity(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in ('model.pt', 'metadata.json', 'manifest.json'):
                shutil.copyfile(root / 'artifacts' / name, path / name)
            content=(path / 'metadata.json').read_bytes().replace(b'\r\n',b'\n')
            (path / 'metadata.json').write_bytes(content.replace(b'\n',b'\r\n'))
            _, metadata=load_model_bundle(path)
            self.assertEqual(metadata['feature_names'],self.metadata['feature_names'])

    def test_ccle_external_limitation_is_returned(self):
        bundle = dataset_resource('CCLE')
        response = predict_pairs('CCLE', [(bundle.drug_ids[0], bundle.cell_ids[0])])
        self.assertTrue(any('Poor cross-study' in warning for warning in response['warnings']))
        self.assertTrue(np.isfinite(response['results'][0]['predicted_neglog10_ic50']))
