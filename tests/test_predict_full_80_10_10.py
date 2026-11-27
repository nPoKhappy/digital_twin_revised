import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts.prediction import predict_full_80_10_10 as predictor


def digest(data):
    return hashlib.sha256(data).hexdigest()


class PredictionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'configs/pgin' / predictor.EXPERIMENT / 'config_manifest.json'
        self.result_root = self.root / 'results/pgin' / predictor.EXPERIMENT
        self.name = predictor.PREFIX + 'gain25_seed42'
        self.config = self.manifest.parent / 'seed42/gain25.yaml'
        self.result = self.result_root / 'seed42' / self.name
        self.checkpoint = self.root / 'saved_models' / (self.name + '.pth')
        self.config.parent.mkdir(parents=True)
        self.result.mkdir(parents=True)
        self.checkpoint.parent.mkdir(parents=True)
        cfg = {'exp_name': self.name, 'training': {'seed': 42},
               'output': {'results_dir': './results/pgin/' + predictor.EXPERIMENT + '/seed42'}}
        self.config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
        job = {'seed': 42, 'alias': 'gain25', 'exp_name': self.name,
               'config': self.config.relative_to(self.root).as_posix(),
               'config_sha256': digest(self.config.read_bytes()),
               'checkpoint': self.checkpoint.relative_to(self.root).as_posix(),
               'results': self.result.relative_to(self.root).as_posix()}
        self.manifest.write_text(json.dumps([job]), encoding='utf-8')
        files = [self.checkpoint, self.result / 'zscore_mean.csv', self.result / 'zscore_std.csv']
        for path in files:
            path.write_bytes(path.name.encode('utf-8'))
        outputs = {str(path.relative_to(self.root)): digest(path.read_bytes()) for path in files}
        sources = {str(path.resolve()): digest(path.read_bytes()) for path in (self.manifest, self.config)}
        record = {'seed': 42, 'status': 'complete', 'sources_sha256': sources,
                  'completed': [{'job': job, 'outputs_sha256': outputs}]}
        (self.result_root / 'seed42/_batch_run.json').write_text(json.dumps(record), encoding='utf-8')
        patcher = patch.multiple(predictor, ROOT=self.root, MANIFEST=self.manifest, RESULT_ROOT=self.result_root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_accepts_verified_training_artifacts(self):
        self.assertEqual(predictor.resolve_completed(42, 'gain25'), self.config)

    def test_missing_or_changed_scaler_fails_closed(self):
        scaler = self.result / 'zscore_mean.csv'
        scaler.unlink()
        with self.assertRaisesRegex(ValueError, 'scaler missing or changed'):
            predictor.resolve_completed(42, 'gain25')
        scaler.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'scaler missing or changed'):
            predictor.resolve_completed(42, 'gain25')

    def test_changed_checkpoint_is_rejected(self):
        self.checkpoint.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checkpoint or training scaler'):
            predictor.resolve_completed(42, 'gain25')

    def test_changed_config_and_unknown_alias_are_rejected(self):
        self.config.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Configuration missing or differs'):
            predictor.resolve_completed(42, 'gain25')
        with self.assertRaisesRegex(ValueError, 'Expected one manifest job'):
            predictor.resolve_completed(42, 'unknown')


if __name__ == '__main__':
    unittest.main()
