"""Explicit runtime paths shared by the launcher and the current RAW pipeline.

No SDK, CUDA or project inference code is imported here.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parent
REFERENCE_SHA256 = 'ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4'


def executable(value):
    value = os.fspath(value)
    found = shutil.which(value)
    path = Path(found or value).expanduser().resolve()
    if not path.is_file():
        raise ValueError('Python executable not found: ' + value)
    return path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(gpu_python=None, detector_weights=None, detector_sha256=None, cache_dir=None):
    # Optional private developer configuration preserves the existing machine's
    # working setup. Public source bundles exclude this file and all weights.
    local = ROOT / 'configs/local.json'
    cfg = json.loads(local.read_text(encoding='utf-8')) if local.is_file() else {}
    def choose(argument, env, key, default):
        return argument if argument is not None else os.environ.get(env, (cfg.get(key) if cfg.get(key) is not None else default))
    infer = executable(choose(gpu_python, 'EDF_GPU_PYTHON', 'gpu_python', sys.executable))
    weights = Path(choose(detector_weights, 'EDF_DETECTOR_WEIGHTS', 'detector_weights', ROOT / 'weights/best.onnx')).expanduser()
    cache = Path(choose(cache_dir, 'EDF_CACHE_DIR', 'cache_dir', ROOT / '.cache')).expanduser()
    # Relative private configuration values are relative to configs/, as in the CLI.
    for key, value in [('detector_weights', weights), ('cache_dir', cache)]:
        if not value.is_absolute():
            arg = detector_weights if key == 'detector_weights' else cache_dir
            env = 'EDF_DETECTOR_WEIGHTS' if key == 'detector_weights' else 'EDF_CACHE_DIR'
            from_local = arg is None and env not in os.environ and key in cfg
            value = local.parent / value if from_local else Path.cwd() / value
        if key == 'detector_weights': weights = value.resolve()
        else: cache = value.resolve()
    expected = str(choose(detector_sha256, 'EDF_DETECTOR_SHA256', 'detector_sha256', REFERENCE_SHA256)).lower()
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('detector_sha256 must be a 64-character SHA256 digest')
    if not weights.is_file():
        raise ValueError('Detector weights not found: ' + str(weights))
    actual = sha256(weights)
    if actual != expected:
        raise ValueError('Detector SHA256 mismatch: expected ' + expected + ', found ' + actual)
    return dict(gpu_python=str(infer), detector_weights=str(weights),
                detector_sha256=expected, cache_dir=str(cache))
