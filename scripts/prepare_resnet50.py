"""Export torchvision ResNet-50 (IMAGENET1K_V2, FP32) to ONNX and check it.

Run with the dedicated environment, from anywhere:
    .venv-resnet\\Scripts\\python.exe scripts\\prepare_resnet50.py

Outputs go to models/resnet50/ (ignored by Git):
    resnet50_v2_fp32.onnx   dynamic batch axis, input [N,3,224,224], output [N,1000]
    resnet50_v2_fp32.onnx.data  weights stored as external data (keep beside the .onnx)
    export_report.json      what was exported and how, plus the check results
    pip_freeze.txt          installed package versions

The check uses seeded SYNTHETIC random inputs. It only verifies that the ONNX
export matches PyTorch; it says nothing about ImageNet accuracy and its output
must not be mixed with the formal performance experiments.
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'models' / 'resnet50'
ONNX_PATH = OUT_DIR / 'resnet50_v2_fp32.onnx'
DATA_PATH = OUT_DIR / 'resnet50_v2_fp32.onnx.data'  # weights; both files must stay together
REPORT_PATH = OUT_DIR / 'export_report.json'
FREEZE_PATH = OUT_DIR / 'pip_freeze.txt'

# The dynamo exporter implements opset 18 or later; 18 is supported by ONNX Runtime 1.26.0.
OPSET = 18
SEED = 0
RTOL = 1e-3
ATOL = 1e-4
CHECK_BATCH_SIZES = (1, 4)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# The exporter prints non-ASCII symbols, which a cp949 Windows console cannot encode.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from torchvision.models import ResNet50_Weights, resnet50

# Keep the downloaded official weights inside the ignored output folder.
torch.hub.set_dir(str(OUT_DIR / 'torch_cache'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.eval()

    transforms = weights.transforms()
    preprocessing = {
        'resize_size': transforms.resize_size[0],
        'crop_size': transforms.crop_size[0],
        'interpolation': str(transforms.interpolation),
        'mean': list(transforms.mean),
        'std': list(transforms.std),
        'color_order': 'RGB',
        'layout': 'NCHW',
        'scale': 'uint8 [0,255] -> float [0,1] before normalization',
    }

    # Example batch is 2, not 1, so the exporter does not specialize batch size 1.
    example = torch.randn(
        2, 3, 224, 224, generator=torch.Generator().manual_seed(SEED)
    )
    batch = torch.export.Dim('batch', min=1, max=64)

    torch.onnx.export(
        model,
        (example,),
        str(ONNX_PATH),
        input_names=['input'],
        output_names=['logits'],
        dynamic_shapes=({0: batch},),
        opset_version=OPSET,
        dynamo=True,
    )

    onnx.checker.check_model(str(ONNX_PATH))
    onnx_model = onnx.load(str(ONNX_PATH), load_external_data=False)
    input_dim0 = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    output_dim0 = onnx_model.graph.output[0].type.tensor_type.shape.dim[0]
    batch_axis_dynamic = bool(input_dim0.dim_param) and not input_dim0.HasField('dim_value')
    external_data = any(
        t.data_location == onnx.TensorProto.EXTERNAL
        for t in onnx_model.graph.initializer
    )

    session = ort.InferenceSession(
        str(ONNX_PATH), providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Same seeded FP32 inputs for both runtimes. Batch 1 is the first row of batch 4.
    inputs = torch.randn(
        max(CHECK_BATCH_SIZES), 3, 224, 224,
        generator=torch.Generator().manual_seed(SEED + 1),
    )

    checks = []
    for size in CHECK_BATCH_SIZES:
        x = inputs[:size].contiguous()
        with torch.no_grad():
            expected = model(x).numpy()
        actual = session.run([output_name], {input_name: x.numpy()})[0]

        shape_ok = tuple(actual.shape) == (size, 1000)
        finite_ok = bool(np.isfinite(actual).all())
        if shape_ok:
            max_abs_diff = float(np.max(np.abs(actual - expected)))
            allclose = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))
            top1_match = bool(
                np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
            )
        else:
            max_abs_diff, allclose, top1_match = None, False, False

        checks.append({
            'batch_size': size,
            'output_shape': list(actual.shape),
            'output_shape_ok': shape_ok,
            'finite_ok': finite_ok,
            'max_abs_diff': max_abs_diff,
            'allclose_rtol_1e-3_atol_1e-4': allclose,
            'top1_match': top1_match,
            'pytorch_top1': expected.argmax(axis=1).tolist(),
            'onnx_top1': actual.argmax(axis=1).tolist() if shape_ok else None,
        })

    passed = (
        batch_axis_dynamic
        and all(
            c['output_shape_ok'] and c['finite_ok']
            and c['allclose_rtol_1e-3_atol_1e-4'] and c['top1_match']
            for c in checks
        )
    )

    freeze = subprocess.run(
        [sys.executable, '-m', 'pip', 'freeze'],
        stdout=subprocess.PIPE, universal_newlines=True, check=True,
    ).stdout
    FREEZE_PATH.write_text(freeze, encoding='utf-8')

    report = {
        'status': 'PASSED' if passed else 'FAILED',
        'created': datetime.now().isoformat(timespec='seconds'),
        'weights': 'torchvision ResNet50_Weights.IMAGENET1K_V2',
        'weights_meta': {
            'acc@1': weights.meta['_metrics']['ImageNet-1K']['acc@1'],
            'acc@5': weights.meta['_metrics']['ImageNet-1K']['acc@5'],
            'url': weights.url,
        },
        'preprocessing': preprocessing,
        'precision': 'FP32',
        'export': {
            'exporter': 'torch.onnx.export(dynamo=True)',
            'opset': OPSET,
            'input': {'name': input_name, 'shape': ['batch', 3, 224, 224]},
            'output': {'name': output_name, 'shape': ['batch', 1000]},
            'dynamic_batch_range': [1, 64],
            'onnx_input_dim0': input_dim0.dim_param or int(input_dim0.dim_value),
            'onnx_output_dim0': output_dim0.dim_param or int(output_dim0.dim_value),
            'batch_axis_dynamic': batch_axis_dynamic,
            'external_data_used': external_data,
            'onnx_file': ONNX_PATH.name,
            'onnx_size_bytes': ONNX_PATH.stat().st_size,
            'onnx_sha256': sha256(ONNX_PATH),
            'weights_data_file': DATA_PATH.name if external_data else None,
            'weights_data_size_bytes': DATA_PATH.stat().st_size if external_data else None,
            'weights_data_sha256': sha256(DATA_PATH) if external_data else None,
        },
        'verification': {
            'kind': 'synthetic random FP32 input (seeded); conversion check only, '
                    'NOT ImageNet data and NOT a model accuracy metric',
            'seed': SEED + 1,
            'runtime_compared': 'PyTorch CPU vs ONNX Runtime CPUExecutionProvider',
            'tolerance': {'rtol': RTOL, 'atol': ATOL},
            'checks': checks,
        },
        'packages': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torchvision': torchvision.__version__,
            'onnx': onnx.__version__,
            'onnxruntime': ort.__version__,
            'numpy': np.__version__,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('ONNX:', ONNX_PATH, '({:.1f} MB)'.format(ONNX_PATH.stat().st_size / 1e6))
    print('Input dim 0:', report['export']['onnx_input_dim0'],
          '| dynamic:', batch_axis_dynamic, '| external data:', external_data)
    for c in checks:
        print('batch {batch_size}: shape {output_shape} max_abs_diff={max_abs_diff} '
              'allclose={allclose} top1_match={top1_match}'.format(
                  batch_size=c['batch_size'], output_shape=c['output_shape'],
                  max_abs_diff=c['max_abs_diff'],
                  allclose=c['allclose_rtol_1e-3_atol_1e-4'],
                  top1_match=c['top1_match']))
    print('Report:', REPORT_PATH)
    print('STATUS:', report['status'])
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
