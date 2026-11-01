"""Verify the prepared ImageNet subset and real-image inference of ResNet-50.

Run with the dedicated environment, from the repository root, after building p3_inference:
    .venv-resnet\\Scripts\\python.exe scripts\\verify_resnet_inputs.py

Checks (nothing is downloaded; cached files are reused):
 1. Subset file: size, SHA-256, seed/selection reproduce, CSV matches manifest.
 2. Label mapping: the synset in each image's file name equals the synset the dataset's
    class list assigns to its label id; the class list is unique and sorted (the same
    order torchvision uses); names that differ from torchvision are listed.
 3. The stored tensors of the first 4 images equal the official V2 transform applied
    to the original JPEG bytes.
 4. The first 4 stored tensors give matching logits in PyTorch CPU, ONNX Runtime CPU and
    the C++ ONNX Runtime CUDA path (p3_inference model_check), for batch 1 and 4.
 5. During the C++ runs the GPU memory use rises, i.e. the model really runs on the GPU.

This is a connection/consistency check on 4 real images, not a performance or accuracy
experiment. Result: data/imagenet/real_image_verification.json (ignored by Git).
"""
import hashlib
import io
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'imagenet'
CACHE = DATA / 'hf_cache'
EXE = ROOT / 'build' / 'Release' / 'p3_inference.exe'
ONNX_PATH = ROOT / 'models' / 'resnet50' / 'resnet50_v2_fp32.onnx'
REPORT = DATA / 'real_image_verification.json'
RTOL, ATOL = 1e-3, 1e-4
CHECKED_IMAGES = 4
REPO_ID = 'ILSVRC/imagenet-1k'

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

torch.hub.set_dir(str(ROOT / 'models' / 'resnet50' / 'torch_cache'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cached(filename, revision):
    return Path(hf_hub_download(
        REPO_ID, filename, repo_type='dataset', revision=revision,
        cache_dir=str(CACHE), local_files_only=True))


def compare(name, actual, expected):
    """Shape, max abs diff, allclose and per-image top-1 of two logits arrays."""
    result = {'compared': name, 'shape': list(actual.shape),
              'shape_ok': actual.shape == expected.shape}
    if result['shape_ok']:
        result['max_abs_diff'] = float(np.max(np.abs(actual - expected)))
        result['allclose'] = bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL))
        result['top1_match'] = bool(np.array_equal(actual.argmax(1), expected.argmax(1)))
        result['top1'] = actual.argmax(1).tolist()
    result['ok'] = result['shape_ok'] and result.get('allclose', False) and result.get('top1_match', False)
    return result


def run_cpp(images_in_batch):
    """Run the C++ CUDA model_check; return (logits, gpu evidence)."""
    out_path = DATA / 'cpp_logits_b{}.bin'.format(images_in_batch)
    if out_path.exists():
        out_path.unlink()

    def gpu_state():
        text = subprocess.run(
            ['nvidia-smi', '-i', '0', '--query-gpu=memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, universal_newlines=True).stdout
        memory, utilization = [float(v) for v in text.strip().split(',')]
        return memory, utilization

    baseline_memory, _ = gpu_state()
    process = subprocess.Popen(
        [str(EXE), 'low', 'model_check', str(images_in_batch), str(out_path)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, encoding='utf-8', errors='replace')
    peak_memory, peak_util = baseline_memory, 0.0
    while process.poll() is None:
        memory, utilization = gpu_state()
        peak_memory, peak_util = max(peak_memory, memory), max(peak_util, utilization)
        time.sleep(0.05)
    log = process.stdout.read()
    if process.returncode != 0 or not out_path.exists():
        raise RuntimeError('C++ model_check failed (batch {}):\n{}'.format(images_in_batch, log))
    logits = np.fromfile(str(out_path), dtype='<f4').reshape(images_in_batch, 1000)
    return logits, {
        'batch': images_in_batch,
        'gpu_memory_before_mib': baseline_memory,
        'gpu_memory_peak_during_run_mib': peak_memory,
        'gpu_memory_rise_mib': peak_memory - baseline_memory,
        'gpu_util_peak_percent_sampled': peak_util,
        'cuda_provider_configured': 'CUDA Execution Provider configured' in log,
        'input_shape_verified': 'Input shape verified: [-1,3,224,224] float32' in log,
        'cpp_log': [line for line in log.splitlines() if 'onnxruntime:' not in line],
    }


def main():
    report = {'checks': {}, 'packages': {
        'torch': torch.__version__, 'onnxruntime': ort.__version__, 'numpy': np.__version__}}
    ok_all = True

    # 1. subset file and selection
    manifest = json.loads((DATA / 'val1000_manifest.json').read_text(encoding='utf-8'))
    binary = DATA / manifest['binary']['file']
    revision = manifest['dataset']['revision']
    selection = manifest['selection']
    rng_indices = [int(i) for i in np.random.default_rng(selection['seed'])
                   .choice(selection['population'], size=selection['count'], replace=False)]
    rows = [dict(zip(('position', 'global_index', 'shard', 'row_in_shard', 'label'), r[:5]))
            for r in (line.split(',') for line in
                      (DATA / 'val1000_selection.csv').read_text(encoding='utf-8').splitlines()[1:])]
    subset = {
        'size_bytes': binary.stat().st_size,
        'size_ok': binary.stat().st_size == 602112000 == manifest['binary']['total_bytes'],
        'shape': manifest['binary']['shape'], 'dtype': manifest['binary']['dtype'],
        'byte_order': manifest['binary']['byte_order'],
        'sha256_ok': sha256(binary) == manifest['binary']['sha256'],
        'revision': revision, 'seed': selection['seed'],
        'indices_distinct': len(set(selection['global_indices'])) == selection['count'],
        'indices_reproduce_from_seed': rng_indices == selection['global_indices'],
        'csv_matches_manifest': (
            [int(r['global_index']) for r in rows] == selection['global_indices']
            and [int(r['label']) for r in rows] == selection['labels']),
    }
    subset['ok'] = all(v for k, v in subset.items() if k.endswith('_ok') or k in (
        'indices_distinct', 'indices_reproduce_from_seed', 'csv_matches_manifest'))
    report['checks']['subset_file'] = subset

    # 2. explicit synset mapping
    classes_text = cached('classes.py', revision).read_text(encoding='utf-8')
    pairs = re.findall(r'"(n\d{8})":\s*"([^"]*)"', classes_text)
    wnids = [w for w, _ in pairs]
    names = [n for _, n in pairs]
    categories = ResNet50_Weights.IMAGENET1K_V2.meta['categories']
    bad = []
    selection_lines = (DATA / 'val1000_selection.csv').read_text(encoding='utf-8').splitlines()[1:]
    for line in selection_lines:
        position, global_index, shard, row_in_shard, label, rest = line.split(',', 5)
        file_name = rest.rsplit(',', 1)[-1].strip('"')
        match = re.search(r'_(n\d{8})\.JPEG$', file_name)
        if not match or wnids[int(label)] != match.group(1):
            bad.append({'position': int(position), 'label': int(label), 'file': file_name})
    name_differences = [
        {'label': i, 'synset': wnids[i], 'dataset_name': names[i], 'torchvision_name': categories[i]}
        for i in range(1000) if names[i].split(',')[0].strip() != categories[i]]
    labels_check = {
        'class_list_size': len(wnids), 'synsets_unique': len(set(wnids)) == 1000,
        'synsets_sorted': wnids == sorted(wnids),
        'torchvision_category_count': len(categories),
        'images_whose_file_synset_differs_from_label_synset': len(bad), 'examples': bad[:5],
        'name_only_differences_vs_torchvision': name_differences,
        'note': 'Both label ids and torchvision indices follow sorted synset order; every '
                'selected image carries its own synset in its file name and it matches.',
    }
    labels_check['ok'] = (labels_check['synsets_unique'] and labels_check['synsets_sorted']
                          and len(wnids) == len(categories) == 1000 and not bad)
    report['checks']['label_mapping'] = labels_check

    # 3. stored tensors vs official transform of the original JPEG bytes
    data = np.memmap(str(binary), dtype='<f4', mode='r', shape=(1000, 3, 224, 224))
    x = np.ascontiguousarray(data[:CHECKED_IMAGES])
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    preprocess = []
    for line in selection_lines[:CHECKED_IMAGES]:
        position, global_index, shard, row_in_shard, label, rest = line.split(',', 5)
        parquet = pq.ParquetFile(str(cached('data/' + shard, revision)))
        start = 0
        for group in range(parquet.metadata.num_row_groups):
            group_rows = parquet.metadata.row_group(group).num_rows
            if start <= int(row_in_shard) < start + group_rows:
                image = parquet.read_row_group(group, columns=['image']).column('image')[
                    int(row_in_shard) - start].as_py()
                break
            start += group_rows
        with Image.open(io.BytesIO(image['bytes'])) as pil:
            reference = transform(pil.convert('RGB')).numpy()
        preprocess.append({
            'position': int(position),
            'identical': bool(np.array_equal(reference, x[int(position)])),
            'max_abs_diff': float(np.max(np.abs(reference - x[int(position)])))})
    report['checks']['preprocessing_vs_official_v2'] = {
        'images': preprocess, 'ok': all(p['identical'] for p in preprocess)}

    # 4. logits: PyTorch CPU vs ORT CPU vs C++ CUDA
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    session = ort.InferenceSession(str(ONNX_PATH), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    inference = []
    gpu = []
    for n in (1, 4):
        batch = x[:n]
        with torch.no_grad():
            torch_logits = model(torch.from_numpy(batch)).numpy()
        ort_logits = session.run(None, {input_name: batch})[0]
        cpp_logits, evidence = run_cpp(n)
        gpu.append(evidence)
        inference.append({
            'batch': n,
            'labels': [int(r['label']) for r in rows[:n]],
            'pytorch_top1': torch_logits.argmax(1).tolist(),
            'comparisons': [
                compare('ort_cpu_vs_pytorch', ort_logits, torch_logits),
                compare('cpp_cuda_vs_pytorch', cpp_logits, torch_logits),
                compare('cpp_cuda_vs_ort_cpu', cpp_logits, ort_logits)]})
    report['checks']['logits'] = {
        'tolerance': {'rtol': RTOL, 'atol': ATOL},
        'results': inference,
        'ok': all(c['ok'] for r in inference for c in r['comparisons'])}

    # 5. the C++ runs really used the GPU
    report['checks']['cuda_execution'] = {
        'runs': gpu,
        'note': 'Evidence: CUDA EP configured without error (session creation throws '
                'otherwise) and GPU memory rises while the process runs. ORT may keep '
                'shape-related ops on the CPU; that does not move the convolutions.',
        'ok': all(g['cuda_provider_configured'] and g['gpu_memory_rise_mib'] > 100 for g in gpu)}

    ok_all = all(c['ok'] for c in report['checks'].values())
    report['status'] = 'PASSED' if ok_all else 'FAILED'
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')

    print('subset file     :', subset['ok'], '| size', subset['size_bytes'], '| sha256 ok', subset['sha256_ok'])
    print('label mapping   :', labels_check['ok'], '| synset mismatches', len(bad),
          '| name-only differences', [(d['label'], d['dataset_name'], d['torchvision_name'])
                                      for d in name_differences])
    print('preprocessing   :', report['checks']['preprocessing_vs_official_v2']['ok'])
    for r in inference:
        for c in r['comparisons']:
            print('batch {} {:<22} shape_ok={} max_abs_diff={:.3e} allclose={} top1_match={}'.format(
                r['batch'], c['compared'], c['shape_ok'], c.get('max_abs_diff', float('nan')),
                c.get('allclose'), c.get('top1_match')))
        print('   pytorch top-1 {} | dataset labels {}'.format(r['pytorch_top1'], r['labels']))
    for g in gpu:
        print('batch {} GPU memory {:.0f} -> {:.0f} MiB (+{:.0f}), sampled util peak {:.0f}%'.format(
            g['batch'], g['gpu_memory_before_mib'], g['gpu_memory_peak_during_run_mib'],
            g['gpu_memory_rise_mib'], g['gpu_util_peak_percent_sampled']))
    print('STATUS:', report['status'], '| report:', REPORT)
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
