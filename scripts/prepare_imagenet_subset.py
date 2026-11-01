"""Prepare a fixed 1,000-image ImageNet validation subset for ResNet-50 serving.

Run with the dedicated environment, from the repository root:
    .venv-resnet\\Scripts\\python.exe scripts\\prepare_imagenet_subset.py --plan
    .venv-resnet\\Scripts\\python.exe scripts\\prepare_imagenet_subset.py

Data source: Hugging Face dataset ILSVRC/imagenet-1k (gated; you must have accepted
its terms and logged in with `hf auth login`). Only the 14 validation parquet files
(about 6.7 GB in total) are downloaded; train and test files are never requested.

Selection: seed 42, 1,000 distinct indices out of the 50,000 validation images, in the
order they were drawn. The images and any model output play no part in the selection.

Output (data/imagenet/, ignored by Git):
    val1000_resnet50_v2_nchw_f32.bin   raw float32, little-endian, C order, [1000,3,224,224]
    val1000_selection.csv              one row per image, in the same order as the .bin
    val1000_manifest.json              how everything was produced and verified
"""
import argparse
import hashlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'data' / 'imagenet'
CACHE_DIR = OUT_DIR / 'hf_cache'
BIN_PATH = OUT_DIR / 'val1000_resnet50_v2_nchw_f32.bin'
SELECTION_PATH = OUT_DIR / 'val1000_selection.csv'
MANIFEST_PATH = OUT_DIR / 'val1000_manifest.json'

REPO_ID = 'ILSVRC/imagenet-1k'
# Pinned dataset revision (commit) seen on 2026-11-01, so later edits cannot change the data.
REVISION = '49e2ee26f3810fb5a7536bbf732a7b07389a47b5'
VALIDATION_IMAGES = 50000
SEED = 42
SUBSET_SIZE = 1000
IMAGE_SHAPE = (3, 224, 224)
BYTES_PER_IMAGE = 3 * 224 * 224 * 4
MIN_SANITY_TOP1 = 0.50  # a label-order mistake would give about 0.1%; this only catches that

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pyarrow
import pyarrow.parquet as pq
from huggingface_hub import HfApi, get_token, hf_hub_download
from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def choose_indices():
    rng = np.random.default_rng(SEED)
    return [int(i) for i in rng.choice(VALIDATION_IMAGES, size=SUBSET_SIZE, replace=False)]


def list_validation_files():
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    files = sorted(
        (s.rfilename, s.size) for s in info.siblings
        if s.rfilename.startswith('data/validation-')
    )
    return info.sha, files


def print_plan(indices, files):
    total = sum(size for _, size in files)
    print('Dataset:', REPO_ID, '| revision:', REVISION)
    print('Validation files to download: {} ({:.2f} GB). train/test files: none.'.format(
        len(files), total / 1e9))
    print('Selected {} distinct indices out of {} (seed {}). First 10: {}'.format(
        len(indices), VALIDATION_IMAGES, SEED, indices[:10]))
    print('Output: {} bytes ({:.1f} MB) per file {}'.format(
        SUBSET_SIZE * BYTES_PER_IMAGE, SUBSET_SIZE * BYTES_PER_IMAGE / 1e6, BIN_PATH.name))
    print('Download cache:', CACHE_DIR)


def read_class_names(parquet_file):
    """Class names stored by the datasets library in the parquet schema, if present."""
    metadata = parquet_file.schema_arrow.metadata or {}
    try:
        features = json.loads(metadata[b'huggingface'])['info']['features']
        return features['label']['names']
    except (KeyError, ValueError):
        return None


def process_shard(parquet_path, offset, wanted, transform, output):
    """Preprocess the wanted rows of one shard into the output array.

    wanted maps global index -> output position. Returns (rows_in_shard, records, names).
    """
    parquet_file = pq.ParquetFile(str(parquet_path))
    rows = parquet_file.metadata.num_rows
    names = read_class_names(parquet_file)
    records = []

    group_start = offset
    for group in range(parquet_file.metadata.num_row_groups):
        group_rows = parquet_file.metadata.row_group(group).num_rows
        hits = [g for g in wanted if group_start <= g < group_start + group_rows]
        if hits:
            table = parquet_file.read_row_group(group, columns=['image', 'label'])
            images = table.column('image')
            labels = table.column('label')
            for global_index in hits:
                local = global_index - group_start
                image = images[local].as_py()
                with Image.open(io.BytesIO(image['bytes'])) as pil:
                    tensor = transform(pil.convert('RGB'))
                position = wanted[global_index]
                output[position] = tensor.numpy()
                records.append({
                    'position': position,
                    'global_index': global_index,
                    'shard': parquet_path.name,
                    'row_in_shard': global_index - offset,
                    'label': int(labels[local].as_py()),
                    'image_path': image.get('path'),
                })
        group_start += group_rows

    return rows, records, names


def check_with_model(mmap_array, labels):
    """Top-1 of the torchvision model on the saved file: a label-order sanity check only."""
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    torch.hub.set_dir(str(ROOT / 'models' / 'resnet50' / 'torch_cache'))
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(labels), 50):
            batch = torch.from_numpy(np.ascontiguousarray(mmap_array[start:start + 50]))
            predicted = model(batch).argmax(dim=1).tolist()
            correct += sum(int(p == t) for p, t in zip(predicted, labels[start:start + 50]))
    return correct / len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--plan', action='store_true',
                        help='print the selection and download size, then stop (no download)')
    args = parser.parse_args()

    indices = choose_indices()
    revision, files = list_validation_files()
    if revision != REVISION or len(files) != 14:
        raise RuntimeError('Unexpected dataset revision or validation file count')
    print_plan(indices, files)
    if args.plan:
        return 0

    if BIN_PATH.exists():
        raise FileExistsError('Output exists; move it first: ' + str(BIN_PATH))
    if get_token() is None:
        print('\nNo Hugging Face login found. In your own terminal run (token is stored locally,\n'
              'never in this repository):\n'
              '    .venv-resnet\\Scripts\\hf.exe auth login\n'
              'You must also have accepted the dataset terms at '
              'https://huggingface.co/datasets/ILSVRC/imagenet-1k', file=sys.stderr)
        return 2

    from torchvision.models import ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    transform = weights.transforms()  # official: resize 232, center crop 224, [0,1], mean/std

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partial = BIN_PATH.with_name(BIN_PATH.name + '.partial')
    output = np.memmap(str(partial), dtype='<f4', mode='w+',
                       shape=(SUBSET_SIZE,) + IMAGE_SHAPE)

    wanted = {g: position for position, g in enumerate(indices)}
    records = []
    class_names = None
    offset = 0
    for name, size in files:
        print('Downloading {} ({:.0f} MB)'.format(name, size / 1e6), flush=True)
        try:
            path = Path(hf_hub_download(
                REPO_ID, name, repo_type='dataset', revision=REVISION,
                cache_dir=str(CACHE_DIR)))
        except Exception as error:
            print('Download failed: {}: {}\nCheck that you accepted the dataset terms and '
                  'are logged in.'.format(type(error).__name__, error), file=sys.stderr)
            return 3
        rows, shard_records, names = process_shard(path, offset, wanted, transform, output)
        class_names = class_names or names
        records.extend(shard_records)
        offset += rows
        print('  rows: {} | selected here: {}'.format(rows, len(shard_records)), flush=True)

    if offset != VALIDATION_IMAGES:
        raise RuntimeError('Expected {} validation rows, found {}'.format(VALIDATION_IMAGES, offset))
    if len(records) != SUBSET_SIZE:
        raise RuntimeError('Selected {} images, found {}'.format(SUBSET_SIZE, len(records)))
    records.sort(key=lambda r: r['position'])

    output.flush()
    labels = [r['label'] for r in records]
    top1 = check_with_model(output, labels)
    del output
    if top1 < MIN_SANITY_TOP1:
        raise RuntimeError('Top-1 {:.1%} is too low: labels do not match the model classes'.format(top1))
    partial.replace(BIN_PATH)

    # Class order: Hugging Face label ids against torchvision's category names.
    categories = weights.meta['categories']
    mismatches = None
    if class_names is not None:
        mismatches = [i for i in range(1000)
                      if class_names[i].split(',')[0].strip() != categories[i]]
        for r in records:
            r['class_name'] = class_names[r['label']]

    with SELECTION_PATH.open('w', encoding='utf-8', newline='') as f:
        f.write('position,global_index,shard,row_in_shard,label,class_name,image_path\n')
        for r in records:
            f.write('{},{},{},{},{},"{}","{}"\n'.format(
                r['position'], r['global_index'], r['shard'], r['row_in_shard'],
                r['label'], r.get('class_name', ''), r['image_path'] or ''))

    manifest = {
        'created': datetime.now().isoformat(timespec='seconds'),
        'dataset': {'repo': REPO_ID, 'revision': REVISION, 'split': 'validation',
                    'validation_rows': VALIDATION_IMAGES,
                    'files': [name for name, _ in files],
                    'downloaded_bytes': sum(size for _, size in files)},
        'selection': {
            'seed': SEED, 'count': SUBSET_SIZE, 'population': VALIDATION_IMAGES,
            'method': 'numpy.random.default_rng(42).choice(50000, 1000, replace=False); '
                      'index = row position in the validation split (parquet files in '
                      'name order); stored in drawn order; images and model output not used',
            'numpy': np.__version__,
            'global_indices': indices,
            'labels': labels,
            'details_csv': SELECTION_PATH.name,
        },
        'preprocessing': {
            'weights': 'torchvision ResNet50_Weights.IMAGENET1K_V2',
            'transform': 'weights.transforms() applied to PIL RGB (resize 232 bilinear, '
                         'center crop 224, scale to [0,1], normalize mean/std)',
            'repr': str(transform),
        },
        'binary': {
            'file': BIN_PATH.name,
            'dtype': 'float32', 'byte_order': 'little-endian',
            'shape': [SUBSET_SIZE] + list(IMAGE_SHAPE), 'layout': 'N,C,H,W (RGB), C-contiguous',
            'order': 'image i occupies bytes [i*{0}, (i+1)*{0}); i matches "position" in the CSV'
                     .format(BYTES_PER_IMAGE),
            'bytes_per_image': BYTES_PER_IMAGE,
            'floats_per_image': 3 * 224 * 224,
            'total_bytes': BIN_PATH.stat().st_size,
            'header': 'none',
            'sha256': sha256(BIN_PATH),
        },
        'label_verification': {
            'label_source': 'ILSVRC/imagenet-1k "label" column (0-999)',
            'class_names_in_parquet_schema': class_names is not None,
            'name_mismatches_vs_torchvision_categories': (
                None if mismatches is None else len(mismatches)),
            'mismatching_label_ids': mismatches,
            'sanity_top1_pytorch_cpu': top1,
            'sanity_note': 'torchvision V2 top-1 on these 1,000 images, computed AFTER '
                           'selection from the saved file; only detects a label-order error '
                           'and is not a model accuracy metric (n=1000, about +/-2.5 points)',
        },
        'packages': {'numpy': np.__version__, 'pyarrow': pyarrow.__version__},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    print('Binary:', BIN_PATH, '({} bytes)'.format(BIN_PATH.stat().st_size))
    print('Name mismatches vs torchvision categories:',
          'n/a' if mismatches is None else len(mismatches))
    print('Sanity top-1 on the 1000 images: {:.1%}'.format(top1))
    print('Manifest:', MANIFEST_PATH)
    return 0


if __name__ == '__main__':
    sys.exit(main())
