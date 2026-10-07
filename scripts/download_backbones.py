#!/usr/bin/env python3
import argparse
import os
import shutil
import sys


ALLOW_FILE_PATTERNS = [
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.txt',
    'vocab.json',
    'merges.txt',
    'pytorch_model.bin',
    'model.safetensors',
    'added_tokens.json',
    'sentencepiece.bpe.model',
]


def _script_root():
    return os.path.dirname(os.path.abspath(__file__))


def _project_root():
    return os.path.dirname(_script_root())


def _target_name_from_model_id(model_id):
    lower_id = model_id.lower()
    if 'roberta-base' in lower_id:
        return 'roberta-base'
    if 'bert-large-cased' in lower_id:
        return 'bert-large-cased'
    if 'biobert' in lower_id:
        return 'biobert-v1.1'
    if 'biomedbert-base-uncased-abstract-fulltext' in lower_id:
        return 'biomedbert-base-uncased-abstract-fulltext'
    if 'bert-base-cased' in lower_id:
        return 'bert-base-cased'
    return lower_id.split('/')[-1]


def _replace_with_symlink(src_dir, dst_dir):
    if os.path.lexists(dst_dir):
        if os.path.islink(dst_dir) or os.path.isfile(dst_dir):
            os.unlink(dst_dir)
        else:
            shutil.rmtree(dst_dir)
    os.symlink(src_dir, dst_dir)


def _download_with_modelscope(model_id, cache_dir):
    from modelscope.hub.snapshot_download import snapshot_download

    return snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_file_pattern=ALLOW_FILE_PATTERNS,
    )


def _download_with_wisemodel(model_id, cache_dir, endpoint):
    os.environ['HF_ENDPOINT'] = endpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=ALLOW_FILE_PATTERNS,
        resume_download=True,
    )


def _build_download_plan(source, selected_groups):
    if source == 'modelscope':
        candidates = {
            'ace': [
                'AI-ModelScope/roberta-base',
            ],
            'genia': [
                'dmis-lab/biobert-v1.1',
                'AI-ModelScope/biobert-v1.1',
                'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                'AI-ModelScope/bert-base-cased',
            ],
            'conll': [
                'AI-ModelScope/bert-large-cased',
                'AI-ModelScope/bert-base-cased',
            ],
        }
    elif source == 'wisemodel':
        candidates = {
            'ace': [
                'roberta-base',
            ],
            'genia': [
                'dmis-lab/biobert-v1.1',
                'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                'bert-base-cased',
            ],
            'conll': [
                'bert-large-cased',
                'bert-base-cased',
            ],
        }
    else:
        raise ValueError(f'Unsupported source: {source}')

    return {group: candidates[group] for group in selected_groups}


def _download_one_group(group_name, model_ids, source, cache_dir, output_root, wisemodel_endpoint):
    downloader = _download_with_modelscope
    if source == 'wisemodel':
        downloader = lambda model_id, _cache_dir: _download_with_wisemodel(
            model_id=model_id,
            cache_dir=_cache_dir,
            endpoint=wisemodel_endpoint,
        )

    last_error = None
    for model_id in model_ids:
        try:
            print(f'[INFO] downloading {group_name} backbone from {source}: {model_id}', flush=True)
            downloaded_dir = downloader(model_id, cache_dir)
            target_name = _target_name_from_model_id(model_id)
            target_dir = os.path.join(output_root, target_name)
            _replace_with_symlink(os.path.abspath(downloaded_dir), target_dir)
            print(f'[OK] {group_name}: {model_id} -> {target_dir}', flush=True)
            return target_name, model_id
        except Exception as exc:
            last_error = exc
            print(f'[WARN] failed: {model_id} ({type(exc).__name__}: {exc})', flush=True)

    raise RuntimeError(
        f'All candidates failed for group={group_name}, source={source}. last_error={last_error}'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Download DiFiNet backbones without direct HuggingFace access.'
    )
    parser.add_argument(
        '--source',
        default='modelscope',
        choices=['modelscope', 'wisemodel'],
        help='Download source. Recommend modelscope when HuggingFace is unreachable.',
    )
    parser.add_argument(
        '--groups',
        default='ace,genia,conll',
        help='Comma-separated groups in {ace,genia,conll}.',
    )
    parser.add_argument(
        '--cache-dir',
        default=os.path.join(os.path.expanduser('~'), '.cache', 'difinet_backbones'),
    )
    parser.add_argument(
        '--output-root',
        default=os.path.join(_project_root(), 'pretrained_models'),
    )
    parser.add_argument(
        '--wisemodel-endpoint',
        default=os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com'),
        help='HF-compatible endpoint used when --source wisemodel. Example: https://hf-mirror.com',
    )
    args = parser.parse_args()

    groups = [x.strip() for x in args.groups.split(',') if x.strip()]
    valid_groups = {'ace', 'genia', 'conll'}
    invalid_groups = sorted(set(groups) - valid_groups)
    if invalid_groups:
        raise ValueError(f'Invalid groups: {invalid_groups}, expected subset of {sorted(valid_groups)}')

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_root, exist_ok=True)

    plan = _build_download_plan(args.source, groups)
    summary = []
    for group_name, model_ids in plan.items():
        target_name, used_model_id = _download_one_group(
            group_name=group_name,
            model_ids=model_ids,
            source=args.source,
            cache_dir=args.cache_dir,
            output_root=args.output_root,
            wisemodel_endpoint=args.wisemodel_endpoint,
        )
        summary.append((group_name, target_name, used_model_id))

    print('\n[SUMMARY]')
    for group_name, target_name, used_model_id in summary:
        print(f'- {group_name}: {target_name} (from {used_model_id})')
    print('\nDone.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(1)
