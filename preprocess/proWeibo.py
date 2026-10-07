import argparse
import json
from pathlib import Path


def iter_sentences(path):
    tokens = []
    tags = []
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if tokens:
                    yield tokens, tags
                    tokens, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        yield tokens, tags


def bio_to_entities(tags):
    entities = []
    cur_type = None
    cur_start = None
    for i, tag in enumerate(tags):
        if tag == 'O':
            if cur_type is not None:
                entities.append((cur_start, i, cur_type))
                cur_type, cur_start = None, None
            continue

        if '-' not in tag:
            if cur_type is not None:
                entities.append((cur_start, i, cur_type))
                cur_type, cur_start = None, None
            continue

        prefix, ent_type = tag.split('-', 1)
        if prefix == 'B':
            if cur_type is not None:
                entities.append((cur_start, i, cur_type))
            cur_type = ent_type
            cur_start = i
        elif prefix == 'I':
            if cur_type == ent_type and cur_start is not None:
                continue
            if cur_type is not None:
                entities.append((cur_start, i, cur_type))
            cur_type = ent_type
            cur_start = i
        else:
            if cur_type is not None:
                entities.append((cur_start, i, cur_type))
            cur_type, cur_start = None, None

    if cur_type is not None:
        entities.append((cur_start, len(tags), cur_type))
    return entities


def convert_split(input_path, output_path, split_name):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_sent = 0
    num_ent = 0
    with output_path.open('w', encoding='utf-8') as w:
        for idx, (tokens, tags) in enumerate(iter_sentences(input_path)):
            entities = bio_to_entities(tags)
            entity_mentions = [
                {
                    'start': s,
                    'end': e,
                    'entity_type': t,
                    'text': ''.join(tokens[s:e]),
                }
                for s, e, t in entities
            ]
            sample = {
                'tokens': tokens,
                'doc_id': f'weibo-{split_name}-{idx}',
                'sent_id': f'weibo-{split_name}-{idx}',
                'entity_mentions': entity_mentions,
            }
            w.write(json.dumps(sample, ensure_ascii=False) + '\n')
            num_sent += 1
            num_ent += len(entity_mentions)
    return num_sent, num_ent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input_dir',
        default=str(Path(__file__).resolve().parents[1] / 'data' / 'weibo'),
        help='Directory containing train.txt/dev.txt/test.txt',
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        default=str(Path(__file__).resolve().parent / 'outputs' / 'weibo'),
        help='Directory to write train/dev/test jsonlines',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    split_map = {
        'train': ('train.txt', 'train.jsonlines'),
        'dev': ('dev.txt', 'dev.jsonlines'),
        'test': ('test.txt', 'test.jsonlines'),
    }

    for split, (src_name, tgt_name) in split_map.items():
        src = input_dir / src_name
        if not src.exists():
            raise FileNotFoundError(f'Missing split file: {src}')
        tgt = output_dir / tgt_name
        s_cnt, e_cnt = convert_split(src, tgt, split)
        print(f'{split}: sentences={s_cnt}, entities={e_cnt}, output={tgt}')


if __name__ == '__main__':
    main()
