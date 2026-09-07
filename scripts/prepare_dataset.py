"""Validate pretokenized records and convert JSONL to a training dataset."""
import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk

SPECS = {'maskgit': (256, 1024), 'alitok': (273, 4096)}


def validate_record(record, tokenizer):
    length, vocab = SPECS[tokenizer]
    label, tokens = record['label'], record['tokens']
    if type(label) is not int or not 0 <= label < 1000:
        raise ValueError('label must be an integer in [0, 999]')
    if not isinstance(tokens, list) or not tokens:
        raise ValueError('tokens must be a nonempty list')
    crops = tokens if isinstance(tokens[0], list) else [tokens]
    for crop in crops:
        if not isinstance(crop, list) or len(crop) != length:
            raise ValueError(f'each crop must contain {length} tokens')
        if any(type(t) is not int or not 0 <= t < vocab for t in crop):
            raise ValueError(f'token IDs must be integers in [0, {vocab - 1}]')
    return {'label': label, 'tokens': crops}


def records(paths, tokenizer, label_key, tokens_key):
    for path in paths:
        with open(path) as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    yield validate_record({'label': item[label_key], 'tokens': item[tokens_key]}, tokenizer)
                except (ValueError, TypeError, KeyError) as error:
                    raise ValueError(f'{path}:{line_number}: {error}') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--jsonl', nargs='+', help='Input shards in the desired order')
    source.add_argument('--dataset', help='Validate an existing save_to_disk directory')
    parser.add_argument('--output', help='New dataset directory for JSONL conversion')
    parser.add_argument('--tokenizer', required=True, choices=SPECS)
    parser.add_argument('--label-key', default='label')
    parser.add_argument('--tokens-key', default='tokens')
    parser.add_argument('--expected-count', type=int)
    args = parser.parse_args()
    if args.jsonl:
        if not args.output:
            parser.error('--output is required with --jsonl')
        if Path(args.output).exists():
            parser.error('--output already exists; choose a new directory')
        ds = Dataset.from_generator(records, gen_kwargs={
            'paths': args.jsonl, 'tokenizer': args.tokenizer,
            'label_key': args.label_key, 'tokens_key': args.tokens_key,
        }, features=Features({'label': Value('int64'), 'tokens': Sequence(Sequence(Value('int64')))}))
    else:
        ds = load_from_disk(args.dataset)
        if isinstance(ds, DatasetDict):
            parser.error('Save the train split separately; trainer expects a Dataset, not DatasetDict')
    if args.expected_count is not None and len(ds) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} images, found {len(ds)}')
    if not len(ds):
        raise ValueError('Dataset is empty')
    counts = [0] * 1000
    variants = set()
    for i, item in enumerate(ds):
        try:
            row = validate_record(item, args.tokenizer)
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'Row {i}: {error}') from error
        counts[row['label']] += 1
        variants.add(len(row['tokens']))
    if args.jsonl:
        ds.save_to_disk(args.output)
    print(json.dumps({'images': len(ds), 'classes_present': sum(n > 0 for n in counts),
                      'crop_counts': sorted(variants), 'tokenizer': args.tokenizer}))


if __name__ == '__main__':
    main()
