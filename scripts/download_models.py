"""Download an ORT checkpoint and matching recipe, then verify SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    manifest=json.loads((Path(__file__).resolve().parents[1]/'configs/pretrained.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=list(manifest['models']), default='ort-e-alitok-xl-400')
    parser.add_argument('--output', default='weights')
    args=parser.parse_args()
    item=manifest['models'][args.model]
    if not item['standard_loader_supported']:
        parser.error('This archived model has an unsupported parameter layout; download it directly from the Hub for research.')
    checkpoint=hf_hub_download(repo_id=manifest['repo_id'],filename=item['file'],local_dir=args.output)
    config=hf_hub_download(repo_id=manifest['repo_id'],filename=item['config'],local_dir=args.output)
    digest=hashlib.sha256()
    with open(checkpoint,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=item['sha256']:
        raise RuntimeError('Checkpoint SHA-256 mismatch; do not use this file')
    print(f'Checkpoint verified: {checkpoint}\nConfig: {config}')

if __name__=='__main__':
    main()
