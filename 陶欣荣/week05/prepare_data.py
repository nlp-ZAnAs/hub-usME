"""准备预训练文本和公开中文短问答 SFT 数据。"""
import argparse
from collections import Counter
import hashlib
import json
import random
import unicodedata
import urllib.request
from pathlib import Path
from model import ROOT, INITIAL_CHECKPOINT, initialize_from_previous

SOURCE_URL = 'https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM'


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                 for row in records), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_len', type=int, default=256)
    args = parser.parse_args()
    data = ROOT / 'data'
    _, tokenizer, _ = initialize_from_previous(INITIAL_CHECKPOINT, args.seq_len)
    lines = (ROOT.parent / 'week04/corpus_general.txt').read_text(encoding='utf-8').splitlines()
    unique = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
    random.Random(args.seed).shuffle(unique)
    split = int(len(unique) * .95)
    for name, records in [('pretrain_train', unique[:split]), ('pretrain_val', unique[split:])]:
        (data / f'{name}.txt').write_text('\n'.join(records), encoding='utf-8')
    source = data / 'alpaca_gpt4_data_zh.json'
    if not source.exists():
        url = "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data_zh.json"
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        expected = "6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903"
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("下载的数据版本与本次实验不同，请核对来源")
        source.write_bytes(content)
    records = json.loads(source.read_text(encoding='utf-8'))
    selected, seen, rejected = [], set(), Counter()
    for index, row in enumerate(records):
        instruction = row.get('instruction', '').strip()
        extra = row.get('input', '').strip()
        answer = row.get('output', '').strip()
        question = instruction + ('\n' + extra if extra else '')
        key = ''.join(unicodedata.normalize('NFKC', question).split()).casefold()
        if not instruction or not answer or key in seen:
            rejected['empty_or_duplicate'] += 1
            continue
        seen.add(key)
        if len(question) > 100 or len(answer) > 180 or len(question) + len(answer) + 2 > args.seq_len:
            rejected['too_long'] += 1
            continue
        if not any('\u4e00' <= char <= '\u9fff' for char in question):
            rejected['non_chinese_prompt'] += 1
            continue
        ids = tokenizer.encode(question + answer)
        if ids.count(tokenizer.unk) / len(ids) > .02:
            rejected['too_many_unknown_characters'] += 1
            continue
        selected.append(dict(question=question, answer=answer, source_index=index))
    random.Random(args.seed).shuffle(selected)
    selected = selected[:args.samples]
    if len(selected) < 20:
        raise ValueError('有效问答太少')
    train_end, val_end = int(len(selected) * .9), int(len(selected) * .95)
    splits = {'train': selected[:train_end], 'val': selected[train_end:val_end], 'test': selected[val_end:]}
    for name, rows in splits.items():
        write_jsonl(data / f'sft_{name}.jsonl', rows)
    files = ['pretrain_train.txt', 'pretrain_val.txt', 'sft_train.jsonl', 'sft_val.jsonl', 'sft_test.jsonl']
    manifest = dict(
        seed=args.seed, seq_len=args.seq_len, source_url=SOURCE_URL,
        source_sha256=fingerprint(source), source_records=len(records),
        sft_counts={name: len(rows) for name, rows in splits.items()}, rejected=dict(rejected),
        original_corpus_lines=len(lines), unique_corpus_lines=len(unique),
        pretrain_train_lines=split, pretrain_val_lines=len(unique)-split,
        files_sha256={name: fingerprint(data / name) for name in files},
        prompt_format='<bos>问题<sep>答案<eos>',
        loss_policy='预训练预测所有有效下一个 token，SFT 仅预测答案及 EOS',
        dataset_note='公开 GPT-4 合成中文指令数据，筛选短样本并保留来源答案。')
    (data / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (data / 'vocab.json').write_text(json.dumps(tokenizer.vocab, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
