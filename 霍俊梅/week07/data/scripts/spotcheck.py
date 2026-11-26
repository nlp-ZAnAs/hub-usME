"""
spotcheck.py — 抽样检查数据集质量。

用法:
  python3 spotcheck.py                 # 每类型每集合抽 2 条
  python3 spotcheck.py --n 5           # 每类型每集合抽 5 条
  python3 spotcheck.py --file dev.json # 只看某个集合

渲染: 在句子文本中按 span 插入标记, 形如
  三车[追尾]事故，造成〔肇事者: 司机李某〕受伤...
  - [X]        = trigger
  - 〔role: X〕  = argument
"""
import os
import json
import random
import argparse
from collections import defaultdict

OUT_DIR = os.path.dirname(__file__)


def render(text, trig, args):
    """按 span 位置在 text 中插入标记; 同一位置的 closer 在 opener 之前 (相邻 span 正确分隔)"""
    from collections import defaultdict
    by_pos = defaultdict(list)
    by_pos[trig['start']].append(('open', '〔T:'))
    by_pos[trig['end']].append(('close', '〕'))
    for a in args:
        by_pos[a['start']].append(('open', f'〔{a["role"]}:'))
        by_pos[a['end']].append(('close', '〕'))
    out = []
    for i in range(len(text) + 1):
        if i in by_pos:
            for _, s in sorted(by_pos[i], key=lambda x: 0 if x[0] == 'close' else 1):
                out.append(s)
        if i < len(text):
            out.append(text[i])
    return ''.join(out)


def load_json_flex(path):
    """兼容 JSON 数组 (parse_cec 当前输出) 和旧版 JSONL"""
    txt = open(path, encoding='utf-8').read().strip()
    if not txt:
        return []
    if txt.startswith('['):
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def load_samples(files):
    samples = []
    for fn in files:
        path = os.path.join(OUT_DIR, fn)
        if not os.path.exists(path):
            continue
        samples.extend(load_json_flex(path))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2, help='每类型抽样条数')
    ap.add_argument('--file', default=None, help='只抽某个集合文件')
    ap.add_argument('--seed', type=int, default=13)
    args = ap.parse_args()

    files = [args.file] if args.file else ['train.json', 'dev.json', 'test.json']
    samples = load_samples(files)
    if not samples:
        print('no samples in', files, '(先运行 parse_cec.py)')
        return

    rng = random.Random(args.seed)
    by_type = defaultdict(list)
    for s in samples:
        by_type[s['event']['type']].append(s)

    for et in ['地震', '火灾', '交通事故', '恐怖袭击', '食物中毒']:
        group = by_type.get(et, [])
        if not group:
            continue
        print('=' * 90)
        print(f'## {et}   (集合内 {len(group)} 条, 抽 {min(args.n, len(group))} 条)')
        print('=' * 90)
        for s in rng.sample(group, min(args.n, len(group))):
            ev = s['event']
            print(f'id={s["id"]}')
            print('  原文 :', s['text'])
            print('  标注 :', render(s['text'], ev['trigger'], ev['args']))
            print(f'  trigger={ev["trigger"]["text"]!r}  args={len(ev["args"])}')
            for a in ev['args']:
                print(f'     - {a["role"]}: {a["text"]!r} [{a["start"]}:{a["end"]}]')
            print()


if __name__ == '__main__':
    main()
