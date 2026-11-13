"""在独立问答测试集上计算答案 PPL，并保存前后生成示例。"""
import argparse
import json
import torch
from model import ROOT, load_model
from train import make_loader, run_epoch
from prepare_data import read_jsonl, fingerprint
from generate import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = read_jsonl(ROOT / 'data/sft_test.jsonl')
    report = dict(test_examples=len(rows), metric='答案字符和 EOS 的 token 加权 PPL', models={})
    for name in ('pretrained', 'sft'):
        path = ROOT / 'checkpoints' / f'{name}.pt'
        model, tokenizer, checkpoint = load_model(path, device)
        for data_name, digest in checkpoint['data_manifest']['files_sha256'].items():
            if fingerprint(ROOT / 'data' / data_name) != digest:
                raise ValueError('评估数据与训练记录不一致')
        if name == 'sft' and checkpoint['init_sha256'] != fingerprint(ROOT / 'checkpoints/pretrained.pt'):
            raise ValueError('SFT 初始化权重的指纹不一致')
        loader = make_loader('sft', 'test', tokenizer, model.max_len, 32)
        score = run_epoch(model, loader, device)
        examples = []
        for row in rows[:args.examples]:
            output = generate(model, tokenizer, row['question'])
            examples.append(dict(question=row['question'], reference=row['answer'], **output))
        report['models'][name] = dict(checkpoint_sha256=fingerprint(path), epoch=checkpoint['epoch'],
                                      score=score, examples=examples)
        print(name, score, flush=True)
    report['note'] = '测试集仅用于训练后评估。生成使用贪心选择和 1.1 重复惩罚，最多 128 token。答案 PPL 衡量参考答案概率，问答正确性需结合生成文本判断。'
    destination = ROOT / 'evaluation.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(f'评分与生成示例已保存到 {destination}')


if __name__ == '__main__':
    main()
