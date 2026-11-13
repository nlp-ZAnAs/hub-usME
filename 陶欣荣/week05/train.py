"""继续预训练与全参数 SFT。运行 python train.py pretrain 或 python train.py sft。"""
import argparse
import json
import math
import random
from pathlib import Path
from time import perf_counter
import torch
from torch.utils.data import Dataset, DataLoader
from model import ROOT, INITIAL_CHECKPOINT, initialize_from_previous, load_model, answer_example, collate_examples
from prepare_data import read_jsonl, fingerprint


class TextDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        ids = []
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            ids.extend([tokenizer.bos] + tokenizer.encode(line) + [tokenizer.eos])
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.seq_len = seq_len
        self.length = math.ceil((len(ids) - 1) / seq_len)
        if self.length < 1:
            raise ValueError('预训练语料过短')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        start = index * self.seq_len
        end = min(start + self.seq_len, len(self.ids) - 1)
        return self.ids[start:end], self.ids[start+1:end+1]


def make_loader(stage, split, tokenizer, seq_len, batch_size, shuffle=False):
    extension = 'txt' if stage == 'pretrain' else 'jsonl'
    path = ROOT / 'data' / f'{stage}_{split}.{extension}'
    if stage == 'pretrain':
        dataset = TextDataset(path, tokenizer, seq_len)
    else:
        dataset = [answer_example(row, tokenizer, seq_len) for row in read_jsonl(path)]
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_examples)


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_nll, total_tokens = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            loss = model.token_loss(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('训练损失出现非有限值')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = int((y != -100).sum())
            total_nll += loss.item() * count
            total_tokens += count
            if optimizer is not None and step % 50 == 0:
                print(f'  batch {step}/{len(loader)} PPL {math.exp(total_nll / total_tokens):.2f}', flush=True)
    if not total_tokens:
        raise ValueError('评估集没有有效目标')
    loss = total_nll / total_tokens
    return dict(loss=loss, ppl=math.exp(loss), tokens=total_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['pretrain', 'sft'])
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=.0003)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init', type=Path)
    parser.add_argument('--save', type=Path)
    args = parser.parse_args()
    epochs = args.epochs if args.epochs is not None else (1 if args.stage == 'pretrain' else 8)
    if epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error('epochs、batch_size、lr 必须大于零')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf8'))
    for name, digest in manifest['files_sha256'].items():
        if fingerprint(ROOT / 'data' / name) != digest:
            raise ValueError(f'数据被修改，请重新准备数据: {name}')
    initial = args.init or (INITIAL_CHECKPOINT if args.stage == 'pretrain'
                            else ROOT / 'checkpoints/pretrained.pt')
    if args.stage == 'pretrain':
        model, tokenizer, config = initialize_from_previous(initial, manifest['seq_len'])
    else:
        model, tokenizer, checkpoint = load_model(initial)
        if checkpoint['stage'] != 'pretrain':
            raise ValueError('SFT 必须从预训练权重初始化')
        if checkpoint['data_manifest'] != manifest:
            raise ValueError('预训练与当前数据清单不一致')
        config = checkpoint['config']
    model.to(device)
    destination = args.save or ROOT / 'checkpoints' / ('pretrained.pt' if args.stage == 'pretrain' else 'sft.pt')
    if destination.resolve() == initial.resolve():
        raise ValueError('输出路径必须与初始化权重不同')
    destination.parent.mkdir(parents=True, exist_ok=True)
    train = make_loader(args.stage, 'train', tokenizer, config['seq_len'], args.batch_size, True)
    val = make_loader(args.stage, 'val', tokenizer, config['seq_len'], args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    report = dict(stage=args.stage, init=str(initial), init_sha256=fingerprint(initial),
                  epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed,
                  parameters=sum(p.numel() for p in model.parameters()), device=str(device),
                  config=config, data_manifest=manifest, history=[])
    report['before_training_val'] = run_epoch(model, val, device)
    print(f"{args.stage} 参数 {report['parameters']:,} 初始验证 {report['before_training_val']}", flush=True)
    best = float('inf')
    for epoch in range(1, epochs+1):
        started = perf_counter()
        training = run_epoch(model, train, device, optimizer)
        validation = run_epoch(model, val, device)
        report['history'].append(dict(epoch=epoch, train=training, val=validation, seconds=perf_counter()-started))
        if validation['loss'] < best:
            best = validation['loss']
            report['best_epoch'] = epoch
            report['best_val'] = validation
            checkpoint = dict(model_state=model.state_dict(), vocab=tokenizer.vocab, config=config,
                              stage=args.stage, epoch=epoch, val=validation, init_sha256=report['init_sha256'],
                              data_manifest=manifest)
            temporary = destination.with_suffix('.tmp')
            torch.save(checkpoint, temporary)
            temporary.replace(destination)
        destination.with_suffix('.metrics.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        print(f"{args.stage} {epoch}/{epochs} train PPL {training['ppl']:.2f} "
              f"val PPL {validation['ppl']:.2f} {perf_counter()-started:.0f}s", flush=True)
    print(f"完成，最佳轮次 {report['best_epoch']}，权重 {destination}", flush=True)


if __name__ == '__main__':
    main()
