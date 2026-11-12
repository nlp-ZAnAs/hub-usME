"""逐 token 生成。python generate.py --question '中国的首都是哪里？'"""
import argparse
import torch
from torch import nn
from model import ROOT, load_model


@torch.inference_mode()
def generate(model, tokenizer, question, max_new_tokens=128, temperature=0.0,
             top_k=40, repetition_penalty=1.1, raw=False):
    if not question.strip():
        raise ValueError('请输入非空问题')
    if max_new_tokens < 1 or temperature < 0 or top_k < 0 or repetition_penalty < 1:
        raise ValueError('生成长度须为正，temperature 和 top_k 须非负，重复惩罚须至少为 1')
    ids = [tokenizer.bos] + tokenizer.encode(question)
    if not raw:
        ids.append(tokenizer.sep)
    if len(ids) >= model.max_len:
        raise ValueError(f'问题过长，最多支持 {model.max_len - 3} 个字符')
    device = next(model.parameters()).device
    answer, reason = [], 'length'
    limit = min(max_new_tokens, model.max_len - len(ids))
    model.eval()
    for _ in range(limit):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        h = model.features(x)[:, -1, :]
        if isinstance(model.output, nn.AdaptiveLogSoftmaxWithLoss):
            scores = model.output.log_prob(h)[0]
        else:
            scores = model.output(h).log_softmax(-1)[0]
        scores[[tokenizer.pad, tokenizer.unk, tokenizer.bos, tokenizer.sep]] = -float('inf')
        if answer:
            scores[list(set(answer))] *= repetition_penalty
        if temperature == 0:
            next_id = int(scores.argmax())
        else:
            scores = scores / temperature
            if top_k:
                threshold = scores.topk(min(top_k, scores.numel())).values[-1]
                scores[scores < threshold] = -float('inf')
            next_id = int(torch.multinomial(scores.softmax(-1), 1))
        if next_id == tokenizer.eos:
            reason = 'eos'
            break
        answer.append(next_id)
        ids.append(next_id)
    return dict(text=tokenizer.decode(answer), generated_tokens=len(answer), finish_reason=reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(ROOT / 'checkpoints/sft.pt'))
    parser.add_argument('--question')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--raw', action='store_true', help='以普通文本续写格式输入，适合预训练权重')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer, checkpoint = load_model(args.checkpoint, device)
    print(f"已加载 {checkpoint['stage']}，最佳轮次 {checkpoint['epoch']}，输入 /exit 退出。")
    while True:
        try:
            question = args.question if args.question is not None else input('问题> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == '/exit':
            break
        try:
            result = generate(model, tokenizer, question, args.max_new_tokens,
                              args.temperature, args.top_k, raw=args.raw)
            print('回答> ' + result['text'])
        except ValueError as error:
            if args.question is not None:
                parser.error(str(error))
            print(error)
        if args.question is not None:
            break


if __name__ == '__main__':
    main()
