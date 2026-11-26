"""BERT token tagging and schema-aware BIO decoding."""
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from data import TRIGGER, source_span

BERT = "google-bert/bert-base-chinese"
MAX_LENGTH = 384
LAYERS = 8


def make_labels(schema):
    triggers = [f"{TRIGGER}:{kind}" for kind in schema]
    roles = sorted({role for roles in schema.values() for role in roles})
    return ["O", *[f"{bio}-{tag}" for tag in triggers + roles for bio in ("B", "I")]]


class Tagger(nn.Module):
    def __init__(self, base, labels):
        super().__init__()
        config = AutoConfig.from_pretrained(base, num_hidden_layers=LAYERS, return_dict=True)
        encoder = AutoModel.from_pretrained(base, config=config, add_pooling_layer=False)
        self.encoder = get_peft_model(encoder, LoraConfig(
            task_type="FEATURE_EXTRACTION", r=12, lora_alpha=24, lora_dropout=0.05,
            target_modules=["query", "key", "value", "dense"],
        ))
        self.head = nn.Linear(config.hidden_size, len(labels))
        self.drop = nn.Dropout(0.1)
        self.labels = labels

    def forward(self, inputs):
        return self.head(self.drop(self.encoder(**inputs).last_hidden_state))


def encode(tokenizer, rows, device, label_ids=None):
    batch = tokenizer([row["text"] for row in rows], padding=True, truncation=True,
                      max_length=MAX_LENGTH, return_offsets_mapping=True, return_tensors="pt")
    offsets = batch.pop("offset_mapping").numpy()
    if label_ids is None:
        return batch.to(device), offsets
    labels = np.full(offsets.shape[:2], -100, dtype=np.int64)
    for i, (pairs, row) in enumerate(zip(offsets, rows, strict=True)):
        starts, ends = pairs.T
        valid = ends > starts
        labels[i, valid] = 0
        event = row["event"]
        spans = [(f"{TRIGGER}:{event['type']}", event["trigger"]),
                 *((a["role"], a) for a in event["args"])]
        # Retain the trigger first, then fit nonoverlapping arguments.
        for tag, span in spans:
            if not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int):
                continue
            left, right = span["start"], span["end"]
            if not (0 <= left < right <= len(row["text"])) or right > ends.max():
                continue
            ids = np.flatnonzero(valid & (starts < right) & (ends > left))
            if len(ids) and np.all(labels[i, ids] == 0):
                labels[i, ids[0]] = label_ids[f"B-{tag}"]
                labels[i, ids[1:]] = label_ids[f"I-{tag}"]
    return batch.to(device), torch.as_tensor(labels, device=device)


def viterbi(emissions):
    """Best BIO path. Each I state can only follow its matching B or I."""
    length, states = emissions.shape
    begin = np.arange(1, states, 2)
    inside = begin + 1
    scores = emissions[0].copy()
    scores[inside] = -np.inf
    back = np.empty((length, states), dtype=np.int16)
    for i in range(1, length):
        best = int(scores.argmax())
        prev = np.full(states, best, dtype=np.int16)
        prev[inside] = np.where(scores[begin] >= scores[inside], begin, inside)
        scores = scores[prev] + emissions[i]
        back[i] = prev
    path = [int(scores.argmax())]
    for i in range(length - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    return path[::-1]


def decode(text, offsets, logits, labels, schema, kind=None):
    valid = offsets[:, 1] > offsets[:, 0]
    offsets, logits = offsets[valid], logits[valid].copy()
    if not len(offsets):
        return None
    if kind is not None:
        allowed = {f"{TRIGGER}:{kind}", *schema[kind]}
        logits[:, [i for i, tag in enumerate(labels) if i and tag[2:] not in allowed]] = -np.inf
    path = viterbi(logits)
    spans = []
    i = 0
    while i < len(path):
        state = path[i]
        if not state:
            i += 1
            continue
        end = i + 1
        while end < len(path) and path[end] == state + 1:
            end += 1
        span = source_span(text, int(offsets[i, 0]), int(offsets[end-1, 1]))
        weight = float(np.mean([logits[t, path[t]] - logits[t, 0] for t in range(i, end)]))
        spans.append((labels[state][2:], span, weight))
        i = end
    triggers = [s for s in spans if s[0].startswith(TRIGGER + ":")]
    if not triggers:
        return None
    tag, trigger, _ = max(triggers, key=lambda s: s[2])
    kind = tag.split(":", 1)[1]
    return {"type": kind, "trigger": trigger,
            "args": [{**span, "role": tag} for tag, span, _ in spans if tag in schema[kind]]}


def restore_weights(model, state):
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Checkpoint keys differ from the trainable BERT parameters")
    model.load_state_dict(state, strict=False)


def load_tagger(base, checkpoint, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Tagger(base, saved["labels"])
    restore_weights(model, saved["state"])
    return model.to(device).eval(), AutoTokenizer.from_pretrained(base, use_fast=True)
