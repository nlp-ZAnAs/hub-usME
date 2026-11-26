"""关键回归检查：CRF 正确性、BIO 往返与约束、指标口径、生成式解析与回填、数据泄漏。"""

import itertools
import unittest

import torch

from common import (LABELS, ROOT, TYPES, bio_to_spans, count_illegal, error_categories, evaluate,
                    spans_to_bio, transition_constraints)
from crf import CRF, constrained_argmax, constraint_scores
from llm_common import (build_messages, mentions_to_spans, parse_mentions, parse_to_entities,
                        pick_few_shot, target_json)


def ent(kind, start, end, text):
    return {"type": kind, "start": start, "end": end, "text": text[start:end]}


class CRFTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.crf = CRF(4)
        self.emissions = torch.randn(2, 4, 4)
        self.mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)

    def brute(self, b, extra_start=None, extra_trans=None):
        n = int(self.mask[b].sum())
        start = self.crf.start + (extra_start if extra_start is not None else 0)
        trans = self.crf.trans + (extra_trans if extra_trans is not None else 0)
        scores = {}
        for path in itertools.product(range(4), repeat=n):
            s = start[path[0]] + self.emissions[b, 0, path[0]] + self.crf.end[path[-1]]
            for i in range(1, n):
                s = s + trans[path[i - 1], path[i]] + self.emissions[b, i, path[i]]
            scores[path] = s
        return scores

    def test_log_partition_and_path_score_match_enumeration(self):
        logz = self.crf.log_partition(self.emissions, self.mask)
        tags = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        path = self.crf.path_score(self.emissions, tags, self.mask)
        for b in range(2):
            scores = self.brute(b)
            expected = torch.logsumexp(torch.stack(list(scores.values())), 0)
            self.assertAlmostEqual(logz[b].item(), expected.item(), places=4)
            n = int(self.mask[b].sum())
            self.assertAlmostEqual(path[b].item(), scores[tuple(tags[b, :n].tolist())].item(),
                                   places=4)

    def test_viterbi_matches_enumeration_with_and_without_constraints(self):
        allowed_start = torch.tensor([True, True, False, True])
        allowed = torch.ones(4, 4, dtype=torch.bool)
        allowed[0, 2] = allowed[1, 3] = False
        start, trans = constraint_scores(allowed_start.numpy(), allowed.numpy(), "cpu")
        for constraints in [None, (start, trans)]:
            paths = self.crf.decode(self.emissions, self.mask, constraints)
            for b in range(2):
                scores = self.brute(b, *(constraints or (None, None)))
                self.assertEqual(tuple(paths[b]), max(scores, key=lambda p: scores[p].item()))


class BIOTest(unittest.TestCase):
    text = "浙商银行行长叶老桂在北京"

    def test_round_trip(self):
        entities = [ent("company", 0, 4, self.text), ent("position", 4, 6, self.text),
                    ent("name", 6, 9, self.text), ent("address", 10, 12, self.text)]
        self.assertEqual(bio_to_spans(self.text, spans_to_bio(self.text, entities)), entities)

    def test_lenient_decoding_and_illegal_count(self):
        tags = ["I-company", "I-company", "O", "B-name", "I-game"] + ["O"] * 7
        spans = bio_to_spans(self.text, tags)
        self.assertEqual([(s["type"], s["start"], s["end"]) for s in spans],
                         [("company", 0, 2), ("name", 3, 4), ("game", 4, 5)])
        self.assertEqual(count_illegal(tags), 2)

    def test_constrained_decoding_never_illegal(self):
        torch.manual_seed(1)
        emissions = torch.randn(64, 20, len(LABELS)) * 5
        mask = torch.ones(64, 20, dtype=torch.bool)
        mask[::2, 15:] = False
        constraints = constraint_scores(*transition_constraints(), "cpu")
        raw = emissions.argmax(-1).tolist()
        self.assertGreater(sum(count_illegal([LABELS[i] for i in p]) for p in raw), 0)
        for paths in [constrained_argmax(emissions, mask, constraints),
                      CRF(len(LABELS)).decode(emissions, mask, constraints)]:
            for path, m in zip(paths, mask):
                self.assertEqual(len(path), int(m.sum()))
                self.assertEqual(count_illegal([LABELS[i] for i in path]), 0)


class MetricTest(unittest.TestCase):
    def test_strict_boundary_text_views(self):
        text = "张三在北京大学当教授"
        gold = [{"id": "a", "text": text, "entities": [
            ent("name", 0, 2, text), ent("organization", 3, 7, text), ent("position", 8, 10, text)]}]
        pred = [{"id": "a", "entities": [
            ent("name", 0, 2, text), ent("address", 3, 5, text), ent("scene", 8, 10, text)]}]
        m = evaluate(gold, pred)
        self.assertEqual((m["strict"]["tp"], m["strict"]["fp"], m["strict"]["fn"]), (1, 2, 2))
        self.assertEqual(m["boundary"]["tp"], 2)
        self.assertEqual(m["per_type"]["position"]["support"], 1)
        errors = error_categories(gold[0]["entities"], pred[0]["entities"])
        self.assertEqual(len(errors["type_error"]), 1)
        self.assertEqual(len(errors["boundary_error"]), 1)
        self.assertEqual(errors["missed"], [])

    def test_empty_prediction_stays_in_denominator(self):
        text = "张三"
        gold = [{"id": "a", "text": text, "entities": [ent("name", 0, 2, text)]},
                {"id": "b", "text": text + "。", "entities": [ent("name", 0, 2, text)]}]
        pred = [{"id": "a", "entities": [ent("name", 0, 2, text)]}, {"id": "b", "entities": []}]
        self.assertAlmostEqual(evaluate(gold, pred)["strict"]["recall"], 0.5)

    def test_id_order_is_checked(self):
        gold = [{"id": "a", "text": "x", "entities": []}, {"id": "b", "text": "y", "entities": []}]
        with self.assertRaises(ValueError):
            evaluate(gold, [{"id": "b", "entities": []}, {"id": "a", "entities": []}])


class LLMParsingTest(unittest.TestCase):
    def test_fenced_truncated_and_alias(self):
        mentions, stats = parse_mentions('```json\n{"name":["张三"],"公司":["阿里"],"x":["y"]}\n```')
        self.assertEqual(mentions, {"name": ["张三"], "company": ["阿里"]})
        self.assertEqual((stats["status"], stats["type_alias"], stats["invalid_type"]), ("ok", 1, 1))
        mentions, stats = parse_mentions('{"name":["张三","李')
        self.assertEqual(stats["status"], "repaired")
        self.assertEqual(mentions["name"][0], "张三")
        self.assertEqual(parse_mentions("我无法回答")[1]["status"], "invalid")

    def test_longest_first_and_policies(self):
        text = "北京大学位于北京，北京很大"
        mentions = {"address": ["北京"], "organization": ["北京大学"], "name": ["王五"]}
        spans, missing = mentions_to_spans(text, mentions, "all")
        self.assertEqual(missing, 1)
        self.assertEqual([(s["type"], s["start"]) for s in spans],
                         [("organization", 0), ("address", 6), ("address", 9)])
        spans, _ = mentions_to_spans(text, mentions, "first")
        self.assertEqual([(s["type"], s["start"]) for s in spans],
                         [("organization", 0), ("address", 6)])

    def test_gold_target_round_trip(self):
        text = "张三和李四在北京"
        row = {"id": "a", "text": text,
               "entities": [ent("name", 0, 2, text), ent("name", 3, 5, text),
                            ent("address", 6, 8, text)]}
        spans, stats = parse_to_entities(text, target_json(row), "all")
        self.assertEqual(spans, row["entities"])
        self.assertEqual(stats["status"], "ok")
        self.assertEqual(build_messages(text)[-1]["content"], "句子：" + text)


class SFTLossTest(unittest.TestCase):
    def test_answer_only_logits_match_standard_masked_loss(self):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        from run_llm_local import answer_loss_sum, collate
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64))
        rows = [{"input_ids": [5, 6, 7, 8, 9, 10], "labels": [-100, -100, -100, 8, 9, 10]},
                {"input_ids": [11, 12, 13, 14], "labels": [-100, -100, 13, 14]}]
        batch = collate(rows, pad_id=0)
        loss_sum, count = answer_loss_sum(model, batch)
        (loss_sum / count).backward()
        fast = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()
        reference = model(**batch).loss
        reference.backward()
        self.assertEqual(count, 5)
        self.assertAlmostEqual((loss_sum / count).item(), reference.item(), places=5)
        for a, b in zip(fast, (p.grad for p in model.parameters())):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


class DataTest(unittest.TestCase):
    def test_splits_and_few_shot(self):
        from common import load_data
        splits = load_data()   # 内部校验 ID/文本跨集合不重复、offset 正确、指纹一致
        examples = pick_few_shot(splits["train"], 5, 42)
        self.assertEqual({e["type"] for r in examples for e in r["entities"]}, set(TYPES))
        train_ids = {r["id"] for r in splits["train"]}
        self.assertTrue(all(r["id"] in train_ids for r in examples))
        self.assertTrue((ROOT / "data" / "api_subset_ids.json").exists())


if __name__ == "__main__":
    unittest.main()
