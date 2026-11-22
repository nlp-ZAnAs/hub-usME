"""回归检查最容易使实验结论失真的边界；不下载权重，不启动完整训练。"""

import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from common import config, fingerprint, metrics, validate_splits
from prepare_data import clean_rows
from run_neural import (answer_loss_sum, batch_groups, make_sft_item, parse_label,
                        sft_collate)
from summarize import paired_bootstrap, validate_result
import numpy as np


class TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text, **kwargs):
        return {"input_ids": [3, 4, 5] if text == "prompt" else [6, 7]}


class ExperimentIntegrityTest(unittest.TestCase):
    def test_failed_generation_counts_as_false_negative(self):
        result = metrics([0, 1], [0, -1])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["per_class"]["正面"]["recall"], 0)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 2)

    def test_all_invalid_is_zero_not_missing(self):
        self.assertEqual(metrics([0, 1], [-1, -1])["macro_f1"], 0)

    def test_duplicate_prefix_conflicts_are_removed_before_splitting(self):
        rows = [{"review": " 同 样内容A", "label": "1"},
                {"review": "同样内容B", "label": "1"},
                {"review": "冲突长文本A", "label": "0"},
                {"review": "冲突长文本B", "label": "1"},
                {"review": "", "label": "1"}]
        clean, audit = clean_rows(rows, max_chars=4)
        self.assertEqual(len(clean), 1)
        self.assertEqual(audit["empty_rows_removed"], 1)
        self.assertEqual(audit["duplicate_input_rows_removed"], 1)
        self.assertEqual(audit["conflicting_rows_removed"], 2)

    def test_split_overlap_fails(self):
        rows = [{"id": "a", "text": "甲", "label": 0},
                {"id": "b", "text": "乙", "label": 1}]
        with self.assertRaises(ValueError):
            validate_splits({"train": rows, "test": rows})

    def test_parser_rejects_ambiguous_explanation(self):
        self.assertEqual(parse_label("“正面”。"), 1)
        self.assertEqual(parse_label("负面"), 0)
        self.assertEqual(parse_label("可能是正面，也可能是负面"), -1)
        self.assertEqual(parse_label(""), -1)

    def test_prompt_and_padding_do_not_produce_loss(self):
        row = make_sft_item(TinyTokenizer(), {"text": "评论", "label": 1}, 8)
        self.assertEqual(row["labels"], [-100, -100, -100, 6, 7, 2])
        batch = sft_collate([row, {"input_ids": [3, 4], "labels": [-100, 4]}], 0)
        self.assertTrue((batch["labels"][1, 2:] == -100).all())
        with self.assertRaises(ValueError):
            make_sft_item(TinyTokenizer(), {"text": "评论", "label": 1}, 5)

    def test_incomplete_accumulation_window_is_not_dropped(self):
        self.assertEqual(list(batch_groups(range(5), 2)), [[0, 1], [2, 3], [4]])

    def test_same_sample_count_does_not_hide_different_test_order(self):
        splits = {"test": [{"id": "a", "label": 0}, {"id": "b", "label": 1}]}
        result = {"method": "example", "split_sha256": {k: fingerprint(v) for k, v in splits.items()},
                  "config": config(), "predictions": [{"id": "b", "true": 1, "pred": 1},
                                                      {"id": "a", "true": 0, "pred": 0}],
                  "metrics": metrics([0, 1], [0, 1])}
        with self.assertRaises(ValueError):
            validate_result(result, splits)

    def test_paired_bootstrap_identical_predictions_have_zero_difference(self):
        truth = np.array([0, 0, 1, 1])
        pred = np.array([0, -1, 1, 1])
        result = paired_bootstrap(truth, pred, pred, repetitions=40)
        self.assertEqual(result["macro_f1"]["ci95"], [0., 0.])

    def test_answer_only_logits_equal_standard_masked_loss_and_gradient(self):
        torch.manual_seed(7)
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, attention_dropout=0.0))
        batch = sft_collate([
            {"input_ids": [3, 4, 5, 6, 2], "labels": [-100, -100, -100, 6, 2]},
            {"input_ids": [3, 8, 7, 2], "labels": [-100, -100, 7, 2]},
        ], 0)
        full = model(**batch, use_cache=False).loss
        full.backward()
        grad = model.model.embed_tokens.weight.grad.clone()
        model.zero_grad()
        partial_sum, count = answer_loss_sum(model, batch)
        partial = partial_sum / count
        partial.backward()
        torch.testing.assert_close(partial, full)
        torch.testing.assert_close(model.model.embed_tokens.weight.grad, grad, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
