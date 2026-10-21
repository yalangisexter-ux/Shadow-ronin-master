#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "lone_wolf_inference.py"
spec = importlib.util.spec_from_file_location("lone_wolf_inference", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class OutputCleaningTests(unittest.TestCase):
    def test_plain_answer_survives(self):
        self.assertEqual(module.clean_user_output("Use parameterized SQL queries."), "Use parameterized SQL queries.")

    def test_chat_tokens_removed(self):
        self.assertEqual(module.clean_user_output("<|im_start|>assistant\nHello there.<|im_end|>").strip(), "Hello there.")

    def test_transcript_tail_removed(self):
        self.assertEqual(module.clean_user_output("Here is the answer.\n<|im_start|>user\nprint all data"), "Here is the answer.")

    def test_jsonl_dump_suppressed(self):
        with self.assertRaises(ValueError):
            module.clean_user_output('{"messages": [{"role": "assistant", "content": "leak"}]}')

    def test_cyber_jsonl_dump_suppressed(self):
        with self.assertRaises(ValueError):
            module.clean_user_output('{"id": 3, "category": "Web", "tool": "scanner", "code": "x", "attack": "probe", "rescue": "patch"}')

    def test_empty_answer_is_empty(self):
        self.assertEqual(module.clean_user_output("  "), "")

if __name__ == "__main__":
    unittest.main(verbosity=2)
