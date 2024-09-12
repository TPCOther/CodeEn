from functools import lru_cache
import json
from typing import List
import sys
import jsonlines
from transformers import AutoTokenizer

from ATLAS_prompt_construction import split_focal_test_to_parts


if __name__ == "__main__":
    mode = "old"
    category = "Testing" # in ["Training", "Validation", "Testing"] 
    result_dir ={"Training": "train", "Testing": "test", "Validation": "val"}
    dataset_path = f"./Dataset/{mode}/"
    count = 0
    error_number = 0
    with open(dataset_path + f"{category}/" + "assertLines.txt", "r") as fassert, open(
            dataset_path + f"{category}/" + "testMethods.txt",
            "r") as fbody, jsonlines.open(dataset_path + f"{result_dir[category]}.raw.lou.jsonl", "w") as fjsonl:
        assert_lines = fassert.readlines()
        body_lines = fbody.readlines()
        print(len(assert_lines), len(body_lines))
        assert len(assert_lines) == len(body_lines)
        # 逐行对齐
        for assertion, body in zip(assert_lines, body_lines):
            # 通过codeT5的tokenizer对齐 gold 和 output

            item = {}
            item["gold"] = assertion.strip()
            item["focal_test"] = body.strip()
            fjsonl.write(item)
