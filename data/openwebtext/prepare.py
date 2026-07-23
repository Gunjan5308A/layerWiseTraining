import os
import numpy as np
import tiktoken
from datasets import load_dataset

TARGET_TOKENS = 60_000_000
VAL_RATIO = 0.10

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":

    dataset = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
    )

    train_target = int(TARGET_TOKENS * (1 - VAL_RATIO))
    val_target = TARGET_TOKENS - train_target

    train_ids = []
    val_ids = []

    train_tokens = 0
    val_tokens = 0

    for example in dataset:
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)

        # Fill train first
        if train_tokens < train_target:
            remaining = train_target - train_tokens

            if len(ids) <= remaining:
                train_ids.extend(ids)
                train_tokens += len(ids)
            else:
                train_ids.extend(ids[:remaining])
                train_tokens += remaining

                leftover = ids[remaining:]
                if val_tokens < val_target:
                    remaining_val = val_target - val_tokens
                    val_ids.extend(leftover[:remaining_val])
                    val_tokens += min(len(leftover), remaining_val)

        elif val_tokens < val_target:
            remaining = val_target - val_tokens

            if len(ids) <= remaining:
                val_ids.extend(ids)
                val_tokens += len(ids)
            else:
                val_ids.extend(ids[:remaining])
                val_tokens += remaining

        if train_tokens >= train_target and val_tokens >= val_target:
            break

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)

    train_path = os.path.join(os.path.dirname(__file__), "train.bin")
    val_path = os.path.join(os.path.dirname(__file__), "val.bin")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"train.bin: {len(train_ids):,} tokens")
    print(f"val.bin:   {len(val_ids):,} tokens")
    print(f"total:     {len(train_ids)+len(val_ids):,} tokens")