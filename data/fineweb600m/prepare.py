import os
import time
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

TARGET_TOKENS = 600_000_000
VAL_TOKENS = 6_000_000
TRAIN_TOKENS = TARGET_TOKENS - VAL_TOKENS

enc = tiktoken.get_encoding("gpt2")


def tokenize(doc):
    ids = enc.encode_ordinary(doc["text"])
    ids.append(enc.eot_token)
    return np.array(ids, dtype=np.uint16)


if __name__ == "__main__":
    t0 = time.time()
    data_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    train_arr = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(TRAIN_TOKENS,))
    val_arr = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(VAL_TOKENS,))

    fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

    train_idx = 0
    val_idx = 0
    done = 0
    nprocs = max(1, os.cpu_count() - 2)

    try:
        with mp.Pool(nprocs) as pool:
            it = pool.imap(tokenize, fw, chunksize=16)
            pbar = tqdm(it, total=TARGET_TOKENS, unit="tokens", desc="tokenizing fineweb600m")
            for ids in pbar:
                n = len(ids)
                if train_idx < TRAIN_TOKENS:
                    take = min(n, TRAIN_TOKENS - train_idx)
                    train_arr[train_idx:train_idx + take] = ids[:take]
                    train_idx += take
                    ids = ids[take:]
                    n -= take
                if n > 0 and val_idx < VAL_TOKENS:
                    take = min(n, VAL_TOKENS - val_idx)
                    val_arr[val_idx:val_idx + take] = ids[:take]
                    val_idx += take
                done = train_idx + val_idx
                pbar.update(done - pbar.n)
                if train_idx >= TRAIN_TOKENS and val_idx >= VAL_TOKENS:
                    break
    except KeyboardInterrupt:
        print("interrupted, flushing partial data...")

    train_arr.flush()
    val_arr.flush()
    del train_arr, val_arr

    print(f"train.bin: {train_idx:,} tokens")
    print(f"val.bin:   {val_idx:,} tokens")
    print(f"total:     {train_idx + val_idx:,} tokens")
    print(f"elapsed:   {time.time() - t0:.1f}s")