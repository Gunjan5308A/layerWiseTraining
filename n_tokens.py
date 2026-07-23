import numpy as np
import tiktoken

enc = tiktoken.get_encoding("gpt2")
print(enc.max_token_value)
# 1. Read the binary file using the same dtype (uint16)
m = np.memmap('data/openwebtext/val.bin', dtype=np.uint16, mode='r')

# 2. Look at the first 20 raw token IDs
print(len(m))