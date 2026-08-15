
## OpenWebText Dataset

OpenWebText is an open reproduction of OpenAI's WebText dataset (the dataset used to train GPT-2).

### Data Preparation

After running `python data/openwebtext/prepare.py`:

- **train.bin**: ~17GB (GPT-2 BPE tokens stored as raw uint16)
- **val.bin**: ~8.5MB
- **Train tokens**: ~9B (9,035,582,198)
- **Val tokens**: ~4M (4,434,897)
- **Total documents**: 8,013,769

### Tokenization

The dataset is tokenized using OpenAI's GPT-2 BPE tokenizer (`tiktoken`):
- Vocabulary size: 50,257
- Token format: uint16 (saves memory vs. uint32)
- Stored as memory-mapped files for efficient loading

### How It's Used

The training loop uses `np.memmap` to load tokens directly from disk without loading the entire dataset into memory:

```python
data = np.memmap('train.bin', dtype=np.uint16, mode='r')
# Random sampling of sequences of length `block_size`
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
```

The target `y` is the input `x` shifted by one position (next-token prediction).

### References

- [OpenAI GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/)
