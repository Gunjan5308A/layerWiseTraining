# FineWeb5B Dataset

Extracted ~1.5B tokens from HuggingFace's FineWeb dataset (sample-10BT) for training GPT-2 scale models.

## Data Preparation

```sh
python data/fineweb5b/prepare.py
```

### Output

| File | Size | Tokens |
|------|------|--------|
| `train.bin` | ~3 GB | ~1.49B |
| `val.bin` | ~20 MB | ~10M |

- **Total documents**: Sampled from FineWeb sample-10BT
- **Tokenization**: GPT-2 BPE (`tiktoken`)
- **Token format**: uint16 (saves memory vs uint32)
- **Storage**: Memory-mapped files (`np.memmap`) for efficient loading

### How It Works

1. Streams FineWeb dataset from HuggingFace
2. Tokenizes each document with GPT-2 BPE
3. Fills `train.bin` first (~1.49B tokens), overflow goes to `val.bin` (~10M tokens)
4. Uses multiprocessing for parallel tokenization
5. Preallocates on-disk arrays (no RAM blowup)

## Usage

### With layerWiseTrain.py (default)

```sh
python layerWiseTrain.py config/gpt2_124m.py
```

### With train.py

```sh
python train.py config/gpt2_124m.py
```

### Override dataset

```sh
python layerWiseTrain.py --dataset=fineweb5b
python layerWiseTrain.py --data_dir=/path/to/custom/data
```

## Tokenization Details

The dataset uses OpenAI's GPT-2 BPE tokenizer:
- Vocabulary size: 50,257
- `enc.encode_ordinary()` ignores special tokens
- `enc.eot_token` (50256) appended per document

```python
import tiktoken
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode_ordinary(text)
ids.append(enc.eot_token)  # end of document
```

## References

- [FineWeb dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [HuggingFace FineWeb paper](https://arxiv.org/abs/2406.17557)
- [nanoGPT data loading](https://github.com/karpathy/nanoGPT/blob/master/train.py)
