# FineWeb600M Dataset

Extracted ~600M tokens from HuggingFace's FineWeb dataset (sample-10BT) for training smaller GPT models.

## Data Preparation

```sh
python data/fineweb600m/prepare.py
```

### Output

| File | Size | Tokens |
|------|------|--------|
| `train.bin` | ~1.2 GB | ~594M |
| `val.bin` | ~12 MB | ~6M |

- **Total documents**: Sampled from FineWeb sample-10BT
- **Tokenization**: GPT-2 BPE (`tiktoken`)
- **Token format**: uint16 (saves memory vs uint32)
- **Storage**: Memory-mapped files (`np.memmap`) for efficient loading

## How It Works

1. Streams FineWeb dataset from HuggingFace
2. Tokenizes each document with GPT-2 BPE
3. Fills `train.bin` first (~594M tokens), overflow goes to `val.bin` (~6M tokens)
4. Uses multiprocessing for parallel tokenization
5. Preallocates on-disk arrays (no RAM blowup)

## Usage

### With layerWiseTrain.py

```sh
python layerWiseTrain.py config/train_gpt2_64m.py --dataset=fineweb600m
```

### With train.py

```sh
python train.py config/train_gpt2_64m.py --dataset=fineweb600m
```

### Override dataset

```sh
python layerWiseTrain.py --dataset=fineweb600m
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