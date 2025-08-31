# Build mini-gpt

## Goal:

1. Build a nano-gpt that works with a small dataset locally.
2. fine tune the llm

Out of scope:

- building transformers and other components from scratch

## Architecture:

### Dataset

- openweb-10k

### Training

- Autoregressive next work prediction

### Loss function:

- token level cross entropy

### Metrics:

- perplexity measured as log(loss)

### Acronyms

#### Encodings:

BPE - Byte Pair Encoding
CLM - causal language model
