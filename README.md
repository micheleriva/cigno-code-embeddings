# cigno-code-small-v1

A small (22M parameter, 384-dimensional) code embedding model optimized for code search. Built via knowledge distillation from [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code).

Designed for [Cigno](https://github.com/micheleriva/cigno), a local-first code search engine.

## Features

- **22M parameters, 384 dimensions** — loads in ~1s, embeds at ~3ms/symbol on CPU
- **Ships as ~25MB quantized ONNX** — no GPU required
- **8 languages** — Python, TypeScript, JavaScript, Go, Rust, Java, C, C++
- **Code-aware** — understands that `getUserById` and `fetch_user_by_id` are semantically equivalent

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("micheleriva/cigno-code-small-v1")
embeddings = model.encode([
    "function: getUserById\nsignature: getUserById(id: string)",
    "function: fetch_user_by_id\nsignature: fetch_user_by_id(id: str)",
])
```

## Training

Distilled in two stages from `jinaai/jina-embeddings-v2-base-code` (137M params, 768d, Apache 2.0):

1. **Stage A — MSE distillation**: student learns to reproduce teacher embeddings on ~1M code snippets from [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata)
2. **Stage B — Contrastive fine-tuning**: sharpened on [CodeSearchNet](https://github.com/github/CodeSearchNet) (query, code) pairs

Base student architecture: [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) (MIT, Microsoft).

## Acknowledgments

- **Teacher model**: [jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) (Apache 2.0, Jina AI)
- **Student base**: [nreimers/MiniLM-L6-H384-uncased](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) (MIT, Microsoft)
- **Training data**: [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata) (OpenRAIL-M v1, BigCode)
- **Evaluation data**: [CodeSearchNet](https://github.com/github/CodeSearchNet) (MIT, GitHub) — Husain et al., 2019 ([arXiv:1909.09436](https://arxiv.org/abs/1909.09436))

## Use Restrictions

This model was trained on data governed by the [BigCode OpenRAIL-M v1](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) license. By using this model, you agree not to use it to:

1. Violate applicable laws or regulations
2. Exploit or harm minors
3. Generate malware or code designed to disrupt, damage, or gain unauthorized access to computer systems
4. Generate or spread disinformation intended to harm others
5. Generate content that is presented to the public without disclosing it is machine-generated

For the full list of use restrictions, see [BigCode OpenRAIL-M v1 Attachment A](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement).

## License

[MIT](LICENSE) — subject to the use restrictions above.
