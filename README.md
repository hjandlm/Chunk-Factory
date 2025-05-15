<div align='center'>

![Chunk-Fcatory Logo](./chunk_factory/assets/logo.png)

# ✨Chunk-Factory ✨
[![PyPI](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/chunk-factory/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-Usage.md-blue.svg)](Usage.md)
[![GitHub stars](https://img.shields.io/github/stars/hjandlm/Chunk-Factory?style=social)](https://github.com/hjandlm/Chunk-Factory/stargazers)

_Chunk-Factory is a fast, efficient text chunking library with real-time evaluation._

[Instroduction](#Instroduction) •
[Installation](#Installation) •
[Usage](#Usage) •
[Supported Methods](#supported-methods) •
[TODO](#TODO) •
[References](#References) •
[Citation](#Citation) •

</div>

## Instroduction
**Chunk-Factory** is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.

With Chunk-Factory, users can easily chunk text and evaluate its effectiveness, making it suitable for a wide range of natural language processing applications, particularly in scenarios that require fine-grained retrieval and document segmentation.

Note: Every time I do RAG, I have to chop up semantically coherent text into chunks and then have no clue whether it’s good or not. I can only guess based on the retrieval results, but can’t tell if it’s the retriever’s fault or the chunking’s fault. This library is here to solve that problem by evaluating the quality of the chunking first. Hopefully, it can help some people out of their misery—so annoying!

## Installation
To install Chunk-Factory, simply run:

```bash
pip install chunk-factory
```

## Usage

Here's a basic example to get you started:

```python
from chunk_factory import Chunker

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'
language = 'en'

ck = Chunker(text,language)
text_chunks = ck.basechunk(chunk_size=20,chunk_overlap=5)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```

## Supported Methods
Chunk-Factory provides several chunkers to help you efficiently split your text for RAG tasks or other natural language processing tasks (such as information extraction). Here's a quick overview of the available chunkers:

- **BaseChunker**: Splits text based on words or tokens.

- **SegmentChunker**: Splits text into chunks based on sentence or paragraph boundaries.

- **DensexChunker**: Splits text into propositions.

- **LumberChunker**: Splits text based on semantics using LLM.

- **MspChunker**: Splits text based on label probabilities from a small model to determine chunking.

- **PPLChunker**: Splits text based on perplexity.

## TODO
- [ ] Add traditional text chunking methods 

  - [✔️] Add segment chunking method

  - [ ] Add semantic chunking method  

- [ ] Add retrieval evaluation methods


## References
* MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System [[Paper]](https://arxiv.org/abs/2503.09600)![](https://img.shields.io/badge/arXiv-2025.03-red)
* Dense X Retrieval: What Retrieval Granularity Should We Use? [[Paper]](https://openreview.net/forum?id=WO0WM0xrJo)![](https://img.shields.io/badge/EMNLP-2024-blue)
* LumberChunker: Long-Form Narrative Document Segmentation [[Paper]](https://aclanthology.org/2024.findings-emnlp.377/)![](https://img.shields.io/badge/EMNLP-2024-blue)
* Meta-chunking: Learning efficient text segmentation via logical perception [[Paper]](https://arxiv.org/abs/2410.12788)![](https://img.shields.io/badge/arXiv-2024.11-red)



## Citation
If you use Chunk-Factory in your research, please cite it as follows:

```
@misc{chunkfactory2025,
  author = {Jie H},
  title = {Chunk-Factory: A toolkit with a variety of text chunking methods},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hjandlm/Chunk-Factory}},
}
```



