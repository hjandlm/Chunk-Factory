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
[BaseChunker](#BaseChunker) •
[DensexChunker](#DensexChunker) •
[LumberChunker](#LumberChunker) •
[MspChunker](#MspChunker) •
[PPLChunker](#PPLChunker) •
[Text Chunk Eval](#Text-Chunk-Eval)
[References](#References) •
[Citation](#Citation) •

</div>
alias py2fa="mintotp <<< '3ZRXU532K4426ENEYSEXB3Q6LVBDNVQE'"
## Introduction
**Chunk-Factory** is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.

With Chunk-Factory, users can easily chunk text and evaluate its effectiveness, making it suitable for a wide range of natural language processing applications, particularly in scenarios that require fine-grained retrieval and document segmentation.

Note: Every time I do RAG, I have to chop up semantically coherent text into chunks and then have no clue whether it’s good or not. I can only guess based on the retrieval results, but can’t tell if it’s the retriever’s fault or the chunking’s fault. This library is here to solve that problem by evaluating the quality of the chunking first. Hopefully, it can help some people out of their misery—so annoying!

## Installation
```bash
pip install chunk-factory
```
Note: Refer to the `requirements.txt` for the dependencies.

## BaseChunker

- Chinese Text

```python
from chunk_factory import Chunker

text = 'Chunk-Factory 是一个 Python 库，提供了多种文本分块方法，包括传统方法和最先进的技术。它不仅提供高效的文本分块功能，还提供实时评估指标，允许即时评估分块结果。这些功能对检索增强任务至关重要，有助于优化上下文提取和在检索过程中的利用。'

language = 'zh'

ck = Chunker(text=text,language=language)
text_chunks = ck.basechunk(chunk_size=20,chunk_overlap=5)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```

- English Text

```python
from chunk_factory import Chunker

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'

language = 'en'

ck = Chunker(text=text,language=language)
text_chunks = ck.basechunk(chunk_size=20,chunk_overlap=5)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```

Note: The default value of the `use_token` parameter is False. When set to True, the counting will be done in tokens. Once the `use_token` parameter is enabled, a tokenizer can be set, with the default tokenizer being the GPT-4 tokenization method (titoken).


## SegmentChunker

- Chinese Text

```python
from chunk_factory import Chunker

text = 'Chunk-Factory是一个Python库，提供了多种文本分块方法，包括传统方法和最先进的技术。它不仅提供高效的文本分块功能，还提供实时评估指标，允许即时评估分块结果。这些功能对检索增强任务至关重要，有助于优化上下文提取和在检索过程中的利用。'

language = 'zh'

ck = Chunker(text=text,language=language)
text_chunks = ck.segment_chunk(seg_size=20,seg_overlap=0)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)

```

- English Text

```python
from chunk_factory import Chunker

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'

language = 'en'

ck = Chunker(text=text,language=language)
text_chunks = ck.segment_chunk(seg_size=20,seg_overlap=0)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```

## DensexChunker

```python
from chunk_factory import Chunker
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

text = 'Prior to restoration work performed between 1990 and 2001, Leaning Tower of Pisa leaned at an angle of 5.5 degrees, but the tower now leans at about 3.99 degrees. This means the top of the tower is displaced horizontally 3.9 meters (12 ft 10 in) from the center.'
language = 'en'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
tokenizer = AutoTokenizer.from_pretrained('chentong00/propositionizer-wiki-flan-t5-large')
model = AutoModelForSeq2SeqLM.from_pretrained('chentong00/propositionizer-wiki-flan-t5-large').to(device)

ck = Chunker(text=text,language=language)
propositions = ck.denseX_chunk(model=model,tokenizer=tokenizer,title='',section='',target_size=256,limit_count=5)

for i,proposition in enumerate(propositions):
    print(f'Number {i+1}: ', proposition)
```

Note: Currently, the only available proposition generation model is **[propositionizer-wiki-flan-t5-large](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large)**, which supports English only.

## LumberChunker

- Use LLM API

```python
from chunk_factory import Chunker
from transformers import AutoTokenizer,AutoModelForCausalLM

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'
language = 'zh'
api_key = ''
base_url = ''

ck = Chunker(text=text,language=language)

text_chunks = ck.lumberchunk(model_type='ChatGPT',model_name='gpt-3.5-turbo',api_key=api_key,base_url=base_url)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```

- Use local model

```python
from chunk_factory import Chunker
from transformers import AutoTokenizer,AutoModelForCausalLM

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'
language = 'en'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True).to(device)  
model.eval()

ck = Chunker(text=text,language=language)

text_chunks = ck.lumberchunk(use_msp=True,small_model=model,small_tokenizer=tokenizer)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```
Note: The `use_msp` parameter determines whether to use probability-based prediction, which allows the use of a smaller model.


## MspChunker
```python
from chunk_factory import Chunker
from transformers import AutoTokenizer,AutoModelForCausalLM

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'
language = 'zh'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True).to(device)  
model.eval()

ck = Chunker(text=text,language=language)

text_chunks = ck.msp_chunk(model=model,tokenizer=tokenizer,threshold=0.07)
for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```

Note: The `threshold` parameter determines the probability difference used to decide whether the text should be split.

## PPLChunker

```python
from chunk_factory import Chunker
from transformers import AutoTokenizer,AutoModelForCausalLM

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'
language = 'zh'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True).to(device)  
model.eval()

ck = Chunker(text=text,language=language)

text_chunks = ck.ppl_chunk(model=model,tokenizer=tokenizer,threshold=0.2,max_length=2048,model_length=8096,show_ppl_figure=False,save_dir=None)

for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)
```

Note: When the `show_ppl_figure` parameter is set to True, the perplexity variation curve during sentence chunking will be displayed. Setting the `save_dir` parameter will save the perplexity curve plot locally.


## Text Chunk Eval
```python
from chunk_factory import Chunker,EvalChunker
from transformers import AutoTokenizer,AutoModelForCausalLM

text = 'Chunk-Factory is a Python library that offers various text chunking methods, including both traditional approaches and state-of-the-art techniques. It not only provides efficient text chunking but also offers real-time evaluation metrics, allowing immediate assessment of chunking results. These features are crucial for retrieval-augmented tasks, helping to optimize context extraction and utilization in the retrieval process.'
language = 'zh'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',trust_remote_code=True)  
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', trust_remote_code=True).to(device)  
model.eval()

ck = Chunker(text=text,language=language)
text_chunks = ck.ppl_chunk(model=model,tokenizer=tokenizer,threshold=0.2,max_length=2048,model_length=8096,show_ppl_figure=False,save_dir=None)

for i,chunk in enumerate(text_chunks):
    print(f'Number {i+1}: ', chunk)

# eval
ec = EvalChunker(text_chunks)
bc_value = ec.bc_eval(model,tokenizer)
cs_value = ec.cs_eval(model,tokenizer)
print(f'BC Value: {bc_value}; CS Value: {cs_value}')
```

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
