'''
Description:  
Author: Huang J
Date: 2025-04-02 10:12:59
'''

import math
from typing import List,TYPE_CHECKING


import torch
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedModel, PreTrainedTokenizer

from .util import get_ppl_next, find_values_greater_than,build_graph,calculate_structural_entropy

def get_semantic_similarity(
    text1: str,
    text2: str,
    encode_model: 'SentenceTransformer'
) -> float:
    """
    Computes semantic similarity between two texts using a sentence embedding model.

    Args:
        text1 (str): First input text.
        text2 (str): Second input text.
        encode_model (SentenceTransformer): Pretrained sentence embedding model.

    Returns:
        float: Cosine similarity score between the two embeddings.
    """
    q_embeddings = encode_model.encode([q for q in text1], normalize_embeddings=True)
    p_embeddings = encode_model.encode(text2, normalize_embeddings=True)
    scores = q_embeddings @ p_embeddings.T
    return scores.item()


def bc_calculate(
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
    pre_text: str,
    next_text: str
) -> float:
    """
    Calculates the perplexity (PPL) of `next_text` given `pre_text` as context,
    using a causal language model.

    Args:
        model (PreTrainedModel): The language model used for evaluation.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        pre_text (str): Context text (prefix).
        next_text (str): Target text to evaluate given the context.

    Returns:
        float: Perplexity of `next_text` conditioned on `pre_text`.
    """
    pre_tokenized = tokenizer(pre_text, return_tensors="pt", add_special_tokens=False)
    next_tokenized = tokenizer(next_text, return_tensors="pt", add_special_tokens=False)
    pre_input_ids = pre_tokenized['input_ids']
    next_input_ids = next_tokenized['input_ids']
    pre_mask = pre_tokenized['attention_mask']
    next_mask = next_tokenized['attention_mask']
    input_ids=torch.cat([pre_input_ids, next_input_ids],dim=-1).to(model.device)
    attention_mask = torch.cat([pre_mask, next_mask],dim=-1).to(model.device)
    with torch.no_grad():
        response = model(input_ids, attention_mask=attention_mask)
    pre_text_len = pre_input_ids.shape[1]
    next_logits = response.logits[..., pre_text_len-1:-1, :].contiguous()
    next_labels = input_ids[..., pre_text_len:].contiguous()
    active = (attention_mask[:, pre_text_len:] == 1).view(-1)
    active_logits = next_logits.view(-1, next_logits.size(-1))[active]
    active_labels = next_labels.view(-1)[active]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    next_text_loss = loss_fct(active_logits, active_labels)  
    avg_next_text_loss = next_text_loss.mean()
    next_text_ppl = torch.exp(avg_next_text_loss).item()
    return next_text_ppl
    

def cs_calculate(
    chunks: List[str],
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
    delta: float = 0.0
) -> float:
    """
    Calculate the structural coherence (CS value) of text chunks based on perplexity graph.

    Args:
        chunks (List[str]): List of input text chunks.
        model (PreTrainedModel): Language model to compute perplexity.
        tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
        delta (float): Position-based penalty scaling factor (default is 0.0).

    Returns:
        float: The structural entropy value.
    """
    graph_ppl = {}
    chunks_ppl = []
    for chunk in chunks:
        chunk_ppl = get_ppl_next(' ',chunk,model,tokenizer)
        chunks_ppl.append(chunk_ppl)
        
    for i in range(len(chunks)):
        graph_ppl[i] = {}
        for j in range(len(chunks)):
            if i==j:
                graph_ppl[i][j] = chunks_ppl[i]
            else:
                chunk_ppl_next = get_ppl_next(chunks[i],chunks[j],model,tokenizer)
                graph_ppl[i][j] = chunk_ppl_next
    chunks_token = [tokenizer.encode(chunk, return_tensors='pt').shape[1] for chunk in chunks]
    graph_weight = {}
    for i in range(len(chunks)):
        graph_weight[i] = {}
        for j in range(len(chunks)):
            if i==j:
                graph_weight[i][j] = 1
            else:
                weight_temp=(math.exp(graph_ppl[j][j] /chunks_token[j])-math.exp(graph_ppl[i][j] /chunks_token[j]))/math.exp(graph_ppl[j][j] /chunks_token[j])
                weight=-weight_temp+1+delta*abs(i-j)/(len(chunks)-1) 
                graph_weight[i][j] = weight   
    find_values=find_values_greater_than(graph_weight, 0.8)
    graph_cs = build_graph(find_values)
    cs_value = calculate_structural_entropy(graph_cs)
    return cs_value
    
    


                
    



