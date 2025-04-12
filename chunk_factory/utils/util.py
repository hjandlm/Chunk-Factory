'''
Description:  
Author: Huang J
Date: 2025-03-31 14:40:18
'''
import os
import re
import time
from typing import Optional, List,TYPE_CHECKING
from collections import defaultdict
from typing import Dict, List, Any
from nltk.tokenize import sent_tokenize
import math
import jieba
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def count_words(sentence: str, language: str) -> int:
    """
    Approximate the number of tokens in a given sentence based on the language.

    For English, it splits the sentence into words and estimates the number of tokens
    by multiplying the word count by a factor of 1.2. For Chinese, it treats each character 
    as a separate token and estimates the number of tokens by multiplying the character 
    count by 1.2.

    Args:
        sentence (str): The input text to be processed.
        language (str): The language of the input text ('en' for English, 'zh' for Chinese).

    Returns:
        int: The estimated number of tokens.

    Raises:
        ValueError: If the language is not supported.
    """
    if language == 'en':
        words = sentence.split()
        return round(1.2 * len(words))
    elif language == 'zh':
        return round(1.2 * len(sentence))
    else:
        raise ValueError("Unsupported language. Please use 'zh' for Chinese or 'en' for English.")


def add_ids(row: pd.Series, row_index: int) -> pd.Series:
    """
    Adds a unique ID to each row in the DataFrame based on its index.

    This function modifies the 'Chunk' column by prepending a unique ID 
    to the existing content of the 'Chunk' column. The ID is formatted 
    as 'ID <row_index>: <existing_chunk_value>'.

    Args:
        row (pd.Series): A single row from the DataFrame.
        row_index (int): The index of the current row.

    Returns:
        pd.Series: The updated row with a unique ID in the 'Chunk' column.
    """

    row['Chunk'] = f'ID {row_index}: {row["Chunk"]}'
    return row


def split_text_by_punctuation(text: str, language: str) -> List[str]:
    """
    Splits the input text into sentences based on punctuation or length, depending on the language.

    For Chinese, it uses the jieba library to segment the text and identifies sentence boundaries
    based on punctuation marks like "。", "！", "？", and "；".
    
    For English, it uses the NLTK library's sent_tokenize function to split the text into sentences.
    The resulting sentences are returned as a list.

    Args:
        text (str): The input text to be split into sentences.
        language (str): The language type ('zh' for Chinese, 'en' for English).

    Returns:
        List[str]: A list of segmented sentences.

    Raises:
        ValueError: If the language is not supported (i.e., neither 'zh' nor 'en').
    """
    text = text.strip().strip('\n')
    
    match language:
        case 'zh':  
            words = jieba.cut(text, cut_all=False)
            words_list = list(words)
            sentences = []
            temp_sentence = ""
            
            for word in words_list:
                if word in ["。", "！", "？", "；"]:
                    sentences.append(temp_sentence.strip() + word)
                    temp_sentence = ""
                else:
                    temp_sentence += word
            if temp_sentence:
                sentences.append(temp_sentence.strip())
            return sentences
        
        case 'en':
            full_sentences = sent_tokenize(text)
            sentences = []
            for sentence in full_sentences:
                sentences.append(sentence.strip())
            return sentences
        
        case _:
            raise ValueError("Unsupported language. Please use 'zh' for Chinese or 'en' for English.")
        

def match_answer_id(llm_response: str) -> int:
    """
    Extracts the chunk ID from the LLM response string.

    The function looks for the chunk ID in the response formatted as 'Answer: ID XXXX'. 
    If no chunk ID is found, it returns -1.

    Args:
        llm_response (str): The response from the LLM, which may contain a chunk ID.

    Returns:
        int: The extracted chunk ID as an integer. Returns -1 if no chunk ID is found.
    """
    pattern_answer = r"Answer: ID \w+"
    match = re.search(pattern_answer, llm_response)
    if match is None:
        return -1  
    temp_content = match.group(0)
    pattern_id = r'\d+'
    match = re.search(pattern_id, temp_content)
    if match is None:
        return -1  
    return int(match.group())

def show_figure(
    data: List[float],
    row_name: str = 'Text Chunk',
    col_name: str = 'PPL Value',
    show_flag: bool = True,
    save_dir: Optional[str] = None
) -> None:
    """
    Displays a line plot of the given data and optionally saves the plot to a file.

    Args:
        data (List[float]): A list of values to be plotted on the y-axis.
        row_name (str, optional): Label for the x-axis. Defaults to 'Text Chunk'.
        col_name (str, optional): Label for the y-axis. Defaults to 'PPL Value'.
        show_flag (bool, optional): If True, displays the plot. Defaults to True.
        save_dir (Optional[str], optional): Directory to save the plot. If None, the plot is not saved.

    Returns:
        None
    """
    df = pd.DataFrame({
        'Index': list(range(1,len(data)+1)),
        'Value': data
    })
    
    sns.set_theme(style='ticks')
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='Index', y='Value', marker='o', color='teal')
    plt.xlabel(row_name)
    plt.ylabel(col_name)
    
    if save_dir:
        if os.path.exists(save_dir):
            os.makedirs(save_dir)
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        figure_name = f'Text_Chunk_Value_{now}.png'
        save_path = os.path.join(save_dir,figure_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    if show_flag:
        plt.show()

def get_min_ppl(ppl_values: List[float], threshold: float) -> List[int]:
    """
    Identifies the indices of local minima in perplexity values where the difference between 
    the current perplexity value and its neighbors exceeds a given threshold.

    Args:
        ppl_values (List[float]): A list of perplexity values.
        threshold (float): The threshold value to detect significant changes in perplexity.

    Returns:
        List[int]: A list of indices where local minima occur and the difference from neighbors exceeds the threshold.
    """
    min_index = []  
    for i in range(1, len(ppl_values) - 1):  
        if ppl_values[i] < ppl_values[i-1] and ppl_values[i] < ppl_values[i+1]:
            if (ppl_values[i-1]-ppl_values[i]>=threshold) or (ppl_values[i+1]-ppl_values[i]>=threshold):
                min_index.append(i)  
        elif ppl_values[i] < ppl_values[i-1] and ppl_values[i] == ppl_values[i+1]:
            if ppl_values[i-1]-ppl_values[i]>=threshold:
                min_index.append(i) 
        else:
            continue
    return min_index

def get_token_loss(
    input_id: torch.Tensor,
    attention_mask: torch.Tensor,
    model: 'PreTrainedModel',
    past_key_values=None,
    use_cache: bool = False
) -> tuple[torch.Tensor, any] | torch.Tensor:
    """
    Compute token-level loss for a given input using a language model.

    Args:
        input_ids (torch.Tensor): The input token IDs of shape [batch_size, seq_len].
        attention_mask (torch.Tensor): The attention mask indicating valid tokens.
        model (PreTrainedModel): HuggingFace-style language model.
        past_key_values (optional): Cached past key values for fast decoding.
        use_cache (bool): Whether to use and return past key values.

    Returns:
        torch.Tensor: Loss tensor of shape [valid_token_count].
        If use_cache is True, also returns updated past_key_values.
    """
    with torch.no_grad():
        response = model(input_id,attention_mask=attention_mask,past_key_values=past_key_values,use_cache=use_cache)
    if use_cache:
        past_key_values = response.past_key_values
    logits = response.logits[..., :-1, :].contiguous() 
    label = input_id[..., 1:].contiguous() 
    active_mask = (attention_mask==1)[..., :-1].view(-1)
    
    active_logits = logits.view(-1, logits.size(-1))[active_mask]
    active_labels = label.view(-1)[active_mask]
    loss_func = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_func(active_logits, active_labels)  
    if use_cache:
        return loss,past_key_values
    return loss

def build_graph(edges: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    Build a graph from a list of edge dictionaries by counting the degree of each node.

    Args:
        edges (List[Dict[str, Any]]): List of edges, each represented as a dict 
                                      with 'row' and 'column' keys.

    Returns:
        Dict[int, int]: A dictionary where keys are node indices and values are node degrees.
    """
    zgraph = defaultdict(int)
    for edge in edges:
        node1 = edge['row']
        node2 = edge['column']
        zgraph[node1] += 1
        zgraph[node2] += 1
    return zgraph

def calculate_structural_entropy(graph: Dict[int, int]) -> float:
    """
    Calculate the structural entropy of a graph based on node degrees.

    Args:
        graph (Dict[int, int]): A dictionary of node degrees.

    Returns:
        float: The entropy value representing structural diversity.
    """

    total_degree = sum(graph.values())
    entropy = 0
    for node, degree in graph.items():
        if degree > 0:
            p = degree / total_degree
            entropy -= p * math.log(p, 2)
    return entropy

def find_values_greater_than(matrix: Dict[int, Dict[int, float]], threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Find entries in a nested matrix (dictionary of dictionaries) that exceed a threshold.

    Args:
        matrix (Dict[int, Dict[int, float]]): Nested dictionary with float values.
        threshold (float): Minimum value to include in the result.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries with 'row', 'column', and 'value' keys.
    """

    results = []
    for row_key, row_value in matrix.items():
        for col_key, value in row_value.items():
            if value > threshold and row_key != col_key:
                results.append({'row':row_key, 'column':col_key, 'value':value})
    return results

def get_ppl_next(
    chunk1: str,
    chunk2: str,
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer'
) -> float:
    """
    Calculate the next-word prediction perplexity (PPL) loss of chunk2 conditioned on chunk1.

    Args:
        chunk1 (str): The preceding text chunk.
        chunk2 (str): The target text chunk to evaluate.
        model (PreTrainedModel): The language model to evaluate with.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.

    Returns:
        float: The total cross-entropy loss over chunk2 given chunk1.
    """
    chunk1_tokenized = tokenizer(chunk1, return_tensors="pt", add_special_tokens=False)
    chunk2_tokenized = tokenizer(chunk2, return_tensors="pt", add_special_tokens=False)
    input_ids=torch.cat([chunk1_tokenized["input_ids"],chunk2_tokenized["input_ids"]],dim=-1).to(model.device)
    attention_mask = torch.cat([chunk1_tokenized["attention_mask"],chunk2_tokenized["attention_mask"]],dim=-1).to(model.device)
    with torch.no_grad():
        response = model(input_ids,attention_mask=attention_mask)
    chunk1_length=chunk1_tokenized["input_ids"].shape[1]
    
    shift_logits = response.logits[..., chunk1_length-1:-1, :].contiguous()
    shift_labels = input_ids[..., chunk1_length: ].contiguous()
    active = (attention_mask[:, chunk1_length:] == 1).view(-1)
    active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
    active_labels = shift_labels.view(-1)[active]
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(active_logits, active_labels)
    total_loss = loss.sum().item()
    return total_loss
    