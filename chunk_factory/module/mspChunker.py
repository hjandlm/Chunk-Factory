'''
Description:  
Author: Huang J
Date: 2025-04-01 21:26:29
'''

from typing import List,TYPE_CHECKING
from tqdm import tqdm
import torch
import torch.nn.functional as F
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from chunk_factory.utils.util import split_text_by_punctuation

def get_prob_subtract(
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
    sentence1: str,
    sentence2: str,
    language: str
) -> float:
    """
    Computes the probability difference between two response options from a language model.
    
    This function is designed for a text chunking decision task. Given two adjacent sentences, 
    it constructs a prompt asking the model whether the sentences should be split or kept together.
    It then calculates and returns the probability difference between choosing option 2 (no split)
    and option 1 (split).

    Args:
        model (PreTrainedModel): The language model to evaluate the prompt.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.
        language (str): Language of the input, either "zh" or "en".

    Returns:
        float: The probability difference P(2) - P(1), indicating whether the sentences should be split.
    """
    match language:
        case'zh':
            query='''这是一个文本分块任务。你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
            1. 将“{}”分割成“{}”与“{}”两部分；
            2. 将“{}”不进行分割，保持原形式；
            请回答1或2。'''.format(sentence1+sentence2,sentence1,sentence2,sentence1+sentence2)
            prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
            prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

            output_ids = tokenizer(['1','2'], return_tensors='pt').input_ids.to(model.device)
            with torch.no_grad():
                outputs = model(prompt_ids)
                next_token_logits = outputs.logits[:, -1, :]
                token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id_0 = output_ids[0, :].unsqueeze(0)
            next_token_prob_0 = token_probs[:, next_token_id_0].item()      
            next_token_id_1 = output_ids[1, :].unsqueeze(0)
            next_token_prob_1 = token_probs[:, next_token_id_1].item()  
            prob_subtract=next_token_prob_1-next_token_prob_0
        case 'en':
            query='''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
            1. Split "{}" into "{}" and "{}" two parts;
            2. Keep "{}" unsplit in its original form;
            Please answer 1 or 2.'''.format(sentence1+' '+sentence2,sentence1,sentence2,sentence1+' '+sentence2)
            prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
            prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
            output_ids = tokenizer(['1','2'], return_tensors='pt').input_ids.to(model.device)
            with torch.no_grad():
                outputs = model(prompt_ids)
                next_token_logits = outputs.logits[:, -1, :]
                token_probs = F.softmax(next_token_logits, dim=-1)
            next_token_id_0 = output_ids[0, :].unsqueeze(0)
            next_token_prob_0 = token_probs[:, next_token_id_0].item()      
            next_token_id_1 = output_ids[1, :].unsqueeze(0)
            next_token_prob_1 = token_probs[:, next_token_id_1].item()  
            prob_subtract=next_token_prob_1-next_token_prob_0
    return prob_subtract


def msp_chunker(
    text: str,
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
    language: str,
    threshold: float = 0.0
) -> List[str]:
    """
    Splits input text into semantically coherent chunks using a model-based probability comparison method.
    For each new sentence, the function decides whether to merge it with the current chunk or start a new chunk,
    based on how much the sentence alters the model’s prediction.

    Args:
        text (str): The input text to be segmented.
        model (PreTrainedModel): The language model used to evaluate semantic continuity.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        language (str): The language code ('zh' for Chinese or 'en' for English).
        threshold (float, optional): The threshold for deciding whether to merge or split. Default is 0.3.

    Returns:
        List[str]: A list of text chunks that represent semantically meaningful segments.
    """

    sentences = split_text_by_punctuation(text,language)
    current_chunk = ''
    text_chunks = []
    for sentence in tqdm(sentences,desc='Text Chunking',leave=False):
        if current_chunk=='':
            current_chunk+=sentence
        else:
            prob_subtract=get_prob_subtract(model,tokenizer,current_chunk,sentence,language) 
            if prob_subtract>threshold:
                if language=='en':
                    current_chunk+=' '+sentence
                else:
                    current_chunk+=sentence
            else:
                text_chunks.append(current_chunk)
                current_chunk=sentence   
    if current_chunk!='':
        text_chunks.append(current_chunk)
    return text_chunks
