'''
Description:  
Author: Huang J
Date: 2025-04-01 09:48:23
'''
import logging
import math 
from typing import List,TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from chunk_factory.utils.util import split_text_by_punctuation, get_min_ppl,get_token_loss,show_figure

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

def ppl_chunker(
    text: str,
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
    threshold: float,
    language: str = 'zh',
    max_length: int = 2048,
    model_length: int = 8096,
    show_ppl_figure: bool = False,
    save_dir: str = ''
) -> List[str]:
    """
    Splits a long text into semantically coherent chunks based on sentence-level perplexity.

    Args:
        text (str): Input text to be split.
        model (PreTrainedModel): Pretrained language model to compute perplexity.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        threshold (float): Threshold for detecting local minima in perplexity scores.
        language (str): Language code ('zh' or 'en'). Default is 'zh'.
        max_length (int): Maximum input length per forward pass. Default is 2048.
        model_length (int): Max length the model supports with past key values. Default is 8096.

    Returns:
        List[str]: A list of text chunks split based on perplexity boundaries.
    """
    
    sentences = split_text_by_punctuation(text,language)
    use_batch=False
    len_sentences = []
    input_ids=torch.tensor([[]], device=model.device,dtype=torch.long)  
    attention_mask =torch.tensor([[]], device=model.device,dtype=torch.long) 
    
    for sentence in sentences:
        tokenized_sen = tokenizer(sentence, return_tensors="pt", add_special_tokens=False)
        input_id = tokenized_sen["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, input_id],dim=-1)
        len_sentences.append(input_id.shape[1])
        attention_mask_sen = tokenized_sen["attention_mask"].to(model.device)
        attention_mask = torch.cat([attention_mask, attention_mask_sen],dim=-1)
        
    text_token_len = input_ids.shape[1]
    if text_token_len > max_length:
        use_batch = True
        
    past_key_values = None
    if use_batch and text_token_len>max_length:
        batch_num = math.ceil(text_token_len/max_length)
        loss=torch.tensor([], device=model.device,dtype=torch.long)
        for i in range(batch_num):
            start=i*max_length
            end=start+max_length
            input_ids_tmp=input_ids[:,start:end]
            attention_mask_tmp=attention_mask[:,:end]
            if i>0:
                attention_mask_tmp=torch.cat([attention_mask_tmp, torch.ones((1, i), device=model.device, dtype=torch.long)],dim=-1)
            size_tmp=input_ids_tmp.shape[1]
            if attention_mask_tmp.shape[1]>model_length:  
                past_key_values = [  
                    [k[:, :, size_tmp+1: ], v[:, :, size_tmp+1: ]] 
                    for k, v in past_key_values  
                ]
                attention_mask_tmp=attention_mask_tmp[:, attention_mask_tmp.shape[1]-size_tmp-past_key_values[0][0].shape[2]:]
            loss_tmp, past_key_values = get_token_loss( 
                input_ids_tmp,
                attention_mask_tmp,
                past_key_values=past_key_values,
                return_kv=True
            )
            loss = torch.cat([loss, loss_tmp],dim=-1)  
            
    else:
        loss = get_token_loss(input_ids,attention_mask,model)
        
    ppl_sentences = []
    index = 0
    for i in range(len(len_sentences)):
        if i ==0:
            ppl_sentences.append(loss[0:len_sentences[i]-1].mean().item())
            index+=len_sentences[i]-1
        else:
            ppl_sentences.append(loss[index:index+len_sentences[i]].mean().item())
            index+=len_sentences[i]
            
    if show_ppl_figure:
        show_figure(ppl_sentences,row_name="Text Chunk",col_name='PPL Value',show_flag=True,save_dir=save_dir)
        
    min_indices = get_min_ppl(ppl_sentences,threshold)
    chunk_indices=[]
    chunk_sent_list=[]
    chunk_split_points = [0] + min_indices + [len(ppl_sentences)-1]  
      
    for i in range(len(chunk_split_points)-1):
        tmp_index=[]
        tmp_sentence=[]
        if i==0:
            tmp_index.append(0)
            tmp_sentence.append(sentences[0])
        for sp_index in range(chunk_split_points[i]+1,chunk_split_points[i+1]+1):
            tmp_index.append(sp_index)
            tmp_sentence.append(sentences[sp_index])
        chunk_indices.append(tmp_index)
        chunk_sent_list.append(tmp_sentence)
        
    text_chunks = []
    match language:
        case 'en':
            for sent_list in chunk_sent_list:
                text_chunks.append(' '.join(sent_list))
        case 'zh':
            for sent_list in chunk_sent_list:
                text_chunks.append(' '.join(sent_list))
    return text_chunks
