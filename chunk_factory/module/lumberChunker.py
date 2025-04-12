'''
Description:  
Author: Huang J
Date: 2025-03-31 11:01:43
'''
import logging
from typing import List, Optional,TYPE_CHECKING
import pandas as pd
import torch
import torch.nn.functional as F
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from chunk_factory.utils.util import split_text_by_punctuation, add_ids, count_words,match_answer_id
from chunk_factory.utils.prompts import lumberchunker_prompt,msp_lumberchunker_prompt
from chunk_factory.utils.llm import llm_response_api

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

def get_maxprob_index(
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
    user_prompt: str,
    len_doc: int
) -> int:
    """
    Selects the index (0-based) of the option with the highest predicted probability 
    among multiple numerical responses (e.g., 0, 1, 2, ..., len_doc-1) from a language model.

    For each index i, the function calculates the probability that the model will generate i
    as the next response token(s) and returns the index with the highest average token probability.

    Args:
        model (PreTrainedModel): The language model to use.
        tokenizer (PreTrainedTokenizer): The tokenizer for encoding input/output strings.
        user_prompt (str): The prompt to be fed into the model (user message).
        len_doc (int): The number of candidate indices (from 0 to len_doc - 1).

    Returns:
        int: The index with the highest predicted probability.
    """
    prompt="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(user_prompt)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    output_ids = [tokenizer.encode(str(i), return_tensors='pt').to(model.device)[0].cpu().tolist() for i in range(len_doc)]
    
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        token_probs = F.softmax(next_token_logits, dim=-1)
        
    prob_list=[]
    for ids in output_ids:
        next_token_id = ids[0] 
        next_token_prob = token_probs[:, next_token_id].item()  
        
        if len(ids)==1:
            prob_list.append(next_token_prob)
        else:
            toavg_prob=[next_token_prob]
            for id in ids[1:]:
                with torch.no_grad():
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=model.device)],dim=-1)
                    outputs = model(input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    token_probs = F.softmax(next_token_logits, dim=-1)
                next_token_prob = token_probs[:, id].item()  
                toavg_prob.append(next_token_prob)
                next_token_id=id
            prob_list.append(sum(toavg_prob) / len(toavg_prob))
            
    max_value = max(prob_list)  
    max_prob_index = prob_list.index(max_value)  
    return max_prob_index

def lumberchunker(
    text: str,
    language: str,
    model_type: str = None,
    model_name: str = None,
    api_key: str = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    use_msp = False,
    small_model: Optional[object] = None,
    small_tokenizer: Optional[object] = None
) -> List[str]:
    """
    Splits input text into manageable chunks and processes each using a language model.

    Depending on the `use_msp` flag, this function supports two modes:
    - Using a small model to identify key chunks.
    - Using a general LLM API to process and return structured responses.

    Args:
        text (str): The input text to be chunked and processed.
        language (str): The language of the text, used to determine punctuation splitting rules.
        model_type (str): The type of LLM used (e.g., "ChatGPT", "Gmini").
        model_name (str): The name of the model (e.g., "gpt-3.5-turbo").
        api_key (str): API key for accessing the language model service.
        base_url (Optional[str], optional): Optional base URL for the LLM API. Defaults to None.
        system_prompt (Optional[str], optional): Optional prompt to guide LLM behavior. Defaults to None.
        use_msp (bool, optional): Whether to use a small model for chunk selection. Defaults to False.
        small_model (Optional[object], optional): The small model instance (e.g., HuggingFace model). Required if use_msp is True.
        small_tokenizer (Optional[object], optional): Tokenizer corresponding to the small model. Required if use_msp is True.
    Returns:
        List[str]: A list of processed text chunks.
    """
    if use_msp:
        if small_model and small_tokenizer:
            pass
        else:
            raise ValueError('Please check if the `small_model` and `small_tokenizer` parameters are correct.')
        
    else:
        if model_type and model_name and api_key and base_url:
            pass
        else:
            raise ValueError('Please check if the `model_type`, `model_name`, `api_key`, and `base_url` parameters are correct.')
        
    chunks = split_text_by_punctuation(text,language)
    df_chunks = pd.DataFrame(chunks, columns=['Chunk'])
    
    if not use_msp:
        id_chunks = df_chunks.apply(lambda row: add_ids(row, row.name), axis=1)
    else:
        id_chunks = df_chunks
        
    chunk_number = 0
    chunk_id_list = []
    text_chunks = []
    
    while chunk_number < len(id_chunks)-5:
        word_count = 0
        i = 0
        while word_count < 550 and i+chunk_number<len(id_chunks)-1:
            i += 1
            temp_doc = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
            word_count = count_words(temp_doc,language)
        
        if use_msp:
            if(i ==1):
                doc = [id_chunks.at[k, 'Chunk'] for k in range(chunk_number, i + chunk_number)]
            else:
                doc = [id_chunks.at[k, 'Chunk'] for k in range(chunk_number, i-1 + chunk_number)]
                
            tmp_doc = ''
            tmp_doc_len = len(doc)
            for i,s in enumerate(doc):
                tmp_doc += '\n'+str(i)+': '+s
                
            tmp_doc = f"\nDocument:\n{tmp_doc}"
            user_prompt = msp_lumberchunker_prompt(language)
            user_prompt += tmp_doc
            chunk_number_model = get_maxprob_index(small_model,small_tokenizer,user_prompt,tmp_doc_len)
            
            if chunk_number_model!=0:
                chunk_number = chunk_number + chunk_number_model
                text_chunks.append(' '.join([j for i,j in enumerate(doc) if i<int(chunk_number_model)]))
            else:
                chunk_number = chunk_number + 1
                text_chunks.append(doc[0])
                
        else:   
            if i==1:
                doc = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i + chunk_number))
            else:
                doc = "\n".join(f"{id_chunks.at[k, 'Chunk']}" for k in range(chunk_number, i-1 + chunk_number))
                
            doc = f"\nDocument:\n{doc}"
            user_prompt = lumberchunker_prompt(language)
            user_prompt += doc
            while True:
                response = llm_response_api(user_prompt,model_type,model_name,api_key,base_url=base_url,system_prompt=system_prompt)
                chunk_number = match_answer_id(response)
                if chunk_number==-1:
                    continue
                break
            chunk_id_list.append(chunk_number)
            
    if use_msp:
        doc = [id_chunks.at[k, 'Chunk'] for k in range(chunk_number, len(id_chunks))]
        text_chunks.append(' '.join(doc))
        
    else:
        chunk_id_list.append(len(id_chunks))
        id_chunks['Chunk'] = id_chunks['Chunk'].str.replace(r'^ID \d+:\s*', '', regex=True)
        for i in range(len(chunk_id_list)):
            start_idx = chunk_id_list[i-1] if i > 0 else 0   
            end_idx = chunk_id_list[i]
            text_chunks.append(' '.join(id_chunks.iloc[start_idx:end_idx, 0]))
            
    return text_chunks

