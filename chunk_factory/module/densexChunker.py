'''
Description:  
Author: Huang J
Date: 2025-03-31 16:59:06
'''
import logging
import json
from tqdm import tqdm
from typing import List,TYPE_CHECKING
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from chunk_factory.utils.util import split_text_by_punctuation

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

def denseX_chunker(
    text: str,
    model: 'PreTrainedModel',
    tokenizer: 'PreTrainedTokenizer',
    language: str = 'en',
    title: str = '',
    section: str = '',
    target_size: int = 256,
    limit_count: int = 5
) -> List[dict]:
    """
    Splits input text into smaller chunks, processes them using a transformer model,
    and extracts structured propositions in JSON format.

    Parameters:
    text (str): Input text to be processed.
    model (PreTrainedModel): Transformer model for text generation.
    tokenizer (PreTrainedTokenizer): Tokenizer corresponding to the model.
    language (str, optional): Language of the text (default is 'en').
    title (str, optional): Title of the document (default is '').
    section (str, optional): Section of the document (default is '').
    target_size (int, optional): Maximum token size for each chunk (default is 256).
    limit_count (int, optional): Maximum number of retries if JSON parsing fails (default is 5).

    Returns:
    List[dict]: A list of extracted propositions in JSON format.

    Raises:
    ValueError: If the language is not 'en' or if JSON parsing fails repeatedly.
    """
    if language != 'en':
        raise ValueError("Invalid value for 'language'. It can only take the value 'en'.")
    
    sentences = split_text_by_punctuation(text,language)
    merged_chunks = []  
    current_chunks = "" 
    propositions = []
    use_count = 0
    
    for sentence in sentences:
        tmp_input_text = f"Title: {title}. Section: {section}. Content: {current_chunks+' '+sentence}".strip()
        if len((tokenizer(tmp_input_text, return_tensors="pt").input_ids)[0].tolist()) <= target_size:  
            current_chunks +=' '+sentence  
        else:  
            merged_chunks.append(current_chunks.strip())  
            current_chunks = sentence  
            
    if current_chunks:  
        merged_chunks.append(current_chunks.strip()) 
        
    for chunk in tqdm(merged_chunks,desc='Proposition Generating',leave=False):
        input_text = f"Title: {title}. Section: {section}. Content: {chunk}"
        input_ids = tokenizer(input_text, return_tensors="pt")['input_ids'].to(model.device)
        outputs = model.generate(input_ids, max_new_tokens=512).cpu()
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        while use_count < limit_count:
            try:
                prop_list = json.loads(output_text)
                propositions.extend(prop_list)
                break
            except Exception as e:
                use_count+=1
                logger.info(f"[ERROR] Failed to parse output text as JSON. Error message: {e}")
                
        else:
            raise ValueError(f"The model has failed {limit_count} times. Please check the model and input text. Input text: {text}")
    return propositions

