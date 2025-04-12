'''
Description:  
Author: Huang J
Date: 2025-04-08 21:12:29
'''

from typing import List, Optional,TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def basechunker(
    text: str,
    language: Optional[str] = None,
    use_token: bool = False,
    tokenizer: Optional["PreTrainedTokenizer"] = None,
    chunk_size: int = 256,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Splits input text into smaller chunks based on the specified language, tokenizer, and chunking parameters.

    Args:
        text (str): The input text to be split into chunks.
        language (Optional[str]): The language of the text ('en' for English, 'zh' for Chinese). Default is None.
        use_token (bool): Whether to tokenize the text or not. If False, splits based on characters or spaces. Default is False.
        tokenizer (Optional[PreTrainedTokenizer]): A pretrained tokenizer to tokenize the text if `use_token` is True. Default is None.
        chunk_size (int): The maximum size of each chunk. Default is 256.
        chunk_overlap (int): The number of overlapping tokens/characters between consecutive chunks. Default is 50.

    Returns:
        List[str]: A list of text chunks.
    
    Raises:
        ValueError: If the chunk_overlap is greater than chunk_size, or if the parameters are incorrect.
    """

    if chunk_overlap > chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
            f"({chunk_size}), should be smaller."
        )
        
    text_chunks = []
    
    if not use_token and language:
        match language:
            case 'zh':
                for i in tqdm(range(0,len(text),chunk_size-chunk_overlap),desc='Text Chunking',leave=False):
                    text_chunks.append(text[i:i+chunk_size])
                return text_chunks
            
            case 'en':
                text_split = text.split(' ')
                for i in tqdm(range(0,len(text_split),chunk_size-chunk_overlap),desc='Text Chunking',leave=False):
                    text_tmp = ' '.join(text_split[i:i+chunk_size])
                    text_chunks.append(text_tmp)
                return text_chunks
            
            case _:
                raise ValueError('Check the value of the `language` parameter. The valid options for `language` are `[en, zh]`.')
            
    if not use_token and not language:
        raise ValueError('Please check the `language` parameter or set `use_token` to `False`.')
    
    if tokenizer:
        token_list = tokenizer.tokenize(text)
        for i in tqdm(range(0,len(token_list),chunk_size-chunk_overlap),desc='Text Chunking',leave=False):
            text_tmp = ' '.join(token_list[i:i+chunk_size])
            if language=='zh':
                text_tmp = text_tmp.replace(' ','')
            text_chunks.append(text_tmp)
        return text_chunks
    
    import tiktoken
    tokenize = tiktoken.encoding_for_model('gpt-4')
    text_ids = tokenize.encode(text)
    for i in tqdm(range(0,len(text_ids),chunk_size-chunk_overlap),desc='Text Chunking',leave=False):
        text_ids_tmp = text_ids[i:i+chunk_size]
        text_tmp = ''.join(tokenize.decode(text_ids_tmp))
        if language=='zh':
            text_tmp = text_tmp.replace(' ','')

        text_chunks.append(text_tmp)
    return text_chunks
        


    

