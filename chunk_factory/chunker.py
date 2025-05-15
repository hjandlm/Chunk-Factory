'''
Description:  
Author: Huang J
Date: 2025-04-07 22:09:54
'''


import logging
from typing import Optional, List
# from chunk_factory.module import denseX_chunker,lumberchunker,ppl_chunker,msp_chunker,basechunker

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)

class Chunker:
    """
    A class to split input text into semantically meaningful chunks using various chunking methods.
    
    The class provides methods to perform chunking based on different algorithms, such as:
    - Base chunking
    - DenseX chunking
    - Lumberchunking
    - Perplexity-based chunking
    - MSP-based chunking

    Attributes:
        text (str): The input text to be chunked.
        language (str): The language of the input text (e.g., 'zh' for Chinese, 'en' for English).
    """

    def __init__(
        self, 
        text: str, 
        language: str
    ) -> None:
        """
        Initializes the Chunker class with text and language.
        
        Args:
            text (str): The input text to be chunked.
            language (str): The language of the text.
        """
        self.text = text
        self.language = language
        if not language or language not in ['zh','en']:
            raise Exception("Please check the language parameter. Valid options are: 'zh', 'en'")
        if not text:
            raise Exception("The `text` parameter cannot be empty. Please provide input text.")

    def basechunk(
        self, 
        use_token: bool = False, 
        tokenizer: Optional['PreTrainedTokenizer'] = None, 
        chunk_size: int = 256, 
        chunk_overlap: int = 50
    ) -> None:
        """
        Splits the input text into chunks using the base chunking method.
        
        Args:
            use_token (bool, optional): Whether to use tokenization. Defaults to False.
            tokenizer (Optional[object], optional): The tokenizer to use if tokenization is enabled. Defaults to None.
            chunk_size (int, optional): The maximum size of each chunk. Defaults to 256.
            chunk_overlap (int, optional): The number of overlapping tokens/characters. Defaults to 50.
            
        Returns:
            List[str]: A list of text chunks.
        """
        from chunk_factory.module import basechunker
        text_chunks = basechunker(
            text=self.text, 
            language=self.language, 
            use_token=use_token,
            tokenizer=tokenizer, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_chunks

    def segment_chunk(
        self,
        seg_size: int = 200,
        seg_overlap: int = 5,
        separators: List[str] = None
        ):
        from chunk_factory.module import segment_chunker
        text_segs = segment_chunker(
            text=self.text,
            language=self.language,
            seg_size=seg_size,
            seg_overlap=seg_overlap,
            separators=separators
        )
        return text_segs
    
    def denseX_chunk(
        self, 
        model: object, 
        tokenizer: object, 
        title: str = '', 
        section: str = '', 
        target_size: int = 256, 
        limit_count: int = 5
    ) -> List[dict]:
        """
        Splits the input text into chunks using the denseX chunking method.
        
        Args:
            model (object): The language model to use for chunking.
            tokenizer (object): The tokenizer corresponding to the model.
            title (str, optional): The title for the document. Defaults to ''.
            section (str, optional): The section for the document. Defaults to ''.
            target_size (int, optional): The maximum size of each chunk. Defaults to 256.
            limit_count (int, optional): The maximum number of retries if JSON parsing fails. Defaults to 5.
        
        Returns:
            List[dict]: A list of propositions.
        """
        from chunk_factory.module import denseX_chunker
        propositions = denseX_chunker(
            text=self.text, 
            model=model, 
            tokenizer=tokenizer, 
            language=self.language, 
            title=title, 
            section=section, 
            target_size=target_size, 
            limit_count=limit_count
        )
        return propositions
    
    def lumberchunk(
        self, 
        model_type: Optional[str] = None, 
        model_name: Optional[str] = None, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        system_prompt: Optional[str] = None, 
        use_msp: bool = False, 
        small_model: Optional[object] = None, 
        small_tokenizer: Optional[object] = None
    ) -> List[str]:
        """
        Splits the input text into chunks using the lumberchunk method.
        
        Args:
            model_type (Optional[str], optional): The model type (e.g., 'ChatGPT', 'Gmini'). Defaults to None.
            model_name (Optional[str], optional): The model name. Defaults to None.
            api_key (Optional[str], optional): API key for model access. Defaults to None.
            base_url (Optional[str], optional): Optional base URL for the LLM API. Defaults to None.
            system_prompt (Optional[str], optional): Optional prompt for guiding model behavior. Defaults to None.
            use_msp (bool, optional): Whether to use a small model for chunk selection. Defaults to False.
            small_model (Optional[object], optional): The small model instance. Defaults to None.
            small_tokenizer (Optional[object], optional): Tokenizer for the small model. Defaults to None.
        
        Returns:
            List[str]: A list of text chunks.
        """
        from chunk_factory.module import lumberchunker
        text_chunks = lumberchunker(
            text=self.text, 
            language=self.language, 
            model_type=model_type,
            model_name=model_name, 
            api_key=api_key, 
            base_url=base_url, 
            system_prompt=system_prompt, 
            use_msp=use_msp, 
            small_model=small_model, 
            small_tokenizer=small_tokenizer
        )
        return text_chunks
    
    def ppl_chunk(
        self, 
        model: object, 
        tokenizer: object, 
        threshold: float, 
        max_length: int = 2048, 
        model_length: int = 8096, 
        show_ppl_figure: bool = False, 
        save_dir: str = ''
    ) -> List[str]:
        """
        Splits the input text into chunks using the perplexity-based chunking method.
        
        Args:
            model (object): The language model to use for perplexity evaluation.
            tokenizer (object): The tokenizer corresponding to the model.
            threshold (float): The threshold for detecting chunk boundaries.
            max_length (int, optional): The maximum length per forward pass. Defaults to 2048.
            model_length (int, optional): The maximum length supported by the model. Defaults to 8096.
            show_ppl_figure (bool, optional): Whether to display a perplexity figure. Defaults to False.
            save_dir (str, optional): The directory to save the figure. Defaults to ''.
        
        Returns:
            List[str]: A list of text chunks.
        """
        from chunk_factory.module import ppl_chunker
        text_chunks = ppl_chunker(
            text=self.text, 
            language=self.language, 
            model=model, 
            tokenizer=tokenizer, 
            threshold=threshold, 
            max_length=max_length, 
            model_length=model_length, 
            show_ppl_figure=show_ppl_figure, 
            save_dir=save_dir
        )
        return text_chunks
    
    def msp_chunk(
        self, 
        model: object, 
        tokenizer: object, 
        threshold: float
    ) -> List[str]:
        """
        Splits the input text into chunks using the MSP-based chunking method.
        
        Args:
            model (object): The language model to use for chunking.
            tokenizer (object): The tokenizer corresponding to the model.
            threshold (float): The threshold for detecting chunk boundaries.
        
        Returns:
            List[str]: A list of text chunks.
        """
        from chunk_factory.module import msp_chunker
        text_chunks = msp_chunker(
            text=self.text, 
            language=self.language, 
            model=model, 
            tokenizer=tokenizer, 
            threshold=threshold
        )
        return text_chunks
    
    



    
