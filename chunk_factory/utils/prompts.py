'''
Description:  
Author: Huang J
Date: 2025-03-31 10:57:13
'''

import logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


# 要求python 3.10
def lumberchunker_prompt(language: str) -> str:
    """
    Returns the prompt for detecting the first content shift paragraph in a document.
    
    Args:
        language (str): Language of the document. Supports 'en' (English) and 'zh' (Chinese).
        
    Returns:
        str: The corresponding prompt text.
        
    Raises:
        ValueError: If the language is not supported.
    """
    match language:
        case 'en':
            prompt = """You will receive as input an english document with paragraphs identified by 'ID XXXX: <text>'.

Task: Find the first paragraph (not the first one) where the content clearly changes compared to the previous paragraphs.

Output: Return the ID of the paragraph with the content shift as in the exemplified format: 'Answer: ID XXXX'.

Additional Considerations: Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable."""
            return prompt
        case 'zh':
            prompt = """你将收到一个中文文档，其中段落用'ID XXXX: <text>'标识。

任务：找出第一个段落（不是第一个），其中内容与前面的段落相比较，显着地发生变化。

输出：以如下格式返回段落的ID：'Answer: ID XXXX'。

其他注意事项：避免形成过长的段落集群。考虑句子逻辑和语义内容。寻找一个平衡，以识别内容变化并保持段落集群长度适中。"""
            return prompt
        case _:
            logger.info(f'language: {language}')
            raise ValueError("Unsupported language. Please use 'zh' for Chinese or 'en' for English.")
        
        
def msp_lumberchunker_prompt(language: str) -> str:
    """
    Returns a refined version of the content shift detection prompt for each paragraph in the document.
    
    Args:
        language (str): Language of the document. Supports 'en' (English) and 'zh' (Chinese).
        
    Returns:
        str: The corresponding prompt text.
        
    Raises:
        ValueError: If the language is not supported.
    """
    match language:
        case 'en':
            prompt = """You will receive a document as input, where each paragraph is identified by "<ID>: <text>".

Task: Find the first paragraph (not the first one) that shows a clear change in content compared to the previous paragraphs.

Output: Directly return the ID of the paragraph where the content changes.

Other notes: Avoid forming excessively long paragraph clusters. Comprehensively consider the logical structure and semantic content of sentences. Seek a good balance between identifying content changes and maintaining manageable paragraph clusters."""
            return prompt
        case 'zh':
            prompt = """你将会接收到一个文档作为输入，其中各段落通过“<ID>：<文本>”进行标识。

任务：找到第一个（非首个）与前一段落内容相比发生明显变化的段落。

输出：直接返回内容发生变化的段落的ID。

其他注意事项：避免形成过长的段落群。综合考虑句子的逻辑结构和语义内容。要在识别内容变化和保持段落群的可管理性之间寻求良好的平衡。"""
            return prompt
        case _:
            logger.info(f'language: {language}')
            raise ValueError("Unsupported language. Please use 'zh' for Chinese or 'en' for English.")
