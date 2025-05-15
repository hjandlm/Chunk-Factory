'''
Description:  
Author: Huang J
Date: 2025-04-28 09:14:00
'''

import re
from typing import List

def segmentchunker(
    text: str,
    language: str,
    seg_size: int = 200,
    seg_overlap: int = 1,
    separators: List[str] = None
) -> List[str]:
    """
    Segment Chunking with Custom Separators and Overlap Support
    
    Parameters:
    text: The original text to be split.
    language: Language type ('zh' for Chinese / 'en' for English).
    seg_size: Target chunk size (measured in characters for Chinese, words for English).
    seg_overlap: Number of overlapping chunks between segments.
    separators: Custom list of separators.

    Returns:
    List[str]: A list of segmented text chunks.
    """

    if seg_size <= 0:
        raise ValueError("seg_size must be greater than 0")
    if seg_overlap < 0:
        raise ValueError("seg_overlap cannot be negative")

    text = text.replace('\n','')

    lang_config = {
        'zh': {
            'separators': ['\n\n', '\n', '。', '！', '？', '；', '，', '、'],
            'split_space': False
        },
        'en': {
            'separators': ['\n\n', '\n', '.', '!', '?', ';', ','],
            'split_space': True
        }
    }
    lang = lang_config[language]
    separators = separators or lang['separators']
    sep_pattern = r'(' + r'|'.join(map(re.escape, separators)) + r')'
    text_pattern = r'[^' + ''.join(map(re.escape, separators)) + r']+'  # 匹配非分隔符的部分

    text_list_re = re.findall(text_pattern, text)
    if text_list_re:
        text_list = [t.strip() for t in text_list_re]
    else:
        return []
    separator_list = re.findall(sep_pattern, text)

    chunks = []
    
    current_chunk = ''
    pre_chunk = ''
    end_flag = False
    if language=='zh':
        
        for index,text in enumerate(text_list):
            current_chunk+=text
            current_chunk+=separator_list[index]
            try:
                pre_chunk = current_chunk+text_list[index+1]
                pre_chunk = pre_chunk+separator_list[index+1]
            except Exception as e:
                chunks.append(current_chunk)
                current_chunk = ''
                break
            
            if len(pre_chunk)>seg_size:
                if seg_overlap>0:
                    for oi in range(seg_overlap):
                        try:
                            next_text = text_list[index+oi+1]
                            current_chunk+=next_text
                            current_chunk+=separator_list[index+oi+1]
                        except Exception as e:
                            end_flag = True
                        if end_flag:
                            chunks.append(current_chunk)
                            current_chunk = ''
                            break
                    if not end_flag:
                        chunks.append(current_chunk)  
                        current_chunk = ''  
                else:
                    chunks.append(current_chunk)
                    current_chunk = ''   
    elif language=='en':
        for index,text in enumerate(text_list):
            if len(current_chunk)==0:
                current_chunk = text
                current_chunk+=separator_list[index]
            else:
                current_chunk = current_chunk+' '+text
                current_chunk+=separator_list[index]
                
            try:
                pre_chunk = current_chunk+' '+text_list[index+1]
                pre_chunk = pre_chunk+separator_list[index+1]
            except Exception as e:
                chunks.append(current_chunk)
                current_chunk = ''
                break
            if len(pre_chunk.split(' '))>seg_size:
                if seg_overlap>0:
                    for oi in range(seg_overlap):
                        try:
                            next_text = text_list[index+oi+1]
                            current_chunk = current_chunk+' '+next_text
                            current_chunk+=separator_list[index+oi+1]
                        except Exception as e:
                            end_flag = True
                        if end_flag:
                            chunks.append(current_chunk)
                            current_chunk = ''
                            break
                    if not end_flag:
                        chunks.append(current_chunk)  
                        current_chunk = ''  
                else:
                    chunks.append(current_chunk)
                    current_chunk = ''

    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks
            