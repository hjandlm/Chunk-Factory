'''
Description:  
Author: Huang J
Date: 2025-03-31 15:02:24
'''

# from .metric import get_semantic_similarity,bc_calculate,cs_calculate
# from .prompts import lumberchunker_prompt,msp_lumberchunker_prompt
# from .util import count_words,add_ids,split_text_by_punctuation,match_answer_id,show_figure,get_min_ppl,get_token_loss,build_graph,calculate_structural_entropy,find_values_greater_than,get_ppl_next

# llm
def llm_response_api(*args, **kwargs):
    from .llm import llm_response_api
    return llm_response_api(*args, **kwargs)

# metric
def get_semantic_similarity(*args, **kwargs):
    from .metric import get_semantic_similarity
    return get_semantic_similarity(*args, **kwargs)

def bc_calculate(*args, **kwargs):
    from .metric import bc_calculate
    return bc_calculate(*args, **kwargs)

def cs_calculate(*args, **kwargs):
    from .metric import cs_calculate
    return cs_calculate(*args, **kwargs)

# prompts
def lumberchunker_prompt(*args, **kwargs):
    from .prompts import lumberchunker_prompt
    return lumberchunker_prompt(*args, **kwargs)

def msp_lumberchunker_prompt(*args, **kwargs):
    from .prompts import msp_lumberchunker_prompt
    return msp_lumberchunker_prompt(*args, **kwargs)

# util
def count_words(*args, **kwargs):
    from .util import count_words
    return count_words(*args, **kwargs)

def add_ids(*args, **kwargs):
    from .util import add_ids
    return add_ids(*args, **kwargs)

def split_text_by_punctuation(*args, **kwargs):
    from .util import split_text_by_punctuation
    return split_text_by_punctuation(*args, **kwargs)

def match_answer_id(*args, **kwargs):
    from .util import match_answer_id
    return match_answer_id(*args, **kwargs)

def show_figure(*args, **kwargs):
    from .util import show_figure
    return show_figure(*args, **kwargs)

def get_min_ppl(*args, **kwargs):
    from .util import get_min_ppl
    return get_min_ppl(*args, **kwargs)

def get_token_loss(*args, **kwargs):
    from .util import get_token_loss
    return get_token_loss(*args, **kwargs)

def build_graph(*args, **kwargs):
    from .util import build_graph
    return build_graph(*args, **kwargs)

def calculate_structural_entropy(*args, **kwargs):
    from .util import calculate_structural_entropy
    return calculate_structural_entropy(*args, **kwargs)

def find_values_greater_than(*args, **kwargs):
    from .util import find_values_greater_than
    return find_values_greater_than(*args, **kwargs)

def get_ppl_next(*args, **kwargs):
    from .util import get_ppl_next
    return get_ppl_next(*args, **kwargs)


