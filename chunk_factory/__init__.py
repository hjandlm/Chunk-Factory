'''
Description:  
Author: Huang J
Date: 2025-04-07 22:08:25
'''



def Chunker(*args, **kwargs):
    from .chunker import Chunker
    return Chunker(*args, **kwargs)


def EvalChunker(*args, **kwargs):
    from chunk_factory.eval import EvalChunker
    return EvalChunker(*args, **kwargs)

