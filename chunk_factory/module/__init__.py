'''
Description:  
Author: Huang J
Date: 2025-03-31 15:03:03
'''

def basechunker(*args, **kwargs):
    from .baseChunker import basechunker
    return basechunker(*args, **kwargs)

def segment_chunker(*args, **kwargs):
    from .segmentChunker import segmentchunker
    return segmentchunker(*args, **kwargs)

def denseX_chunker(*args, **kwargs):
    from .densexChunker import denseX_chunker
    return denseX_chunker(*args, **kwargs)

def lumberchunker(*args, **kwargs):
    from .lumberChunker import lumberchunker
    return lumberchunker(*args, **kwargs)

def msp_chunker(*args, **kwargs):
    from .mspChunker import msp_chunker
    return msp_chunker(*args, **kwargs)

def ppl_chunker(*args, **kwargs):
    from .pplChunker import ppl_chunker
    return ppl_chunker(*args, **kwargs)
