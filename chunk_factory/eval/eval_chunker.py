'''
Description:  
Author: Huang J
Date: 2025-04-08 10:04:16
'''

from typing import List,TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from sentence_transformers import SentenceTransformer

class EvalChunker:
    """
    Evaluation class for a list of text chunks, using Boundary Clarity, Chunk Stickiness, and semantic irrelevance metrics.
    """
    def __init__(self, chunks: List[str]):
        """
        Initializes the EvalChunker with a list of text chunks.

        Args:
            chunks (List[str]): A list of text segments to evaluate.
        """
        self.chunks = chunks
    
    def bc_eval(
        self,
        model: 'PreTrainedModel',
        tokenizer: 'PreTrainedTokenizer'
    ) -> float:
        """
        Computes the Boundary Clarity metric, which measures the distinctiveness between adjacent chunks.

        Args:
            model (PreTrainedModel): Pretrained language model.
            tokenizer (PreTrainedTokenizer): Tokenizer associated with the model.

        Returns:
            float: The Boundary Clarity score.
        """
        from chunk_factory.utils.metric import bc_calculate
        bc_ppl_scores = []
        for i in tqdm(range(len(self.chunks)-1),desc='BC Eval',leave=False):
            text1 = self.chunks[i]
            text2 = self.chunks[i+1]
            ppl_value = bc_calculate(pre_text=text1,next_text=text2,model=model,tokenizer=tokenizer)
            bc_ppl_scores.append(ppl_value)
        avg_bc_score = sum(bc_ppl_scores)/len(bc_ppl_scores)
        return avg_bc_score
    
    def cs_eval(
        self,
        model: 'PreTrainedModel',
        tokenizer: 'PreTrainedTokenizer',
        delta: float = 0.0
        ) -> float:
        """
        Calculates the Chunk Stickiness score, indicating how contextually connected adjacent chunks are.

        Args:
            model (PreTrainedModel): Pretrained language model.
            tokenizer (PreTrainedTokenizer): Tokenizer associated with the model.
            delta (float): Sensitivity adjustment for the score. Default is 0.0.

        Returns:
            float: The Chunk Stickiness score.
        """
        from chunk_factory.utils.metric import cs_calculate
        cs_score = cs_calculate(chunks=self.chunks,model=model,tokenizer=tokenizer,delta=delta)
        return cs_score
    
    def semantic_eval(
        self,
        encode_model: 'SentenceTransformer'
    ) -> float:
        """
        Calculates the average semantic dissimilarity between adjacent text chunks using a pre-trained 
        SentenceTransformer model.

        Args:
            encode_model (SentenceTransformer): The pre-trained SentenceTransformer model used to encode the chunks.
                This model is responsible for computing sentence embeddings that are used to measure the semantic 
                similarity between adjacent chunks.

        Returns:
            float: The average semantic dissimilarity between adjacent chunks. The value will be between 0 and 1,
                where a higher value indicates that adjacent chunks are more dissimilar.
        """
        from chunk_factory.utils.metric import get_semantic_similarity
        semantic_differences = []
        for i in tqdm(range(len(self.chunks)-1),desc="Semantic Eval",leave=False):
            text1 = self.chunks[i]
            text2 = self.chunks[i+1]
            similarity = get_semantic_similarity(text1=text1,text2=text2,encode_model=encode_model)
            dissimilarity = 1-similarity
            semantic_differences.append(dissimilarity)
        avg_dissimilarity = sum(semantic_differences)/len(semantic_differences)
        return avg_dissimilarity