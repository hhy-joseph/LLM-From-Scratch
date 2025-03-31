from typing import Dict, List, Union, Tuple, Optional, Any
import numpy as np
class Embedding:
    """
    A custom embedding layer that converts token indices to vector representations
    """
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        """
        Initialize embedding layer with random weights
        
        Args:
            vocab_size (int): Size of vocabulary (number of unique tokens)
            embedding_dim (int): Dimension of embedding vectors
            padding_idx (Optional[int]): Index of padding token to zero out
        """
        # Initialize embeddings with small random values
        self.weights: np.ndarray = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
        
        # If padding index provided, set those embeddings to zero
        if padding_idx is not None:
            self.weights[padding_idx] = np.zeros(embedding_dim)
        
        self.vocab_size: int = vocab_size
        self.embedding_dim: int = embedding_dim
    
    def forward(self, indices: Union[int, List[int], List[List[int]], np.ndarray]) -> np.ndarray:
        """
        Convert token indices to embeddings
        
        Args:
            indices: Token indices, can be a single index, a sequence (list/array),
                     or a batch of sequences (2D list/array)
            
        Returns:
            np.ndarray: Corresponding embedding vectors
        """
        # Handle single index
        if isinstance(indices, int):
            return self.weights[indices]
        
        # Handle sequence (list or 1D array)
        if isinstance(indices, list) and (not indices or not isinstance(indices[0], list)):
            return np.array([self.weights[idx] for idx in indices])
        elif isinstance(indices, np.ndarray) and indices.ndim == 1:
            return self.weights[indices]
        
        # Handle batch of sequences (2D list or 2D array)
        # This is where the error occurred - handling sequences of different lengths
        if isinstance(indices, list):
            # For batch processing of different length sequences, we'll use a different approach
            # Process each sequence separately and return them as a list
            return [np.array([self.weights[idx] for idx in seq]) for seq in indices]
        elif isinstance(indices, np.ndarray) and indices.ndim == 2:
            batch_size, seq_length = indices.shape
            result = np.zeros((batch_size, seq_length, self.embedding_dim))
            for i in range(batch_size):
                result[i] = self.weights[indices[i]]
            return result
        
        raise ValueError(f"Unsupported indices type or shape: {type(indices)}")
    
    def __call__(self, indices: Union[int, List[int], List[List[int]], np.ndarray]) -> np.ndarray:
        """
        Make the class callable for convenience
        """
        return self.forward(indices)