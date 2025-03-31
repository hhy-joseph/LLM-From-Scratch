import numpy as np

class Tokenizer:
    def __init__(self, max_vocab_size=10000):
        """
        Initialize a simple tokenizer with vocabulary building capabilities
        
        Args:
            max_vocab_size (int): Maximum number of unique tokens to keep
        """
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.sos_token = '<SOS>'  # Start of Sequence
        self.eos_token = '<EOS>'  # End of Sequence
        self._add_special_tokens()
    
    def _add_to_vocab(self, word):
        """Add a word to the vocabulary"""
        if word not in self.word_to_index:
            # If we still can add vocab
            if self.vocab_size < self.max_vocab_size:
                self.word_to_index[word] = self.vocab_size
                self.index_to_word[self.vocab_size] = word
                self.vocab_size += 1
                # Return the latest index
                return self.word_to_index[word]
        # If the word already in the index of word if there 
        # OR max_vocab reached, return UNKNOWN Token
        return self.word_to_index.get(word, self.word_to_index[self.unk_token])

    def _add_special_tokens(self):
        """Add special tokens to the vocabulary"""
        self._add_to_vocab(self.pad_token)
        self._add_to_vocab(self.unk_token)
        self._add_to_vocab(self.sos_token)
        self._add_to_vocab(self.eos_token)

    def fit_on_texts(self, texts):
        """
        Build vocabulary from input texts
        
        Args:
            texts (list): List of sentences or documents
        """
        # Flatten and tokenize texts; split by space for simplification
        # each item is a word
        all_words = [word for sentence in texts for word in sentence.split()]
        
        # Count word frequencies
        # Ensures most common words are included first
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort words by frequency, descending order
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add most frequent words to vocabulary
        for word, _ in sorted_words:
            self._add_to_vocab(word)

    def texts_to_sequences(self, texts, add_sos_eos=True):
        """
        Convert texts to sequences of indices
        
        Args:
            texts (list): List of sentences or documents
            add_sos_eos (bool): Whether to add Start and End of Sequence tokens
        
        Returns:
            list: List of token indices
        """
        sequences = []
        for text in texts:
            # Tokenize the text
            tokens = text.split()
            
            # Add special tokens if requested
            if add_sos_eos:
                tokens = [self.sos_token] + tokens + [self.eos_token]
            
            # Convert to indices
            sequence = [self.word_to_index.get(word, self.word_to_index[self.unk_token]) 
                        for word in tokens]
            sequences.append(sequence)
        
        return sequences
    def pad_sequences(self, sequences: list, max_length: int = None, padding: str = 'post', 
                    truncating: str = 'post') -> np.ndarray:
        """
        Pad sequences to the same length and return as numpy array
        
        Args:
            sequences (list): List of token sequences
            max_length (int): Maximum length to pad sequences to.
                            If None, uses the length of the longest sequence.
            padding (str): 'pre' or 'post' - where to add padding
            truncating (str): 'pre' or 'post' - where to truncate if needed
        
        Returns:
            np.ndarray: Numpy array of padded sequences
        """
        import numpy as np
        
        # Find the max length if not provided
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        # Pad token index
        pad_id = self.word_to_index[self.pad_token]
        
        # Initialize padded sequences as numpy array
        padded_sequences = np.full((len(sequences), max_length), pad_id, dtype=np.int32)
        
        for i, sequence in enumerate(sequences):
            # Truncate if necessary
            if len(sequence) > max_length:
                if truncating == 'pre':
                    sequence = sequence[-max_length:]
                else:  # truncating == 'post'
                    sequence = sequence[:max_length]
            
            # Add sequence to padded array
            if padding == 'pre':
                padded_sequences[i, -len(sequence):] = sequence
            else:  # padding == 'post'
                padded_sequences[i, :len(sequence)] = sequence
        
        return padded_sequences