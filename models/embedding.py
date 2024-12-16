import torch
from torch import nn


"""
GPTEmbedding combines token embeddings with positional embeddings(learnabel)
to produce input embeddings for a Transformer model. It also applies dropout
for regularization.

Args:
    vocab_size (int): Size of the vocabulary.
    max_len (int): Maximum sequence length for positional encoding.
    d_model (int): Dimensionality of the embeddings.
    p_dropout (float): Dropout probability.
"""
class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, p_dropout):
        super().__init__()

        # Token Embedding
        self.token_embedding= nn.Embedding(vocab_size, d_model)

        # Positional Embedding (learnable)
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len) containing token indices.
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, hidden_size)
                       Summation of token and positional embeddings.
        """
        batch_size, seq_len = src.size()

        # Create pos_ids from [0, ..., (seq_len - 1)]
        pos_ids = torch.arange(seq_len).expand_as(src).to(DEVICE) # pos_ids : [batch_size, seq_len]

        # Token embeddings: (batch_size, seq_len, d_model)
        token_embedding = self.token_embedding(src)
        # Positional embeddings: (batch_size, seq_len, d_model)
        pos_embedding = self.pos_embedding(pos_ids)
         # Combine them by summation
        embeddings = token_embedding + pos_embedding

        return embeddings