import torch
from torch import nn

from models.multi_head_attention import MultiHeadAttention
from models.feed_forward_network import FFN
from models.embedding import GPTEmbedding

"""
Single block in the GPT-2 decoder.

Each block consists of:
    - A self-attention mechanism (decoder-only attention) with residual connections and LayerNorm.
    - A feed-forward network (FFN) with residual connections and LayerNorm.
    - A dropout for each residuals(attention context and FFN)

Args:
    d_model (int): Dimensionality of the input embeddings.
    d_ffn (int): Dimensionality of the hidden layer in the feed-forward network.
    num_heads (int): Number of attention heads in the multi-head attention mechanisms.
    p_dropout (float): Dropout probability for regularization.
"""
class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()

        # Decoder Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(p=p_dropout)

        # Feed-Forward Network
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ffn, p_dropout)
        self.dropout2 = nn.Dropout(p=p_dropout)

    def forward(self, x, self_attn_mask=None):
        """
        Args:
            x (torch.Tensor): Target sequence tensor of shape (batch_size, tgt_seq_len, d_model).
            self_attn_mask (torch.Tensor): Source mask of shape (batch_size, 1, src_seq_len, src_seq_len).
        Returns:
            torch.Tensor: Transformed output tensor of shape (batch_size, seq_len, d_model).
            torch.Tensor: Self-attention scores for the sequence.
        """
        # Self-Attention with residual connection and normalization (Note: Normalize First!)
        normed_x = self.norm1(x)
        attn_context, attn_score = self.self_attention(normed_x, normed_x, normed_x, mask=self_attn_mask)
        attn_context = self.dropout1(attn_context)
        x = x + attn_context

         # Feed-forward network with residual connection and normalization (Note: Normalize First!)
        normed_x = self.norm2(x)
        residual = self.ffn(normed_x)
        residual = self.dropout2(residual)
        x = x + residual

        return x, attn_score



"""
Represents the full GPT-2 decoder consisting of multiple decoder blocks.

The decoder applies:
    - Positional encoding to the target input embeddings.
    - Multiple decoder blocks, each with self-attention, cross-attention, and feed-forward layers.
    - A final linear layer to project the output to the vocabulary size for token predictions.

Args:
    dec_vocab_size (int): Size of the target vocabulary.
    max_len (int): Maximum sequence length for positional encoding.
    num_blocks (int): Number of decoder blocks in the decoder.
"""
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()

        # Create position-encoded embedding
        self.input_emb = GPTEmbedding(vocab_size, d_model, max_len, p_dropout)
        self.dropout = nn.Dropout(p=p_dropout)
        
        # Create decoder blocks
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, d_ffn, num_heads, p_dropout)
                                         for _ in range(num_blocks)])

        # Create last FC Layer with normalization (Note: Normalization First!)
        self.norm_out = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, self_attn_mask=None, save_attn_pattern=False):
        """
        Args:
            x (torch.Tensor): Sequence tensor of shape (batch_size, tgt_seq_len).
            self_attn_mask (torch.Tensor): Source mask to prevent attention to specific positions.
            save_attn_pattern (bool): If True, saves and returns attention patterns for visualization.
        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size).
            torch.Tensor: (Optional) Self-attention patterns from all decoder blocks.
        """
        x = self.input_emb(x)
        x = self.dropout(x)

        attn_patterns = torch.tensor([]).to(DEVICE)
        for block in self.dec_blocks:
            x, attn_pattern = block(x, self_attn_mask)
            # (Optional) if save_attn_pattern is True, save these and return for visualization/investigation
            if save_attn_pattern:
                attn_patterns = torch.cat([attn_patterns, attn_pattern[0].unsqueeze(0)], dim=0)

        x = self.norm_out(x)
        x = self.fc_out(x)

        return x, attn_patterns