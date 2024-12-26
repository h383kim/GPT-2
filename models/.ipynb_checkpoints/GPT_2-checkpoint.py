import torch
from torch import nn

from models.decoder import Decoder


"""
GPT-2 model for Next-Token_Prediction tasks, consisting of decoder only from Transformer.

Args:
    pad_idx (int): Index of the <PAD> token in the vocabulary.
    evocab_size (int): Size of the vocabulary.
    max_len (int): Maximum length of sequences.
    num_blocks (int): Number of decoder blocks.
    d_model (int): Dimensionality of input embeddings and model representations.
    d_ffn (int): Dimensionality of the feed-forward network's hidden layer.
    num_heads (int): Number of attention heads in multi-head attention.
    p_dropout (float): Dropout probability for regularization.
       
Helper Methods:
    _make_mask(src): Generates a combined padding and future mask for the decoder's self-attention.
"""
class GPT2(nn.Module):
    def __init__(self, pad_idx, vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout):
        super().__init__()

        self.pad_idx = pad_idx
        self.num_heads = num_heads

        # Initialize decoder
        self.decoder = Decoder(vocab_size, max_len, num_blocks, d_model, d_ffn, num_heads, p_dropout)


    def forward(self, x, save_attn_pattern=False):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        Returns:
            tuple: A tuple containing:
                - decoder_out (torch.Tensor): Decoder output logits of shape (batch_size, seq_len, vocab_size).
                - attn_patterns (torch.Tensor): Self-attention patterns from the decoder.
        """
        # Create masking
        mask = self._make_mask(x, self.pad_idx)

        # Decoder pass
        decoder_out, attn_patterns = self.decoder(x, mask, save_attn_pattern)

        return decoder_out, attn_patterns


    def _make_mask(self, x, pad_idx):
        """
        Creates a combined padding and future mask for the decoder.
        Args:
            x (torch.Tensor): Target tensor of shape (batch_size, seq_len).
        Returns:
            pad_future_mask (torch.Tensor): Combined mask of shape (batch_size, num_heads, seq_len, seq_len), 
                                            where True indicates positions to mask (padding or future tokens).
        Steps:
            1. **Padding Masking**: Identify positions corresponding to <PAD> tokens.
            2. **Future Masking**: Mask future tokens using a upper triangular matrix.
            3. **Combine Masks**: Apply logical OR to combine both masks.
        Example:
        (PAD MASK)
            Initial pad_mask for a single sequence (sentence): 
            [F F F T T] (F = False, T = True for <PAD>)

            Expanded across heads and queries(columns):
            [F F F T T]
            [F F F T T]
            [F F F T T]  x num_heads
            [F F F T T]
            [F F F T T]

        (Future MASK)
            [F T T T T]
            [F F T T T]
            [F F F T T]
            [F F F F T]
            [F F F F F]
        """
        # Pad Masking
        pad_mask = (x == pad_idx)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        pad_mask = pad_mask.expand(x.shape[0], self.num_heads, x.shape[1], x.shape[1])

        # Future Masking
        upper_trig_mask = torch.tril(torch.ones(x.shape[0], self.num_heads, x.shape[1], x.shape[1]))
        upper_trig_mask = (upper_trig_mask == 0).to(DEVICE)

        # Combining both Maskings
        pad_future_mask = pad_mask | upper_trig_mask

        return pad_future_mask