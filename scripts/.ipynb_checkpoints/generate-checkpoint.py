import torch
from torch import nn

def generate(model, text, tokenizer, save_attn_pattern=False):

    ''' Step 1: Tokenize the input sequence'''
    # sequence Shape: (1, input_seq_len)
    sequence = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False).to(DEVICE)
    max_iter = MAX_LEN - sequence.shape[1]

    ''' Step 2: Auto-regressive next-token prediction through decoder '''
    model.eval() 
    with torch.no_grad():
        for _ in range(max_iter):
            decoder_out, attn_pattern = model.decoder(sequence, None, save_attn_pattern)
            next_token_predicted = decoder_out[:, -1, :].argmax(dim=-1).unsqueeze(0)
            # New sequence created by concatenating the new predicted token
            sequence = torch.cat((sequence, next_token_predicted), dim=-1)

            if tokenizer.decode(next_token_predicted.item()) == '[SEP]':
                break

        output_text = tokenizer.decode(sequence[0])

    return output_text, attn_pattern