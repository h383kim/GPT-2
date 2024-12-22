import torch
from torch import nn

def epoch_train(model, train_dataloader, loss_fn, optimizer):
    model.train()
    train_loss = 0.0
    
    for texts in train_dataloader:
        texts = [text + ' [SEP]' for text in texts]
        texts = tokenizer(texts,
                          padding=True,
                          truncation=True,
                          max_length=100,
                          return_tensors='pt',
                          add_special_tokens=False).input_ids

        # Forward Pass
        # texts Shape: (batch_size, seq_len)
        # y_logits Shape: (batch_size, seq_len - 1, vocab_size)
        y_logits = model(texts[:, :-1])[0]
        
        # Calculate Loss
        loss = loss_fn(y_logits.permute(0, 2, 1), texts[:, 1:])
        batch_loss = loss.item() * texts.shape[0]
        train_loss += batch_loss
        
        # Opimizer zero_grad
        optimizer.zero_grad()

        # Optimizer step
        optimizer.step()

    train_loss /= len(train_dataloader)
    return train_loss