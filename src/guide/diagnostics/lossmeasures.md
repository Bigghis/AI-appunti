# Loss function measures 

Se stiamo usando un **model** pytorch like, possiamo stampare agevolmente i valori del loss
durante le iterazioni:  

```py
max_iters = 3000
eval_iters = 200 
eval_interval = 300

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# usage: 
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


# output exmaple:
# step 0: train loss 4.7305, val loss 4.7241
# step 300: train loss 2.8110, val loss 2.8249
# step 600: train loss 2.5434, val loss 2.5682
```
