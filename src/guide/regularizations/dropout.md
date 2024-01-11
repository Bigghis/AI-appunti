# Dropout

un Elemento di Dropout **torch.nn.Dropout**, spegne un sottoinsieme casuale neuroni, nel layer a cui è applicato.  
Il suo effetto casuale è applicato sia in forward che in backpropagation e il sottoinsieme **cambia** ad ogni loop, durante il training.  
Serve per scongiurare l'**overfitting** della rete.  
Possiamo aggiungere dropout, per esempio nell'head dell'attention:  

```py
dropout = 0.2 # elimino il 20% dei nodi

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)  # <--------DROPOUT

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # <--------DROPOUT
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout) # <--------DROPOUT

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out)) # <--------DROPOUT
        return out

```

e nel layer feed forward:

```py
dropout = 0.2

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout), # <--------DROPOUT
        )

    def forward(self, x):
        return self.net(x)
```

