# Head Attention class implementation

### Implementazione dell'head singolo dell'attention 

Viene usato **register_buffer** per salvare la matrice triangolare **tril** in un buffer specifico, in modo che non venga considerata tra i parametri del modello.  
tril diventa una normale variabile d'istanza **self.tril**.  

```py
class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer salva 'tril' in self.tril
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T) # scaled
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) # decoder
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
```

### Multi Head Attention 
Creazione di più istanze di **Head** che lavorano in parallelo, per sfruttare più canali di comunicazione tra token.  


```py
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

Chiaramente quando viene utilizzata **MultiHeadAttention** i suoi parametri saranno:  

```py
# esempio:
n_embd = 32 
n_head = 4
head_size = n_embd // n_head # <----importante: dividere per n_head!
sa = MultiHeadAttention(n_head, head_size)
```
