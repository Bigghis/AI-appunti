# Normalizations

Normalizzare i dati di input per renderli distribuzioni gaussiane uniformi, aiuta nel processo di training. 
Prevede il calcolo di **media** e **varianza** di un insieme di dati di input.   
Ci sono varie tecniche di normalizzazione, le più usate sono:  

### Batch Normalization
Media e varianza vengono calcolate attraverso le istanze del batch, cioè lungo le **colonne del batch**.  
Dipende da **batch_size**.  
es.: se abbiamo un tensore di input del tipo:  

```py
x = torch.tensor([[ 2.,  4.,  6.],    # x.shape = [3, 3]
                  [ 8., 10., 12.],
                  [14., 16., 18.]])

# la media è per dim 0
mean = x.mean(0, keepdim=True)  # tensor([[ 8., 10., 12.]])
# dove: 8 = (2 + 8 + 14) / 3
#       10 = (4 + 10 + 16) / 3
#      
 ...
# la varianza è per dim 0
xvar = x.var(0, keepdim=True)

```

### Layer Normalization
Media e varianza vengono calcolate per ogni istanza del batch, cioè lungo le **righe del batch**.  
Non dipende da **batch_size**.   
es., considerando sempre il tensore t:  

```py
# la media è per dim 1
mean = x.mean(1, keepdim=True)  # tensor([[4.],
                                #         [10.],
                                #         [16.]] )

# dove: 4 = (2 + 4 + 6) / 3
#       10 = (8 + 10 + 12) / 3
#       ...

# la varianza è per dim 1
xvar = x.var(1, keepdim=True)
```

### Group Normalization
Il batch viene suddiviso in gruppi **num_groups** e vengono normalizzate le features di ogni gruppo.  
Non dipende da **batch_size**.  
es.:  
```py

num_groups = 4
B, T = x.shape
x = x.view(B, num_groups, -1) # ripartizione in gruppi

mean = x.mean(-1, keepdim=True)
var = x.var(-1, keepdim=True)
```


#### esempio di utilizzo dei vari tipi di normalizzazione:

```py
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

def batch_norm(x):
    mean = x.mean(0, keepdim=True)
    var = x.var(0, unbiased=False, keepdim=True)
    x_norm = (x - mean) / (var + 1e-5).sqrt()
    return x_norm

def layer_norm(x):
    mean = x.mean(1, keepdim=True)
    var = x.var(1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / (var + 1e-5).sqrt()
    return x_norm

def group_norm(x, num_groups):
    N, C = x.shape
    x = x.view(N, num_groups, -1)
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    x_norm = (x - mean) / (var + 1e-5).sqrt()
    x_norm = x_norm.view(N, C)
    return x_norm

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm_func):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.norm_func = norm_func
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.norm_func(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

# Create a random tensor with size (batch_size, input_dim)
x = torch.randn(32, 100)

# Create the MLP models with batch norm, layer norm, and group norm
model_bn = MLP(100, 64, 10, batch_norm)
model_ln = MLP(100, 64, 10, layer_norm)
model_gn = MLP(100, 64, 10, partial(group_norm, num_groups=4))

# Pass the input tensor through the models
output_bn = model_bn(x)
output_ln = model_ln(x)
output_gn = model_gn(x)

# Print the outputs
print("Output with batch norm:\n", output_bn)
print("\nOutput with layer norm:\n", output_ln)
print("\nOutput with group norm:\n", output_gn) 
```
