# Embedding class implementation

Implementazione di un layer embedding che esegue l'operazione di indicizzazione degli input durante il forward pass.   

i parametri:  
* **num_embeddings** indica la grandezza del dizionario degli embeddings (es. i 27 caratteri dell'alfabeto)  
* **embedding_dim** indica la grandezza di ogni embedding vector di input (es. le 2 dimensioni che individuano un punto nello spazio, che rappresenta uno dei 27 caratteri dell'alfabeto)  

```py
class Embedding:
  
  def __init__(self, num_embeddings, embedding_dim):
    self.weight = torch.randn((num_embeddings, embedding_dim))
    
  def __call__(self, IX):
    self.out = self.weight[IX] # indicizzazione
    return self.out
  
  def parameters(self):
    return [self.weight]

```

E' simile alla classe pytorch:  


```py
torch.nn.Embedding(num_embeddings, embedding_dim, 
    padding_idx=None, max_norm=None, norm_type=2.0, 
    scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, 
    device=None, dtype=None
)
```