# tanh() class implementation

Una possibile implentazione della funzione di attivazione **tanh()**, richiamando pytorch tanh()

```py
class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

```

che corrisponde a pytorch:  
```py
torch.nn.Tanh()
```