# Linear Layer implementation

Implementazione di una layer lineare, es. un input layer.
Abbiamo la possibilità di escludere il bias
* **fan_in**: numero di input del layer (size of each input sample)
* **fan_out**: numero di output del layer (size of each output sample)
```py
class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight

    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])
```

è simile all'implementazione in pytorch:  
```py
torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
```
