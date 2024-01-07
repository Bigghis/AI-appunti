# Flatten input data

Per varie esigenze è possibile voler cambiare forma ai tensori dei dati di input, per esempio dopo l'embedding, modificandone le dimensioni.  
Si può usare la funzione pytorch **view()**, adatta allo scopo.  

#### implementazione class Flatten

```py
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []
```

che è sostanzialmente uguale alla classe pytorch:  

```py
torch.nn.Flatten(start_dim=1, end_dim=-1)
```