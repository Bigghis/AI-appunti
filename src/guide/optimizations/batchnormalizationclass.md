# Implementing BatchNorm Class Layer

Classe Python che implementa un layer Batch. Useremo xmean e xvar per la **fase di training**,   
mentre useremo le **medie mobili esponenziali** running_mean e running_var per l'**inferenza** di nuovi dati.  
GLi array **gamma** e **beta** sono gli array di scale e shift.


```py
class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)  # gain
    self.beta = torch.zeros(dim)  # bias
    # buffers (trained with a running 'momentum update') [inferenza] 
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      xmean = x.mean(0, keepdim=True) # batch mean
      xvar = x.var(0, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta # self.out è utile per eventuali grafici che vogliamo tracciare
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]
```

è molto simile alla classe in pytorch:  
```py
torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```
