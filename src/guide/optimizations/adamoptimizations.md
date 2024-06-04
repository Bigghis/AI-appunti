# ADAM ( Adaptive Moment Estimation)

Un optimizer che sfrutta **RMSProp** e **momentum** è Adam.  

Prevede parametro **beta1** che è relativo al momentum e **beta2** che è relativo ad RMSprop.  
Il suo **stato** prevede la memorizzazione della media mobile dei gradienti e quella dei quadrati dei gradienti, per ciascun peso (parametro).

Il footprint di memoria dello stato è tipicamente 8 bytes per ogni peso del modello.  

E' molto noto, visto che risulta essere più efficiente dei due metodi che ingloba, se presi singolarmente.  

```py
class Adam(SGD):
    def __init__(self, params, lr, wd=0., beta1=0.9, beta2=0.99, eps=1e-5):
        super().__init__(params, lr=lr, wd=wd)
        self.beta1,self.beta2,self.eps = beta1,beta2,eps

    def opt_step(self, p):
        if not hasattr(p, 'avg'): p.avg = torch.zeros_like(p.grad.data)
        if not hasattr(p, 'sqr_avg'): p.sqr_avg = torch.zeros_like(p.grad.data)
        p.avg = self.beta1*p.avg + (1-self.beta1)*p.grad
        unbias_avg = p.avg / (1 - (self.beta1**(self.i+1)))
        p.sqr_avg = self.beta2*p.sqr_avg + (1-self.beta2)*(p.grad**2)
        unbias_sqr_avg = p.sqr_avg / (1 - (self.beta2**(self.i+1)))
        p -= self.lr * unbias_avg / (unbias_sqr_avg + self.eps).sqrt()
    ```
