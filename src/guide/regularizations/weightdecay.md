# Weight Decay

Un classico esempio di regolarizzazione consiste nel limitare i valori che assumono i pesi, durante il training.  
Ciò viene fatto per contrastare effetti di overfitting della rete.  

Per limitare i valori dei pesi viene, ad esempio, calcolata la somma dei quadrati di tutti i pesi e la si aggiunge alla loss function.  
Questo produce l'effetto di diminuire i valori dei pesi (weight decay) durante il training, perché nel calcolo gradienti aggiungerà un contributo che spinge i pesi ad essere più piccoli.  
L'effetto rende la curva della loss function più smooth (liscia), eliminando eventuali picchi dovuti ai dati del trainig set, in modo da generalizzare meglio il comportamento della rete per dati futuri ed aiuta la discesa della loss function.  

Nella pratica si traduce nell'aggiunta di un **iperparametro** **wd** che viene moltiplicato ai pesi, ad ogni loop:

```py
parameters *= (1 - learning_rate * weight_decay)
```

e usando pytorch le cose si semplificano, visto che, di solito, viene passato come parametro ad un qualsiasi **optimizer**, assieme al learning rate.

esempio:

```py
import torch.optim

optimizer = torch.optim.Adam(params, lr=3e-4, weight_decay=1e-3)
```

Per i valori da usare per **wd**, di solito si fanno delle prove durante i training usando multipli di 10: parti da 1 e dividi per 10 (0.1, 0.01 etc.)  
### Considerazioni 
Si noti che **limitare di poco** il valore dei pesi non ha effetto sul contrasto dell'overfitting, mentre
**limitare eccessivamente** il valore dei pesi, ha l'effetto di ridurre la capacità della rete di produrre buone previsioni sui dati futuri.  
Va sempre cercato un giusto compromesso!