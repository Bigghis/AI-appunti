# SGD optimizer


un **Optimizer** si occupa di aggiornare i valori dei pesi (**parametri**) nel training loop, sottraendo il gradiente al peso attuale moltiplicato per un certo **learning rate**.  

Un optimizer cioè, **ottimizza i pesi del modello per cercare di minimizzare la loss function.**.  

Un optimizer prevede **diversi iperparametri** con cui è possibile configurarlo.   
Generalmente gli iperparametri con cui è possibile configurare il funzionamento di un optimizer sono:

* **learning rate** che definisce la velocità di apprendimento durante il training del modello
* **weight decay** per la regolarizzazione dei pesi
* **momentum** che serve per compensare eventuali irregolarità (picchi e strappi elevati) della loss function, riducendo, appunto tali picchi e rendendo la loss funcion più smooth.  


## Implementazione

Per l'implementazione di un optimizer sono sempre previsti due metodi:  
* **step()** che esegue ottimizzazione dei pesi, essendo richiamato all'interno del training loop
* **zero_grad()**  che azzera i gradienti dopo che sono stati calcolati all'interno del training loop

Possono esserci anche altri metodi, per esempio:  
* **regularization_step()** che applica eventuale iperparametro del **weight decay**, moltiplicandolo per il learning rate, prima dell'ottimizzazione dei pesi  

Esempio di implementazione di un **SGD (Stochastic Gradient Descent) Optimizer**

```py
class SGD:
    def __init__(self, params, learning_rate, weight_decay=0.):
        params = list(params) # params è un generator, viene trasformato in lista
        fc.store_attr()
        self.i = 0

    def step(self):
        with torch.no_grad():
            for p in self.params:
                self.regularization_step(p)
                p -= p.grad * self.learning_rate # ottimizzazione dei pesi
        self.i +=1

    def regularization_step(self, p):
        if self.weight_decay != 0: 
            p *= 1 - self.learning_rate * self.weight_decay

    def zero_grad(self):
        for p in self.params: p.grad.data.zero_()
```

si noti che un SGD optimizer opera ogni volta su un **mini batch** di dati, non sull'intero dataset!  
Non memorizza al suo interno alcuna informazione (**stateless**), per cui non occupa memoria.