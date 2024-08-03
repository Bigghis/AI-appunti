# Custom Loss Functions

Pytorch mette a disposizione diverse loss functions (mse_loss, cross_entropy etc.).  
Tuttavia è possibile implementare delle loss function **custom** da usare durante il training loop.  

Per implementare una loss function custom bisogna creare una sottoclasse di **nn.Module** e definire 
il metodo **forward()**

```py
class CustomLossFunction(nn.Module):
    def __init__(self):
        super(CustomLossFunction, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean((predictions - targets) ** 2)
```


in genere il metodo forward accetta i parametri **predictions** e **targets** (anche detti inputs e targets) per effettuare, appunto, calcoli tra le predizioni fatte dal modello
e il target effettivo desiderato.  

Una loss function custom può essere usata all'interno del **Learner Framework** come le altre loss function già presenti in pytorch.  


## Utilizzo all'interno del framework Accelerate di HuggingFace

### Utilizzo diretto:

Per usare correttamente una loss function custom all'interno del framework **Accelerate di HuggingFace** bisogna, in genere:  

1) recuperare l'output dal modello
2) calcolare il valore numerico **loss** della loss function custom eseguita a partire dall'output del modello e targets desiderati
3) eseguire **accelerator.backward()** passando come parametro il **loss** calcolato  


### Utilizzo tramite Callback (Learner Framework):

Bisogna rispettare la procedura spiegata per l'utilizzo diretto.  
Possiamo creare una classe **AccelerateCustomCB** che eredita da **AccelerateCB**.  

Esempio d'uso di base:  

1) Nel metodo **get_loss()** calcoliamo il valore numerico **loss** della loss function custom
2) Nel metodo **backward()** eseguiamo **accelerator.backward()** passando come parametro il **loss** calcolato prima

Es.:

```py
class AccelerateCustomCB(AccelerateCB):
    def __init(self):
        super(AccelerateCustomCB, self).__init__()
    
    def get_loss(self, learn):
        self.loss = learn.loss_func(learn.preds)
        
    def backward(self, learn):
        self.acc.backward(self.loss)
```
