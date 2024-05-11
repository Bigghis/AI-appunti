# Learner Framework

```py
```

L'implementazione e gestione del training loop può complicarsi da subito. Cambiamenti, aggiunte di codice, necessità di debugging, monitoraggio dei parametri etc. rendono difficilmente gestibile le logiche dentro la funzione **fit()**.  
Per questi motivi si rende necessaria la creazione di un vero e proprio **framework per addestrare le reti neurali**, che sia flessibile e che sappia adattarsi ad ogni modello  applicato.  
L'idea è di creare una classe **Learner** che implementa funzioni per gestire il training in modo ordinato e facilmente modificabile.  

### Callbacks
Quando vogliamo stampare dei dati all'interno del training loop, oppure vogliamo debuggare, se poniamo il codice di queste funzionalità **"accessorie"**, separato ed esterno al training loop, riusciremo a semplificarne di molto la gestione.  
Poniamo il codice che esegue la funzionalità accessoria in una funzione esterna e la eseguiamo passandola come argomento ad un'altra funzione che viene eseguita nel training loop.  
Definiamo cioè, di fatto, delle **callback** richiamate all'interno del training loop, in alcuni punti chiave predeterminati, come per esempio:

* **before_fit** (prima dell'esecuzione del training loop)
* **before_epoch** (prima di ogni iterazione sull'intero Dataset)
* **before_batch** (prima di eseguire **forward()** su un singolo batch)
* **after_batch** (dopo calcolata la **loss function** e aggiornati i pesi se si è in modalità training, per un singolo batch)
* **after_epoch** (dopo l'iterazione sull'intero Dataset)
* **after_fit** (dopo l'esecuzione del training loop)

Possiamo ovviamente individuare altri punti, se necessario. (es.: after_predict, after_loss etc.)
Possiamo eseguire una lista di callback in ognuno di questi punti individuati, definendo una funzione di raccolta **run_cbs()** che le richiama.  
E' importante definire un sistema per stabilire l'eventuale ordine (**order**) di esecuzione delle callback della lista e, nel caso in cui le callback si influenzino tra loro, un elenco di **eccezioni** sollevate durante l'esecuzione di una specifica callback, che ne blocca l'esecuzione di altre callback.

```py
class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

class Callback(): order = 0

def run_cbs(cbs, method_name, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_name, None)
        if method is not None: method(learn)
```

Passiamo alla definizione del framework, strutturato dalla funzione **fit()**:

```py
class Learner():
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=optim.SGD): fc.store_attr()

    def one_batch(self):
        self.preds = self.model(self.batch[0])
        self.loss = self.loss_func(self.preds, self.batch[1])
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()

    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dls.train if train else self.dls.valid
        try:
            self.callback('before_epoch')
            for self.iter,self.batch in enumerate(self.dl):
                try:
                    self.callback('before_batch')
                    self.one_batch()
                    self.callback('after_batch')
                except CancelBatchException: pass
            self.callback('after_epoch')
        except CancelEpochException: pass
    
    def fit(self, n_epochs):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        try:
            self.callback('before_fit')
            for self.epoch in self.epochs:
                self.one_epoch(True)
                self.one_epoch(False)
            self.callback('after_fit')
        except CancelFitException: pass

    def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)
```


Abbiamo così un sistema flessibile e facilmente ampliabile, dove risulta molto semplice aggiungere funzionalità.

Esempio: