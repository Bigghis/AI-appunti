# Pytorch Implementation

Elenchiamo dei concetti che sono alla base di qualsiasi rete neurale ed
esaminiamo il funzionamento di alcuni moduli presenti in pytorch, creati da tali concetti.  

Esistono degli aspetti di base, comuni a tutte le reti neurali, che permettono
il corretto funzionamento del **training loop** e della **backpropagation**.  



esempio di una rete **MLP** custom create a partire da **nn.Module**

```py
class MLP(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.l1 = nn.Linear(n_in,nh)
        self.l2 = nn.Linear(nh,n_out)
        self.relu = nn.ReLU()
        
    def forward(self, x): return self.l2(self.relu(self.l1(x)))
```

notiamo come nell'**__init__()** viene chiamato prima l'**__init__()** della superclasse, poi impostati i 3 layer e poi come viene reimplementata la **forward()**.

instanziando la classe MLP pytorch ci assicura l'accesso ai parametri della rete tramite il metodo **parameters()**

```py
model = MLP(n_in, nh, n_out)
model.parameters()
```

## Sequential

Per assicurarci della correttezza delle connessioni dei layer in sequenza, e quindi della corretta creazione dei parametri etc, possiamo usare **nn.Sequential**

```py
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
```

## Deep dive into the training loop
consideriamo un classico training loop:  

```py
def fit():
    for epoch in range(epochs):
        for i in range(0, n, bs):
            s = slice(i, min(n,i+bs))
            xb, yb = x_train[s], y_train[s]
            preds = model(xb)
            loss = loss_func(preds, yb)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): p -= p.grad * lr
                model.zero_grad()
        report(loss, preds, yb)
```


## Optimizer

Quando nel **training loop** effettuiamo l'operazione di aggiornamento dei pesi, sottraendo il gradiente al peso attuale moltiplicato per un certo **learning rate**, stiamo in realtà **ottimizzando** i pesi del modello per cercare di **minimizzare la loss function**.  
Possiamo considerare come parte di questa ottimizzazione anche l'azzeramento dei gradienti, per non incorrere in calcoli errati nella successiva iterazione del loop.  

```py
    for epoch in range(epochs):
        for i in range(0, n, bs):
            ...
            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): p -= p.grad * lr  # fase di ottimizzazione dei pesi
                model.zero_grad()  # fase di ottimizzazione dei pesi
```

Possiamo quindi usare **optim.SGD**, per esempio, che esegue le stesse ottimizzazioni del codice esposto sopra.  

Infatti l'implementazione di **optim.SGD** è:

```py
class Optimizer():
    def __init__(self, params, lr=0.5): self.params,self.lr=list(params),lr

    def step(self):
        with torch.no_grad():
            for p in self.params: p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params: p.grad.data.zero_()
```



Alla luce di questi strumenti messi a disposizione da pytorch, possiamo rifattorizzare il training loop di partenza:

```py
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):
    for i in range(0, n, bs):
        s = slice(i, min(n,i+bs))
        xb, yb = x_train[s], y_train[s]
        preds = model(xb)
        loss = loss_func(preds, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    report(loss, preds, yb)
```

## Dataset
Quando nel training loop prendiamo la variabile dipendente **xb** e quella indipendente **yb**, con: 

**xb, yb = x_train[s], y_train[s]**  

può essere complicato iterare separatamente i valori del minibatch.  
Per semplificare dovremmo ottenere una assegnazione del tipo:  

**xb, yb = train_dataset[s]** 

dove **train_dataset** è istanza di una classe **Dataset**, del tipo:

```py
class Dataset():
    def __init__(self, x, y): self.x,self.y = x,y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i],self.y[i]

# istanziazione:
train_dataset = Dataset(x_train, y_train)
```
che al suo interno, tramite la **getitem**, si occupa di "sdoppiare" correttamente le variabili dipendente ed indipendente.


## Dataloader

nel training loop, il caricamento di xb, yb avviene con:

```py
for epoch in range(epochs):
    for i in range(0, n, bs):
        s = slice(i, min(n,i+bs)) # fase caricamento xb, yb
        xb, yb = train_dataset[s] # fase caricamento xb, yb
        ...
```

è molto più comodo sostituire le due righe con:

**for xb, yb in train_dataloader:** 

dove **train_dataloader** è istanza della classe:

```py
class DataLoader():
    def __init__(self, ds, bs): self.ds, self.bs = ds, bs
    def __iter__(self):
                # range(start, stop, step)
        for i in range(0, len(self.ds), self.bs): yield self.ds[i:i+self.bs]

# bs = batch size, es.: bs = 50
train_dataloader = DataLoader(train_dataset, bs)
```
reimplementando la **iter**, creiamo un generator .... continue


Il DataLoader fornito da pytorch permette di:  

* creare batch di dati a partire da un dataset
* randomizzare l'ordine dei batch ad ogni iterazione (shuffle)
* elaborare (train e eval) più batch in parallelo


Il DataLoader **DataLoader(data, collate_fn=collate_fn, batch_size=batch_size)** di pytorch si aspetta che la variabile dipendente e la variabile indipendente dentro il batch **data** sia del tipo:

* numeri
* tensori
* array (numpy)
* list
* dict

Quindi, in caso la variabile dipendente e la variabile indipendente dentro il batch **data** non abbiano la forma di un input del tipo elencato, bisogna implementare una **collate function**, come negli esempi:

```py
# esempio che ritorna un batch di tipo dict e che ha al suo interno
# tensori, che sono accettati dal dataloader
def collate_fn(elem):
    return {
        'image': stack([TF.to_tensor(o['image']) for o in elem]), #variabile dipendente di tipo tensor
        'label': tensor([o['label'] for o in elem])  # variabile indipendente di tipo tensor
    }

# esempio che ritorna un batch di tipo list e che ha al suo interno
# tensori, che sono accettati dal dataloader
def collate_fn1(elem):
    image = stack([TF.to_tensor(o['image']) for o in elem]) #variabile dipendente di tipo tensor
    label  = tensor([o['label'] for o in elem]) #variabile dipendente di tipo tensor
    return [image, label]
```

e passarla in argomento al DataLoader. Internamente verrà eseguito un **map()** applicando la **collate_fn** fornita per ogni elem di data.  

Ovviamente la collate function deve ritornare neccessariamente un batch in una delle forme accettate dal DataLoader.


IMPORTNTE: https://medium.com/@phiphatchomchit/lets-deploy-your-deep-learning-model-with-gradio-c7e5ceb4122f

REVERSE :) interesting https://medium.com/@0ssamaak0/building-simple-handwritten-digits-generator-using-dcgan-and-gradio-96b5aace1eb2
