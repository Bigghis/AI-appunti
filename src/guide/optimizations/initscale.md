# Initialization

Un aspetto fondamentale per un corretto allenamento di una rete neurale è fornire dei valori in input corretti, durante la prima inizializzazione.  
Le probabilità dei valori di input, all'inizio, dovrebbero seguire quanto più possibile una **distribuzione gaussiana**.  
Vorremo avere una **media = 0** e una **deviazione standard = 1** per i dati in ingresso.

Una rete neurale esegue, infatti, moltiplicazioni tra matrici e lo fa molte volte.  
Se i valori da moltiplicare non si mantengono limitati, si incorre presto in numeri molto grandi e overflow.  
Perciò anche i pesi vanno tenuti entro i limiti di una distribuzione gaussiana uniforme.  

**I pesi non possono essere troppo grandi, ma neanche troppo piccoli**, per non incorrere nel problema che possano azzerarsi
durante le moltiplicazioni di matrici iterate più volte. 

In aggiunta l'**effetto delle funzioni di attivazione** modifica la media e la varianza della distribuzione dei valori.  

# Normalizzazione dei dati di input

E' possibile provare a rettificare media e varianza dei dati di input presentati in batch calcolando media e varianza del primo mini batch
e poi sottraendo tale media ai mini batch seguenti e dividendo per la varinza trovati all'inizio.  
Se usiamo l'AI Framework possiamo applicare questi calcoli usando una classe Callback apposita

```py
train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
# get a mini batch from the train dataloader
b = next(iter(train_dl)) 

# calculate mean and standard deviation of input data mini batch
xmean,xstd = b[0].mean(), b[0].std()

# create a function to apply means and stds to normalize input data batch
from AIFramework.callbacks import BatchTransformCB
from AIFramework import Learner

def _norm(b):
    return (b[0]-xmean) / xstd, b[1]

normalize_input_batches = BatchTransformCB(_norm, print_means=True)

# and then adds normalize_input_batches to the callbacks list of the learner:
cbs = [get_device_cb('cuda'), normalize_input_batches, get_metrics_cb(), ...]
learn = Learner(mod.apply(init_weights), dlsForConvolutional,  loss_func=F.cross_entropy, lr= 0.01, callbacks=cbs, opt_func=optim.AdamW)
```

# Normalizzazione dei pesi

## Xavier/Glorot scale 
Per non far crescere troppo i valori dentro le matrici si possono inizializzare i pesi scalandoli per un certo fattore
per renderli una distribuzione uniforme o una distribuzione normale.

Immaginando di avere:
```py
import matplotlib
import matplotlib.pyplot as plt


x = torch.randn(1000, 10) # dati di input, 1000 possibili esempi, embeddati in 10 dimensioni
W = torch.randn(10, 200) # 200 neuroni, ognuno accetta 10 input
y = x @ W # attivazione, moltiplicazione per i pesi W

print(x.mean(), x.std())
# tensor(0.0162) tensor(1.0022)  i dati di input sono distribuiti secondo una distribuzione gaussiana normale!

print(y.mean(), y.std())
# tensor(-0.0043) tensor(3.1594) i dati dopo l'attivazione, purtroppo NON sono distribuiti secondo una distribuzione gaussiana normale!
```

Come facciamo per riportare y a forma di gaussiana,  
cioè per riportare una deviazione standard = 1?  

Basta dividere i pesi per la radice quadrata del numero di elementi di input **fan in**, con:  
**fan_in = embedded_size * block_size**:

```py
embedded_size = 2
block_size = 5

W = torch.randn(10, 200) / ((embedded_size * block_size)**0.5)

# otterremo di nuovo media zero e std dev 1:
print(y.mean(), y.std())
# tensor(0.0044) tensor(1.0029)
```
**gain / ((embedded_size * block_size)*\*0.5)** è anche detto **Xavier/Glorot init** dei pesi di input.  
in pytorch possiamo usare funzioni già definite che fanno questo lavoro.

#### Uniform distribution
in una distribuzione uniforme i dati sono equiprobabili. 
Largamente usata e ben tollerata dalle reti.  
Si noti che può essere troppo casuale per dati ben strutturati e troppo dispendiosa per dati sparsi.  

#### Normal distribution
in una distribuzione normale i dati vicini alla media sono più probabili rispetto a quelli lontano dalla media.  
Questo tipo di inizializzazione è largamente usato ed è un buon modo per inizializzare i pesi.  
Si noti che può essere semplicistico in caso di dati distorti, o troppo restrittiva nel caso in cui i dati presentino relazioni tra loro complesse.

L'utilizzo in pytorch viene fatto a livello di layer:
```py
import torch.nn.init as init

# Glorot initialization with uniform distribution
layer = nn.Linear(100, 10)
init.xavier_uniform_(layer.weights)

# Glorot initialization with normal distribution
layer = nn.Linear(100, 10)
init.xavier_normal_(layer.weights)

print (layer.weights)
```
**L'inizializzazione Xavier/Glorot funziona meglio in presenza di funzioni di attivazione di tipo Sigmoid o Tanh**

## He/Kaiming 

Per avere una inizializzazione con media=0 e varianza=1 su reti con attivazioni di tipo **ReLU** si usa 
un metodo un pò diverso, presente sempre nelle due varianti di distribuzione uniforme e normale.

```py
# He initialization with uniform distribution
layer = nn.Linear(100, 10)
init.kaiming_uniform_(layer.weights)

# He initialization with normal distribution
layer = nn.Linear(100, 10)
init.kaiming_normal_(layer.weights)

print (layer.weights)
```

## Gain per compensare l'effetto della funzione di attivazione non lineare

Abbiamo visto che le funzioni di attivazioni non lineari, come **tanh()**, **relu()**, **sigmoid()**, tendono a schiacciare i loro output
verso valori limite, causando **saturazione**.  
**La saturazione rovina la distribuzione normale voluta in input della rete, cambiandone media e varianza!**  
Per **compensare** quest'effetto si usano dei valori di **gain** predeterminati analiticamente.  
Per esempio, per tanh() si usa **5/3**, etc. etc.

Avremo quindi, in aggiunta:
```py
# considerando
embedded_size = 2
block_size = 5

# avremo
W = torch.randn(10, 200) * (5/3) / ((embedded_size * block_size)**0.5)

```
Pytorch mette a disposizione la funzione **calculate_gain**, che determina il gain corretto da usare in base alla funzione di attivazione utilizzata:
```py
# es.: per tanh:
torch.nn.init.calculate_gain('tanh') # 1.666667 = 5/3
```