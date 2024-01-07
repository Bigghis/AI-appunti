# Implementation

Creaimo una rete che determina il prossimo carattere sulla base di un **contesto** di **block_size** caratteri precedenti,
con block_size=8 caratteri.  
Dotiamola di layer **batchNorm**.  

Partendo dal solito boilerplate iniziale, abbiamo già ricavato i set di dati di input:  
```py
Xtr,  Ytr  = build_dataset(words[:n1])     # training set (80% of total)
Xdev, Ydev = build_dataset(words[n1:n2])   # validation set (10% of total)
Xte,  Yte  = build_dataset(words[n2:])     # test set (10% of total)

# shapes:
# Xtr:  torch.Size([182625, 8]), Ytr:  torch.Size([182625])
# Xdev: torch.Size([22655, 8]),  Ydev: torch.Size([22655])
# Xte:  torch.Size([22866, 8]),  Yte:  torch.Size([22866])
```

Creiamo la rete:  

```py
vocab_size = 27 # len(itos) alphabet chars
block_size = 8 # contensto caratteri precedenti 
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP
model = Sequential([
  Embedding(vocab_size, n_embd), # input layer
  Flatten(), Linear(n_embd * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(), # hidden layers
  Linear(n_hidden, vocab_size) # output layer
])

# parameter init
with torch.no_grad():
  model.layers[-1].weight *= 0.1 # last layer make less confident

parameters = model.parameters()
print('total parameters:', sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True
```

Prendiamo un subset di 4 dati a caso e diamoli in input alla rete:  
```py
ix = torch.randint(0, Xtr.shape[0], (4,)) # estraggo 4 numeri a caso tra 0 e dim massima di Xtr (182625)
Xb, Yb = Xtr[ix], Ytr[ix]
# Xb.shape: [4, 8], Xb: tensor([[ 0,  0,  0,  0, 26,  1, 18,  9],
#                               [ 0,  0,  5, 12, 19, 16,  5, 20],
#                               [ 0,  0,  0,  0,  0,  0, 12,  5],
#                               [ 0,  0, 12, 15, 18,  1, 12, 25]])
# Yb.shape: [4], Yb: tensor([12, 20,  1, 20])

# diamoli in pasto alla rete:
logits = model(Xb)
```  

Adesso controlliamo le forme dei layer:  
* **Embedding Layer**: 
la lookup table crea un vettore 10-dimensionale per ogni carattere 
che stiamo cercando di apprendere.  
L'Embedding layer estrae un vettore 10-dimensionale per ogni numero di ogni riga di Xb  
Quindi il layer di embedding traduce ogni intero in input in un vettore 10-dimensionale.  
Facendo passare Xb attraverso l'embedding layer, in output viene creato un tensore [4, 8, 10]

```py
# Embedding layer:
# crea 8 vettori 10-dimensionali per ognuno dei numeri [ 0,  0,  5, 12, 19, 16,  5, 20] 
# che è una riga di Xb.
model.layers[0].out.shape # output shape: [4, 8, 10]
```
* **Flatten Layer**: 
il layer appiattisce le ultime due dimensioni del suo input [4, 8, 10],
restituendo la forma semplificata perché prende i vettori 10-dimensionali
e li accoda in una unica riga, concatenandoli.
```py
# Flatten layer:
model.layers[1].out.shape # output shape: [4, 80]
```

* **Hidden layers**: 
il **Linear layer** prende in input [4, 80] ed esegue moltiplicazione di matrici, siccome abbiamo impostato 
200 neuroni, darà in output una forma del tipo [4, 200], che rimarrà invariata per l'output
del **BatchNorm Layer** e del layer di attivazione non lineare **Tanh**

```py
# Linear layer (hidden layer)
# prende in input [4, 80] ed esegue moltiplicazione di matrici, ricordando
# che abbiamo impostato 200 neuroni,  dando in output
model.layers[2].out.shape # output shape: [4, 200]

# BatchNorm layer
model.layers[3].out.shape # output shape: [4, 200]

# Tanh layer
model.layers[4].out.shape # output shape: [4, 200]
```
* **Output layer**:
Avremo in output 4 righe (visto che abbiamo 4 input..) ognuna che potrà restituire uno dei 27 caratteri dell'alfabeto.
```py
# Linear NN output layer
model.layers[5].out.shape # output shape: [4, 27]
``` 


