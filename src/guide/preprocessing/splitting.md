# Splitting dataset

Per cercari di evitare problemi di **overfitting**, di solito il **training set** dei dati di input viene diviso in tre set:

* train set (dati di addestramento)
* dev or validation set (dati di valutazione)
* test set 

#### Train Set
Generalmente composto dall'80% del training set totale.  
Usato per allenare i **parametri** del modello, ottimizzandoli, per esempio, nella **discesa del gradiente**.
#### Validation Set
Generalmente composto dal 10% del training set totale.  
Serve per dirci quanto il modello si è adattato ai dati di input del train set.  
Usato per allenare gli **iperparametri** del modello, ad esempio per trovare un congruo numero di neuroni di un layer, o per un corretto learning rate iniziale.
#### Test Set
Generalmente composto dal 10% del training set totale.  
Usato per valutare la **performance del modello**, dopo il tuning di parametri ed iperparametri, alla fine del processo di training.  
Il train su questo set viene fatto con molta parsimonia, per non influenzare i risultati raggiunti
sull'ottimizzazione di parametri ed iperparametri ed incorrere in possibili **overfitting**.


```py
# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])     # train set  80%
Xdev, Ydev = build_dataset(words[n1:n2]) # dev set 10%
Xte, Yte = build_dataset(words[n2:])     # test set 10%
```


Addestrando la rete prima con (Xtr, Ytr), poi con (Xdev, Ydev)
se si ottengono risultati soddisfacenti e valori compatibili della funzione di perdita, significa 
che la rete si comporta bene (**underfitting**)
