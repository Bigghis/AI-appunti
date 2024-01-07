# Data Batch


Un **Training set** di dati può essere suddiviso in set più piccoli, chiamati **batch training set**.  
Di solito si prende un sottoinsieme random del training set iniziale.  
Ad esempio:

```py
# Se il training set è: (200k sequenze di 3 caratteri)
X = [
    [5, 13, 13],
    [13, 13, 1],
    [...],
    ...,
    ,,,
]
# X.shape = [200000, 3]

# minibatch construct 
ix = torch.randint(0, 200000, (32,)) # estrae un tensore di forma [32] di 32 numeri a caso nell'intervallo da 0 a 200000
# ix shape = [32]

X[ix]  # set di dati minibatch di input

# ovviamente possiamo fare la stessa cosa anche per i dati di output:
Y[ix] # set di dati minibatch di output

```
Usando il mini batch di dati, la qualità dei gradienti calcolati nella backpropagation è minore, rispetto
a quelli calcolati usando l'intero training set.  
Ma una minore qualità del gradiente è sempre accettabile, invece di un alto costo di elaborazione per il calcolo dei gradienti, fatto sull'intero training set.