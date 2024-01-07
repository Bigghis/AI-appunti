# Tuning neurons size in layer
Cambiando il numero di neuroni all'interno di un layer (tipicamente di un hidden layer),
si modifica il comportamento della rete.  
Ad esempio, considerando uno stesso training set, se si aumenta il numero di neuroni, si allontana il problema dell'underfitting.  

Partendo da una situazione del genere:
```py
g = torch.Generator().manual_seed(2147483647)

# parametri di input (27 caratteri embeddati in 2 dimensioni)
C = torch.randn((27, 2), generator=g)

# hidden layer composto da 100 neuroni, ogni neurone accetta tutti gli input 
# dai neuroni dell'input layer, che in totale sono 6 (3 caratteri per 2 dimensioni)
W1 = torch.randn((6, 100), generator=g)
b1 = torch.randn(100, generator=g)

# output layer composto da 27 neuroni, ogni neurone ha 100 input, corrispondenti
# alle uscite dell'hidden layer
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]
```

Per aumentare i neuroni dell'hidden layer da 100 a 300 facciamo:

```py

C = torch.randn((27, 2), generator=g)

W1 = torch.randn((6, 300), generator=g)
b1 = torch.randn(300, generator=g)

W2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
```