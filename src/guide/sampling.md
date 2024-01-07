# Sampling

Campionamento dei dati  
Di solito la distribuzione delle probabilità normalizzata viene **campionata**
usando **torch.multinomial**.

*torch.multinomial(input, num_samples, replacement=False, \*, generator=None, out=None)*

Multinomial funziona in questo modo:  
ritorna un tensore dove ogni riga contiene num_samples indici interi, campionati a partire dalla distribuzione di probabilità in input,
locata alla corrispondente riga del tensore.

es.: 
```py
g = torch.Generator().manual_seed(2147483647)

# immaginiamo di avere una distribuzione di probabità del tipo
p = torch.tensor([0.6064, 0.3033, 0.0903])

# tale distribuzione ci dice che:
# il valore 0 ha probabilità 0.60
# il valore 1 ha probabilità 0.30
# il valore 2 ha probabilità 0.09

t = torch.multinomial(p, num_samples=10, replacement=True, generator=g)
# t = tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0])

# Avremo un tensore che contiene un'array con i valori 0,1,2
# distribuiti secondo le probabilità dettate da p, quindi '0' comparirà il 60% delle volte,
# '1' il 30% e così via.
```

Considerando la mappatura itos dell'alfabeto, e avendo una distribuzione di probabilità dei 27 caratteri dell'alfabeto,
possiamo campionarne 1 valore:
```py
itos = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}

p = torch.randn(27) # 27 numeri random tra 0. e 1.
# p = torch.tensor tensor([ 1.5555, -0.3182, -0.6450, -0.6836, -2.0897,  0.9151,  0.2523, -1.1074,
# -0.6000, -0.5682,  1.0553,  0.3278, -0.2156,  1.6075,  0.9092, -1.0640,
# 1.0440,  2.1065,  0.4608, -1.4605,  2.5640, -1.0001,  0.1883,  0.0841,
# -0.3244,  0.9603, -1.9544])

# assicuriamoci che i valori siano tutti positivi
p = p.exp()

# e che la distribuzione sia normalizzata
p = p / p.sum()

# p = tensor([0.0123, 0.0208, 0.0279, 0.1123, 0.0113, 0.0551, 0.0053, 0.0669, 0.1635,
#        0.0381, 0.0490, 0.0233, 0.0108, 0.0280, 0.0281, 0.0268, 0.0763, 0.0824,
#        0.0495, 0.0071, 0.0272, 0.0044, 0.0183, 0.0187, 0.0103, 0.0089, 0.0174])

# p.shape = torch.Size([27])

d = torch.multinomial(p, num_samples=1, replacement=True, generator=g)

# d = tensor([3]) che corrisponde a 
itos[d.item()]  # 'c'

```
