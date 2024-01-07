# negative log likelihood loss


Gli elementi della lista di probabilità fornita in output dalla rete neurale possono essere sommati, per ottenere un singolo valore
che misura la perdita della rete.  

Gli elementi possono anche essere **moltiplicati (likelihood)** tra loro, per ottenere il prodotto.  
Maggiore è il valore del prodotto, maggiore è la qualità della rete.  
Per comodità si usa considerare il **logaritmo di questo prodotto**, perché darà un valore tra 0 e numeri negativi.  
Per probabilità  = 1 --> log = 0  
Per probabilità < 1 --> log sempre negativi  

Per lavorare con valori sempre positivi si usa **- log()** log negativo.  
Possiamo dunque sommare tutti valori di **-log(probabilità dell'output)** in modo tale che più il risultato sarà basso, meglio è
per la qualità della rete.  

```py
# immaginiamo di avere già le uscite logits:
counts = logits.exp() # counts, equivalent to N
probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
loss = -probs[torch.arange(num), ys].log().mean()
```

In realtà, per problemi legati ai calcoli, quali overflow etc., si utilizza una funzione già presente in pytorch:

```py
import torch.nn.functional as F

loss = F.cross_entropy(logits, Y)
```


