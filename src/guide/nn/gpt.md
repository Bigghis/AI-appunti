# GPT

Costruiremo un **Generatively Pretrained Transformer (GPT)**, seguendo le linee guida del paper **Attention is all you need** e OpenAI GPT-2 / GPT-3.  

Consideriamo dati in input compattati in **chunks** di 4 righe. Ogni riga ha una dimensione di 8 caratteri. (**block_size**). I possibili caratteri di input sono 65 (minuscole, maiuscole, punteggiatura).
### Bigram 
Per iniziare consideriamo la rete **Bigram**.  
Creiamo la tabella di embedding (**lookup table**) dei 65 possibili caratteri del testo, che avrà 2 dimensioni [65, 65].  
Ogni **idx** passato al metodo **forward**, è un intero di una riga del **chunk** di input, e si trova nel range [0, 64]. E' l'indice di riga della lookup table.  
Es.: se idx = 24, verrà estratta la 24esima riga della **token_embedding_table**:

**token_embedding_table(idx)** crea un tensore **logits** di dimensioni [4, 8, 65], dove:  
Batch = 4, Time = 8, Channel (lokup table dim) = 65    
Avremo, cioè le 4 righe del batch e per ogni riga gli 8 elementi di riga.
Ogni elemento della riga è encodato a 65 caratteri.  
La dimensione del tensore **targets** è: [4, 8]


Per Calcolare il valore del **loss** usiamo la solita **F.cross_entropy**, avendo accortezza 
di modificare le dimensioni dei tensori, perché cross_entropy si aspetta dimensioni differenti da quelle attuali (vedi documentazione di cross_entropy).


Il metodo **generate** estrae il logit ottenuto per **idx**
e ne calcola le probabilità con **softmax** per poi recuperare la distribuzione di probabilità **probs** [1, 65] e quindi campionarne un solo elemento, che è il carattere predetto dalla rete neurale. Tale carattere viene accodato agli altri già trovati che comporranno l'intera stringa predetta lunga **max_new_tokens**.

```py

vocab_size = 65 # i 65 caratteri del vocabolario dei possibili caratteri
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # idx corrisponde ad una riga dell'embedding table
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape # dimensions: Batch, Time, Channel (embedding table) 
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)
```

Possiamo provare a generare del testo dal modello, eseguendo il metodo **generate()**, partendo 
da **idx** tensore di dimensioni [1, 1], contenente un solo numero 0.  
0 corrisponde al carattere di nuova linea **\n**.
Le frasi generate sono del tutto casuali, perché ancora non abbiamo addestrato il modello.  
```py
idx = torch.zeros((1, 1), dtype=torch.long, device=device)  
print(decode(m.generate(idx, max_new_tokens=500)[0].tolist())) # generiamo frasi per un max di 500 caratteri
```

### Training  
Per addestrare il modello inizializziamo prima un ottimizzatore. Ce ne sono diversi, finora abbiamo visto, per esempio, **Stochastic Gradient Descent (SGD)**, per la discesa del gradiente.  
In questo caso usiamo **AdamW**, impostando un learning rate fisso (**1e-3**).  
L'optimizer si occuperà di aggiornare i parametri del modello.  

```py
batch_size = 32 # aumento dimensione del batch

optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for steps in range(1000):
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(loss.item())
```
Con l'elaborazione in loop otterremo una discesa del loss.  

#### Aggregazione dei dati in input tramite calcolo della media.
Consideriamo i dati in input nella matrice [B,T,C], Batch=4, Time=8, Channel=2.  
Quindi i token sono disposti in una matrice [4, 8] e l'informazione riguardante ciascun token è codificata in 2 dimensioni.  
Uno qualsiasi degli 8 token di una riga T del Batch non comunica in alcun modo con gli altri token 
della stessa riga.  
Potremmo pensare ad una semplice forma di comunicazione di un token n-esimo con tutti i token da 
0 a n-1 della stesa riga. (non è possibile pensare ad una comuniazione con i token da n+1 in poi, in quanto sono token presenti nel futuro della sequenza, non nel passato).  
Un token n-esimo può comunicare con i token [0, n-1] se calcoliamo la **media** dei token precedenti.  
Prendiamo, cioè i **channels** del quinto, quarto, terzo, secondo e primo elemento di T, ottenendo una sorta di **vettore delle funzionalità** che mostra le aggregazioni tra i token nel contesto corrente.  
Ovviamente il calcolo della media esclude altri importanti legami tra i token, come la disposizione spaziale degli stessi all'interno dell'embedding table, ma per ora ci accontentiamo:

```py
# calcoliamo  x[b, t] = mean_{i<=t} x[b, i]
# Per ogni riga b del batch, prendiamo ogni token della riga e 
# tutti i suoi token precedenti e ne calcoliamo la media
x_back_of_words = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t, C)
        x_back_of_words[b, t] = torch.mean(xprev, 0) #media fatta sulla dimensione zero, cioè su t (t, C)

# x_back_of_words.shape = [4, 8, 2]
```
Per semplicità possiamo usare il prodotto di matrici con i tensori:
```py

a = torch.tril(torch.ones(3,3)) # crea un tensore "triangolo*
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
a = a / torch.sum(a, 1, keepdim=True)
# a = [1.0000, 0.0000, 0.0000],
#     [0.5000, 0.5000, 0.0000],
#     [0.3333, 0.3333, 0.3333]
# a contiene le medie riga per riga tr elemento n e elementi [0, n-1] 
# precedenti


b = torch.tensor([[1, 2],
                  [3, 4],
                  [6, 7]])

c = a @ b 
# c.shape: [3, 2]
# c contiene le medie come nel calcolo precedente!
# c = [2.0000, 7.0000],
#     [4.0000, 5.5000],
#     [4.6667, 5.3333]
```
Nel caso della nostra rete, possiamo scrivere:  
```py
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
# masked_fill fa il masking, ottenendo:
# tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
#         [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
#         [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
#         [0., 0., 0., 0., -inf, -inf, -inf, -inf],
#         [0., 0., 0., 0., 0., -inf, -inf, -inf],
#         [0., 0., 0., 0., 0., 0., -inf, -inf],
#         [0., 0., 0., 0., 0., 0., 0., -inf],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])

wei = F.softmax(wei, dim=1)
# softmax prima esegue exp() su ogni elemento, eliminando i valori negativi
# e poi divide ogni elemento per la somma di riga, normalizzando.
# ogni riga, infatti avrà somma pari a 1.
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
#         [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
#         [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
#         [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
x_back_of_words = wei @ x
# x_back_of_words.shape: [ 4, 8, 2]
```
usando **-inf** in **masked_fill** si vede che i token n non possono avere interazioni con i token da n+1 in poi, in quanto qualsiasi interazione col futuro è sconosciuta.
L'interazione con i valori da [0, n-1], parte dal valore 0 e aumenterà
nella varie operazioni del softmax().  

L'aggregazione vera e propria viene effettuata nella moltiplicazione matriciale **wei @ x**, che dota ogni token di una sorta di **attenzione verso sé stesso**, facendolo confrontare con i token precedenti.  
