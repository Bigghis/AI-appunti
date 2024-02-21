# Embeddings
Altro tipo di codifica, molto usata, è l'**embedding**, l'incorporamento dei dati in spazi n-dimensionali.  
Possiamo pensare di memorizzare i 27 caratteri dell'alfabeto in uno spazio a 2 dimensioni. Ogni carattere sarà rappresentato
da un array di 2 elementi, cioè da coordinate x, y che individuano un punto nello spazio.  
Quel punto rappresenta il carattere codificato.  

**L'embedding trasforma ogni intero dato in input in un array n-dimensionale in output.**  

Creiamo una **lookup table** o tabella di codifica, con 27 elementi, ogni elemento è un'array [x,y] che codifica un carattere
dell'alfabeto.  
Avremo quindi una tabella d i27 righe e ogni riga contiene parametri addestrabili tramite una rete neurale.  
All'inizio i valori delle coordinate x, y della lookup table saranno casuali e compresi tra 0 e 1.

```py
C = torch.rand(27,2)
```
Sapendo che il carattere 'm' = 13 dentro stoi, possiamo anche dire che 'm' corrisponde al tredicesimo elemento della lookup table C.  

'm' = 13 = C[13] = tensor([0.7546, 0.0540])


Spesso si considera il meccanismo dell'embedding all'input della rete come il primo layer (**input layer**) della rete stessa,
in quanto C sono i pesi dei neuroni in ingresso e i caratteri codificati sono gli input.  
Nel caso dell'esempio, avremo un input layer di 27 neuroni (abbiamo 27 possibili input!) e C sono i pesi.  
X = C * i  
Nell'input layer Non vengono fatte elaborazioni particolari, come attivazioni etc.  


Immaginiamo di avere una rete **n-gram**, con n = 3, che prende in input 3 una sequenza di 3 caratteri per predirne un quarto in
outpt.  
Avremo in input varie sequenze di 3 caratteri, avremo, cioè array di 3 interi, visto che ogni carattere è codificato
in stoi.  
 
```py
# Se le varie sequenze di input sono, ad esempio, 4 array da 3 caratteri ciascuno:
X = tensor([
    [ 5,  5,  5],
    [ 5, 13, 13],
    [13, 13,  1],
    [13,  3,  8]
])

# possiamo embeddarle direttamente dentro C facendo:
C[X]

```
Quindi, nell'esempio:  

['e', 'm', 'm'] = [5, 13 ,13] = tensor([ [0.8008, 0.7908], [0.7546, 0.0540], [0.7546, 0.0540] ])  


#### Positional Encoding 
In alcune reti, come per esempio quelle dotate di **self-attention**, è necessario encodare non solo i possibili caratteri in input (**token**), ma anche la **loro posizione**.  
I **logits** creati dalla rete sono in tensore di dimensioni (B, T, C), come di consueto.  
All'interno della dimensione T abbiamo una informazione duplice: il token encodato in un vettore C e, appunto, la sua posizione.  

es.: [34, 45, 65, 29]  
 il token 45 è encodato nella 45esima riga della lookup table C e si trova in seconda posizione dentro la riga di T.  
Per memorizzare il fatto che 45 si trova in seconda posizione, possiamo usare una ulteriore lookup table.  

Quindi se abbiamo creato una tabella di Embeddings per i token (esempio fatto con pytorch, per semplificare):  
```py 
import torch
import torch.nn as nn

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)
```

Dobbiamo aggiungere una seconda tabella di embeddings per le posizioni dei token, 
in modo che ogni posizione nell'intervallo [0, block_size] del **block_size** sarà rappresentata
da un vettore lungo **n_embd** .  
Di conseguenza per calcolare logits, serviranno entrambe (**tok_emb**, **pos_emb**) le componenti embeddate

```py 
block_size = 8
vocab_size = 27 # caratteri dell'alfabeto
n_embd = 32
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = x
```

## Considerazioni
Nella pratica, possiamo anche definire il concetto dell'embeddings, come il ricercare un elemento all'interno di un'array.  
Tale ricerca non è altro che moltiplicare il contenuto dell'array per un altro array encodato one-hot.  
E', quindi, una moltiplicazione tra matrici.  





