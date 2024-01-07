# Data Chunks

Non possiamo pensare di dare in input alla rete un intero file di testo di grande dimensioni, seppur encodato in sequenze di interi, perché questo tipo di caricamento dei dati è inefficente, sia per il **training** che per l'**inferenza** della rete.  

Vengono utilizzati **chunks** (parti) del set di dati, **campionando** il testo di input.
La rete viene addestrata solo su 1 chunk alla volta. 
I chunks hanno una **dimensione del blocco**, o **dimensione di contesto** e 
una dimensione del chunk stesso: 

```py
# es.: un chunk di 4 righe, ogni riga è lunga 8 caratteri
block_size = 8  
chunk_size = 4
```

Ogni riga del chunk è lunga 8 caratteri, però al suo interno contiene anche altri input, che sono sottostringhe più corte, e quindi altri 8 dati di input:  

```py
# es.: riga [18, 47, 56, 57, 58,  1, 15, 47, 58]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

# when input is tensor([18]) the target: 47 # single char
# when input is tensor([18, 47]) the target: 56
# when input is tensor([18, 47, 56]) the target: 57
# when input is tensor([18, 47, 56, 57]) the target: 58
# when input is tensor([18, 47, 56, 57, 58]) the target: 1
# when input is tensor([18, 47, 56, 57, 58,  1]) the target: 15
# when input is tensor([18, 47, 56, 57, 58,  1, 15]) the target: 47
# when input is tensor([18, 47, 56, 57, 58,  1, 15, 47]) the target: 58 # max dim block size
```
#### Time dimension 
La rete ha 8 esempi di input su cui basarsi per predirre il prossimo carattere nella sequenza, partendo da un singolo carattere del primo esempio, fino all'esempio di dimensione massima del blocco. In questo modo la rete può addestrarsi anche su sottosequenze di caratteri di contesto minore rispetto al solo blocco di 8 caratteri.  Questo è molto utile durante l'**inferenza** (quando si chatta con un LLM GPT), in quanto si può iniziare da un singolo carattere di contesto e GPT ne prevede il successivo. Si noti che durante l'inferenza i blocchi comunque non possono superare la prefissata **block_size**, per cui stringhe di contesti più grandi verranno troncate
fino alla lunghezza di contesto massimo.  
Questi blocchi, vengono mandati in parallelo in input alla rete, per velocizzare i calcoli.  
Questi aspetti della sequenza di input prendono il nome di time dimension del blocco di input.  

#### Creazione del Chunk di dati

creiamo il tensore di batch **xb** delle 4 righe di input ed il tensore **yb** che contiene i target, cioè gli output che vorremmo ottenere a partire da un particolare input.  

es.: se xb = [18, 47, 56, 57, 5,  1, 15, 47]  
yb corrispondente è = [47, 56,  57, 5,  1, 15, 47, 39] 

perché, come visto sopra: a [18] corrisponde [47], [18, 47] -> [56] etc. etc.  

In totale, quindi, avremo 32 samples in singolo batch **xb**, visto che ogni riga incorpora 8 esempi.  

```py
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

# xb.shape = [4, 8]
# yb. shape = [4, 8]


```

#### Aggregazione dati nel channel
Immaginiamo di avere un batch di 4 righe, ogni riga contiene 8 caratteri (quindi 8 esmepi di input differenti) e viene usata una lookup table per l'embedding in 2 dimensioni.  

Batch = 4, Time = 8, Channel (lookup table dim) = 2.  

Gli 8 esempi di input differenti di ogni riga, non comunicano tra di loro.  
Per ipotizzare una forma di comunicazione tra questi dati, possiamo calcolare la **media** 

```py
B, T, C = 4, 8, 2
x = torch.randn(B, T, C) 
# x.shape = torch.Size([4, 8, 2])

```


