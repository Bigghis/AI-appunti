# String Tokenization

Conversione di un testo, che è una sequenza di caratteri, in una sequenza di numeri interi e viceversa.  
Per riferimento della conversione viene usato un **vocabolario di tokens, o lookup table** per esempio i caratteri dell'alfabeto, oppure tutti i caratteri tipografici presenti in un testo, presi una sola volta:  
```py
# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # vocabulary
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # lookup table
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# example:
encode("hii there!") # [46, 47, 47, 1, 58, 46, 43, 56, 43, 52]
decode(("hii there!")) # hii there!

```
Chiaramente se utilizzo vacabolari molto grandi, con tanti token diversi, posso usare 
sequenze di interi del testo encodato più piccole, un pò come nella compressione dei files, viceversa, con vocabolari piccoli, cresce la dimensione dei testi encodati.  

## Tokenization nei LLM
In realtà si usa convertire **chunks** di caratteri in interi, al posto dei singoli caratteri.  

Possiamo considerare un **token** come ad un atomo, un'unità minima di informazione per un LLM.  

GPT2, ad esempio, ha un vocabolario di circa 50k tokens e la dimensione di contesto dell'input layer del transformer usato dal modello è di 1024 tokens. Vuol dire che ogni token, all'interno del contesto, pone attenzione ai token precedenti e l'ultimo token del contesto riesce a porre attenzione fino a 1023 token che lo precedono nella riga di contesto.  
In GPT4 il vacabolario dei tokens e circa 100k, che vuol dire avere una dimensione dei testi encodati più piccola in input.  
Questo è vantaggioso perché un token, all'interno del contesto, riesce a porre attenzione a circa il doppio del numero dei token di GPT2, considerando lo stesso testo in input. Avremo, cioè, un input molto più **denso** per il transformer di GPT4.  
Bisogna ricordare che in output del transformer viene applicata una decodifica per trasformare i token numerici in testo e,    
avere vocabolari di token molto grandi, non aiuta in questo processo di decodifica, perché viene applicato **softmax** su un grande numero di valori, che è inefficiente.  

* Esempio: in GPT2 i 4 spazi che si usano per tabulare il codice python vengono codificati con un singolo token che considera un solo spazio. (1 token totale)  
In GPT4, invece, si usa un token per codificare una sequenza di 3 spazi e un altro token per codificare l'altro spazio singolo.  (2 token totali)

In internet troviamo caratteri di lingue diverse, emojii etc., avremo, cioè, un alfabeto di di circa 150k possibili caratteri, per cui l'unico modo per considerarli tutti è usare la loro rappresentazione in **unicode**, che associa ad ognuno di questi caratteri un intero.  

Per ottenere il codice unicode di un carattere, in python:
```py
ord('c') # 99

# rappresentazione unicode di una stringa in un array di interi:
[ord(x)for x in "안녕하세요  (hello in Korean!)"] # [50504, 49656, 94567, ...]
```

#### Binary data 
In realtà, non usiamo direttamente il valore intero unicode, ma una sua codifica **utf-8**, per codificare gli interi in **byte streams**.  
Esistono anche altre due codifiche, **utf-16** e **utf-32**, ma sono inefficienti per i nostri scopi, sostanzialmente perché di lunghezza fissa.  
Il punto di forza di utf-8 è la sua lunghezza variabile, tra 1 e 4 byte.  

Ogni carattere stringa, codificato in utf-8, può assumere la forma di una sequenza tra 1 byte (da 0 a 256) e 4 bytes (4 numeri tra 0 e 256).  
In pratica per codificare i primi 256 caratteri dell'alfabeto latino basta 1 byte, per gli altri > 256 servono altri bytes.  
**La codifica in utf-8 di qualsiasi testo, quindi, è sempre un array di interi compresi tra 0 e 255**.  

es.:
```py
# encoding di caratteri:
list('a'.encode('utf-8')) # [97], sequenza di 1 byte, perché ord('a') = 97
list('요'.encode('utf-8')) # [236, 154, 148] sequenza di 3 bytes, perché ord('요') = 50836

# encoding in utf-8 della stringa:
list("안녕하세요 👋 (hello in Korean!)".encode("utf-8")) # [236, 139, 154,...]
```
#### Byte pair encoding
E' un [Algoritmo](https://en.wikipedia.org/wiki/Byte_pair_encoding) usato per comprimere **le coppie di due caratteri** (pair) che compaiono più frequentemente in una stringa.  La ricerca viene fatta iterativamente per trovare la coppia in assoluto più frequente che viene sostituita con un carattere (token che verrà aggiunto al vocabolario di tokens), per poi cercare di nuovo la nuova coppia più frequente, sostituirla con un nuovo token e così via.  
Di fatto viene creato un **tokenizzatore** di testi.  

Esempio:  
aaabdaaabac --> coppia più frequente 'aa' --> codificata con token 'Z'  
ZabdZabac --> coppia più frequente 'ab' --> codificata con 'Y'  
ZYdZYac ->  coppia più frequente 'ZY' --> codificata con 'X'   
XdXac  

A partire da una stringa lunga 11 caratteri ne abbiamo ottenuto una compressa, lunga 5 caratteri, usando il vocabolario di token:
```py
{
    Z: 'aa',
    Y: 'ab',
    X: 'ZY'
}
```
Tale algoritmo, eseguito su un testo di input, compie quello che possiamo considerare un training del tokenizzatore.  
Implementiamo l'algoritmo in python:

```py
# funzione che trova coppia più frequente
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
print(sorted(((v,k) for k,v in stats.items()), reverse=True)) # elencati per max num di occorrenze
# es.: [(20, (234, 156)), (...)...] # la coppia (234, 156) compare 20 volte..
```

Identificata la coppia più frequente possiamo sostituirla con un nuovo token che verrà aggiunto al vocabolario dei tokens.  
Il primo nuovo token avrà **numero 256**, perché il testo originario codificato in utf-8 è un array di interi nel range [0, 255]  
Iterando troveremo una nuova coppia più frequente, sostituita dal nuovo token 257 e così via.   

```py
# funzione che sostituisce all'interno della lista di tokens (ids), la coppia più frequente (pair) con il nuovo token (idx)
def merge(ids, pair, idx):
  # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
  newids = []
  i = 0
  while i < len(ids):
    # if we are not at the very last position AND the pair matches, replace it
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# esempio d'uso
print(merge([5, 6, 6, 7, 9, 1], (6, 7), 99))
#  [5, 6, 99, 9, 1] # al posto della coppia 6, 7 c'è 99
```

le due funzioni vanno eseguite in loop un certo numero di volte che determinerà la grandezza del vocabolario dei tokens.  
La grandezza del vocabolario è a tutti gli effetti un **iperparametro** del tokenizzatore.  

```py
vocab_size = 276 # hyperparameter: the desired final vocabulary size,
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

# output example:
# merging (101, 214) into a new token 256
# merging (10, 54) into a new token 257
# merging (112, 140) into a new token 258
# merging (26, 134) into a new token 259
# merging (10, 164) into a new token 260 
# ....

# print some stats.. example: 
print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

# possible output:
# tokens length: 24597
# ids length: 19438
# compression ratio: 1.27X
```

Notiamo che il tokenizzatore è un **componente completamente separato** dal LLM!  
Possiamo eseguire la fase di training del tokenizzatore con un dataset di testi appositamente per creare il vocabolario di tokens, senza considerare il transformer del LLM.  

da minuot 39






