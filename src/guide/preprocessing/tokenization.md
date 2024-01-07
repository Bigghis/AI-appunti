# String Tokenization

Conversione di un testo, che è una sequenza di caratteri, in una sequenza di numeri interi.  
Per semplicità non consideriamo le tokenizazzioni del tipo **subpieces** che convertono sottostringhe in interi.  

Per riferimento della conversione viene usato un **vocabolario di tokens**, esempio i caratteri dell'alfabeto, oppure tutti i caratteri tipografici presenti in un testo, presi una sola volta:  
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

