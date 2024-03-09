# Characters and texts

## Codifica dei caratteri in input

Immaginando di avere il modello **n-gram** , che sulla base di un set di sillabe in input, predice il carattere successivo
Es.: per n = 2 bigram, 3 trigram etc. etc...  
Come codifichiamo questi caratteri per darli in input alla rete?

i caratteri possibili sono 27 (alfabeto + carattere di inizio e fine parola).   
Creiamo due **dict** per mappare i caratteri ai numeri interi.


```py
# words è un elenco di varie parole (nomi femminili) che comprendono tutti i caratteri 
# dell'alfabeto (https://raw.githubusercontent.com/karpathy/makemore/master/names.txt)
# words = ['emma', 'olivia', 'ava', ... ]
chars = sorted(list(set(''.join(words)))) # i 26 caratteri del'alfabeto

# mettiamo i caratteri in un dict, enumerandoli.. mappando carattere e posizione
stoi = { s:i + 1 for i, s in enumerate(chars)}

# aggiungo caratteri speciali dell'inizio e fine riga
stoi['.'] = 0

# creiamo anche il dict con mapping inverso, da numeri a char
itos = {i:s for s, i in stoi.items()}

```
#### Training set
il **Training set** della rete (es.: **bigram**) è un insieme di caratteri creati a partire dalle parole in words.  
Es.: dalla prima parola otteniamo ['.e', 'em', 'mm' 'ma', 'a.']  
Il primo carattere di ogni elemento è il nostro input (xs), il secondo è l'output desiderato (ys) (carattere da prevedere)  
Questi caratteri dentro xs e ys vengono però **codificati** nei corrispondenti numeri della mappatura fatta sopra.  
Es.: 'm' è inserito dentro xs e ys con il numero 13

```py
xs = [] #input
ys = [] #output

# for w in words[:1]:
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1] #indice del carattere attuale
    ix2 = stoi[ch2] #indice del carattere successivo
    # print(ch1, ch2)
    xs.append(ix1)
    ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
```
avremo così i tensori xs e ys, di 228146 elementi ciascuno.  
