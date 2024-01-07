# Implementing Vectoring Calcs

Alcuni esempi di implementazione considerando i tensori:  

Consideriamo la parte finale del forward pass della rete makemore:  
```py
# forward pass
# ...
# ...
# n è la dimensione del batch = 32
# Yb è l'array degli output [32]
loss = -logprobs[range(n), Yb].mean()
```

logprobs ha dim. [32, 27] e dlogprobs dovrà avere stesse dimensioni.   
La media contribuisce alla derivata parziale di un elemento dentro logprobs per 1/n
e logprobs mette segno negativo davanti all'elemento.  
Quindi possiamo dire che:  
**dlogprobs = -1/n**  

In realtà gli unici elementi che saranno impattati sono quelli corrispondenti a **[range(n), Yb]**, mentre tutti gli altri non saranno rilevanti per la derivata.  
Possiamo inizializzare dlogprobs a zero e renderlo = -1/n solo nei punti che ci interessano  

```py
dlogprobs = torch.zeros_like(logprobs) # inizializza un tensore di stessa forma di logprobs a zero 
dlogprobs[range(n), Yb] = -1/n
```  

## moltiplicazione tra tensori
Consideriamo due tensori, **a[3, 3] * b[3, 1]** e  moltiplichiamoli tra loro:  
**c = a * b**,    
 
Per moltiplicare a e b viene prima **replicata** l'unica colonna di b n volte (**implicit broadcasting**),
quante sono le colonne di a, in modo da **coprire** completamente a,   
e quindi viene fatto il prodotto di ogni elemento:

a11\*b1  a12\*b1  a13\*b1  
a21\*b2  a22\*b2  a23\*b2  
a31\*b3  a32\*b3  a33\*b3

A causa della replica, abbiamo la situazione in cui, il nodo b1, (o b2 o b3) viene usato più volte nella rete.  
Per il calcolo del gradiente, quando un nodo viene usato più volte, i gradienti di quel nodo vanno **sommati tra di loro**,
in questo caso riga per riga.  
Quindi, la derivata parziale del tensore **a** rispetto a **c** è:  
**b.sum(1, keepdim=True)**

esempio: vogliamo calcolare **dcounts_sum_inv** a partire da questa situazione:
```py
# forward pass:
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

immaginando di aver già calcolato **dlogprobs** e **dprobs**, per calcolare **dcounts_sum_inv**, bisogna considerare **solo** l'espressione:  
**probs = counts * counts_sum_inv**   

e le forme dei tensori:

```py
probs.shape # [32, 27]
counts.shape # [32, 27]
counts_sum_inv.shape # [32, 1]
```

Nella moltiplicazione si ha una replica di counts_sum_inv, questo comporterà una somma dei gradienti riga per riga.  
la derivata parziale di **counts_sum_inv = counts** .  
Possiamo poi applicare la chain rule moltiplicando per **dprobs**.  
Alla fine applichiamo la somma dei gradienti riga per riga.  

```py
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
```

Allo stesso modo, per calcolare **dcounts**, considerando sempre **probs = counts * counts_sum_inv**: 
**dcounts = counts_sum_inv**   
Applichiamo sempre la chain rule, anche.  

**La replica NON influisce su dcounts, ma su dcounts_sum_inv, in quanto non replichiamo counts ma count_sum_inv.**

```py
dcounts = counts_sum_inv * dprobs
```

## utilizzo dello stesso nodo in più equazioni del forward pass.
Quando un nodo è usato più volte nella rete, i gradienti di tutti gli usi si sommano tra di loro. 

continuando l'esempio di prima:  
```py
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True) # <--- 1
counts_sum_inv = counts_sum**-1 
probs = counts * counts_sum_inv # <--- 2
```
il nodo **counts** è a destra del segno di uguale in 2 espressioni, è usato, cioè due volte nella rete.  
Il gradiente **dcounts** sarà dato dalla somma del gradiente dell'espressione **1** e del gradiente dell'espressione **2**.  
Continuando l'esempio di calcolo che stiamo facendo, al momento abbiamo calcolato solo il contributo **dcounts2** dell'espressione 2:  

```py
dcounts2 = counts_sum_inv * dprobs # contributo dell'espressione 2 
```
Per il contributo dell'espressione 1  **counts_sum = counts.sum(1, keepdims=True)**, le forme sono:  
```py
counts.shape # [32, 27]
counts_sum.shape # [32, 1] 
```
La colonna counts_sum va sommata  riga per riga.    

Possiamo implementare **dcounts1** per l'espressione 1 in questo modo:  

```py
dcounts1 = torch.ones_like(counts) * dcounts_sum
```
avremo che **dcounts** è dato da:

```py
dcounts = dcounts1 + dcounts2
```
