# Self-Attention

Abbiamo visto che riusciamo a relazionare i token [0, a n-1] precedenti a quello attuale, sulla stessa riga di batch, sommando e calcolando la media dei valori. Questo modo, però, tratta le relazioni tra i token in modo **uniforme**, che non è quello che vogliamo, perché ci sono token che si relazionano meglio tra di loro, rispetto ad altri.  
Per esempio, un token che rappresenta una vocale, si relaziona meglio con i token del passato che rappresentano consonanti!  
Vogliamo, cioè, ottenere informazioni che siano **dipendenti** dai dati passati.  
L'algoritmo di **Self-Attention** risolve questo problema.  
Ad ogni singolo token sono associati **2 vettori**:  
* **query** rappresenta quello che sto cercando (es.: sto cercando un token consonante)
* **key**  rappresenta cosa contiene il token (es.: sono un token vocale) 

Facendo un prodotto scalare (**DOT Product**) tra **queries** e **keys** otteniamo le **affinità** tra i token.  
In particolare, per un token, trovare le affinità con gli altri token equivale ad eseguire 
Il prodotto scalare della query del token con le keys di tutti gli altri token e 
il prodotto scalare della key del token con le queries di tutti gli altri token.  

Se queries e keys di vari token sono in qualche modo allineate tra loro, imparerò di più su quei token, rispetto ad altri token.  


#### Implementazione (Single Head)
Viene creato un blocco chiamato **Head** che al suo interno esegue self-attention.  
Introduciamo un nuovo iper parametro **head_size** e i layer lineari **query** e **key**.
La dimensione di questi layer è [C, head_size], in quanto limiteremo l'elaborazione a head_size.  
Verranno poi eseguite le \__call()__\ dei due nuovi layer in modo che ogni token produca una propria query e una propria key.   
Per trovare le affinità effettuiamo il prodotto scalare tra q e la trasposta di k.  
Questo prodotto è il tensore **wei**, che fino ad ora avevamo creato inizializzandolo con tutti zeri.  (**wei = torch.zeros((T,T))**)  
Con questo nuovo modo di inizializzarlo otterremo affinità più realistiche.  



```py
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# Head
head_size = 16 
query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, head_size)  # key of token x
q = query(x) # (B, T, head_size) # query of token x

# calcolo delle affinità
# prodotto di q per la trasposta di k.
# k viene trasposta nelle sue ultime due dimensioni
# ottenendo, al posto di wei = torch.zeros((T,T)):
wei = q @ k.transpose(-2,-1) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)


tril = torch.tril(torch.ones(T, T))

wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)

```

