# Self-Attention

Abbiamo visto che riusciamo a relazionare i token [0, n-1] precedenti a quello attuale, sulla stessa riga di batch, sommando e calcolando la media dei valori. Questo modo, però, tratta le relazioni tra i token in modo **uniforme**, che non è quello che vogliamo, perché ci sono token che si relazionano meglio tra di loro, rispetto ad altri.  
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
Siccome query e key sono prodotti a partire dal'input x del batch, si parla di **attenzione verso sé stesso o self-attention**.  
Tuttavia, mentre le query devono sempre essere create a parire da x, keys e values possono provenire anche da risorse esterne (encoder esterni) ad x, in quel caso parleremo di **cross attention**.  


#### Implementazione (Single Head)
Viene creato un blocco chiamato **Head** che al suo interno esegue self-attention.  
Introduciamo un nuovo iper parametro **head_size** e i layer lineari **query** e **key**.
La dimensione di questi layer è [C, head_size], in quanto limiteremo l'elaborazione a head_size.  
Verranno poi eseguite le \__call()__ dei due nuovi layer in modo che ogni token produca una propria query e una propria key.   
Per trovare le affinità effettuiamo il prodotto scalare tra q e la trasposta di k.  
Questo prodotto è il tensore **wei**, che fino ad ora avevamo creato inizializzandolo con tutti zeri.  (**wei = torch.zeros((T,T))**)  
Con questo nuovo modo di inizializzarlo otterremo affinità più realistiche.  

Nel modello dell'attention, è previsto anche un layer lineare **value**, dove sono memorizzate le informazioni riguardanti X, con cui trattare i risultati del DOT product.  
Per ottenere l'aggregazione dei token facciamo:
**v = values(x)**  
**out = wei @ v** 


```py
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# Head
head_size = 16 
query = nn.Linear(C, head_size, bias=False)
key = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, head_size)  # key of token x
q = query(x) # (B, T, head_size) # query of token x

# calcolo delle affinità
# prodotto di q per la trasposta di k.
# k viene trasposta nelle sue ultime due dimensioni
# ottenendo, al posto di wei = torch.zeros((T,T)):
wei = q @ k.transpose(-2,-1) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)


tril = torch.tril(torch.ones(T, T))

wei = wei.masked_fill(tril == 0, float('-inf')) # decoder
wei = F.softmax(wei, dim=-1)

# calcola l'aggregazione dei token
v = value(x)
out = wei @ x # (B, T, head_size)

# out.shape: [4, 8, 16]

```

### Consideraziooni su Attention
Possiamo concludere dicendo che l'Attention è un meccanismo di comunicazione tra i nodi.  
Non tiene conto della disposizione spaziale dei nodi, perciò è necessario creare una codifica in embeddings anche per le posizioni di tali nodi, per fornire delle informazioni riguardo a posizioni dei token, per sapere dove si torovano.  
Le righe di un batch non comunicano tra di loro, per cui i token di una riga non comunicano con i token di altre righe.  
Se, ad esempio, B = 4, T = 8,  avremo 4 pool separati di 8 nodi. GLi 8 nodi parlano tra di loro, ma non comunicano con gli altri 8 nodi degli altri 3 pool del batch.  

#### Encoder 
In generale, possiamo anche pensare che tutti i token della riga possano comunicare tra di loro, anche quelli futuri, successivi al token n, non solo quelli precedenti.  
In questo caso si parla di **encoder block** del self-attention e corrisponde al blocco già implementato sopra, eliminando il vincolo di non poter considerare i token sucessivi ad n.  
In pratica togliamo l'istruzione **wei = wei.masked_fill(tril == 0, float('-inf'))** per eliminare tale vincolo e permettere la comunicazione tra tutti i nodi del contesto.


#### Decoder
Aggiungendo l'istruzione **wei = wei.masked_fill(tril == 0, float('-inf'))** al blocco del self-attention otteniamo un **decoder block**, così chiamato perché la matrice triangolare aggiunta, crea una sorta di linguaggio di decodifica.  


#### Scaled attention 
L'applicazione del softmax per valori di query e key, soprattutto nell'inizializzazione, è importante che mantenga **diffusi** i valori, non creando picchi verso il centro, che convergono verso one-hot, che causerebbero saturazione, c
es.: valori diffusi: [0.1925, 0.1426, 0.2351, 0.1426, 0.2872]  
valori con picco centrale: [0.0326, 0.0030, 0.1615, 0.0030, 0.8000], 0.1615 è molto più alto degli altri valori, è ancvhe il doppio dell'ultimo valore, non è accettabile.  
Per limitare tale effetto si usa scalare **wei** della radice quadrata:

```py
# da: 
wei = q @ k.transpose(-2,-1) 
# a versione scalata;
wei = q @ k.transpose(-2,-1) **0.5

```
