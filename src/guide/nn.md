# Neural Networks

Una rete neurale è un'insieme di **espressioni matematiche**.  
Questo insieme di espressioni prende in input dei numeri (pesi, parametri etc.) ed esegue su tali numeri delle elaborazioni dette **passaggio in avanti (forward)**,
trasformandoli in output.  
Tra questo insieme di espressioni si trova anche una **funzione di perdita**.  
All'interno di una rete neurale ci sono moltissimi valori numerici, ma sono riconducibili in due sole categorie:  
* **Numeri calcolati** (dentro i layer lineari e non lineari)  
* **Parametri della rete** che sono, per esempio, inizializzati randomicamente e poi vengono ottimizzati. Sono i numeri che definiscono il **modello**.  



### Neurone
Alla base della rete c'è un'entità chiamata **neurone**.  
E' una struttura che prende in ingresso uno o più **input** (i), lo processa usando una funzione matematica **(funzione di attivazione)** e restituisce quindi un **singolo output** (x) in base ai calcoli svolti.  
Gli input vengono processati tramite l'equazione:  

**x = (w\*i) + b** *, con w = peso, b = bias*

* un **peso (w)** è un fattore che viene modificato durante l'addestramento.  

* un **bias (b)** è un valore che viene modificato durante l'addestramento che fa da **soglia di attivazione** del neurone.

La **funzione di attivazione**, può essere, per esempio:

**x = tanh( (w\*i) + b )**, in quanto viene prima calcolato l'output x e poi viene processato da tanh() che squasha il risultato tra -1 e +1.  
Questa funzione dell'esempio, si utilizza nelle reti dette **classificatori binari**.

### Layer
I neuroni sono raggruppati in **layer**. I neuroni dello stesso layer **non comunicano** tra di loro.  

In generale possiamo avere tre tipi di layer:  
* **input layer**: riceve i dati di input dall'esterno e li invia ai layer interni. In teoria i pesi dei neuroni sono inesistenti, oppure sono calcolati randomicamente, 
perché l'informazione (input) viene inviata dentro la rete per la prima volta.

* **hidden layers**: uno o più layer interni della rete, che prende un set di pesi e eseguendo una fuinzione di attivazione produce un output. I pesi possono essere definiti o 
randomicamente oppure attraverso un processo di **backpropagation** che li calibra opportunamente  

* **output layer**: produce l'output della rete prendendo il risultato dagli input layers e trasformandolo in output finale.  

I layer possono essere di tipo:  
* **lineare** che applica una trasformazione del tipo: **y = x*W + b** (W = pesi, b = bias, è l'equazione di una retta)
* **non lineare**

### Funzione di perdita (loss function)
è una misura matematica che valuta di quanto si adatta un modello di machine learning ai dati di addestramento.  
In altre parole, una funzione di perdita **quantifica l'errore** tra le previsioni del modello e i valori target desiderati nei dati di addestramento.  
L'obiettivo principale di una funzione di perdita è fornire una metrica che può essere **minimizzata** durante il **processo di addestramento del modello**.  
Il modello cerca di ridurre l'errore calcolato dalla funzione di perdita in modo da migliorare le sue prestazioni.  
Agire sui parametri serve per diminuire la perdita.  
Agire sui parametri va fatto molte volte per provocare una **discesa del gradiente**, o
**Stochastic Gradient Descent (SGD)**.    
Quando la perdita è minima, la rete fa quello per cui è stata progettata.  

### Backpropagation 
La perdita viene propagata all'indietro, dal layer di output, verso il layer di input, per ottenere i gradienti.  
Backpropagation è un **algoritmo per calcolare i gradienti delle funzioni di perdita** rispetto ai pesi delle reti neurali, consentendo di aggiornare i pesi in modo da ridurre la perdita e migliorare le prestazioni del modello della rete.  
Nelle reti neurali, il calcolo del gradiente è essenziale per il processo di backpropagation.  
Questo processo calcola in che modo gli errori si propagano all'indietro attraverso la rete, consentendo al modello di adattare i suoi pesi in base all'errore commesso.  

Un algoritmo di Backpropagation prevede le seguenti fasi:  

1) **Forward Pass**: Durante la fase di **"passaggio in avanti"**, i dati di addestramento vengono alimentati nella rete neurale e propagati attraverso di essa. Ogni strato della rete applica una serie di trasformazioni ai dati in ingresso, producendo una previsione o un'output del modello.

2) **Calcolo della funzione di perdita**: Una volta ottenuto l'output del modello, si calcola una funzione di perdita che misura la discrepanza tra l'output previsto e l'output desiderato. La funzione di perdita può variare a seconda del tipo di problema che la rete neurale sta risolvendo (ad esempio, regressione o classificazione).  
Si cerca di misurare l'accuratezza delle previsioni, più la perdita calcolata è bassa, più le previsioni coincidono con i tuoi obiettivi, più, cioè, la rete si sta comportando bene.
Può essere manipolata per abbassarne la perdita.

3) **Backward Pass**: La fase cruciale della backpropagation inizia con il calcolo dei gradienti della funzione di costo (o di perdita) rispetto ai pesi di ciascun neurone in ogni strato della rete. Questi gradienti vengono calcolati utilizzando la regola della catena (regola di derivazione delle funzioni composte) e iniziano dallo strato di output e si propagano all'indietro attraverso i livelli nascosti della rete.

4) **Aggiornamento dei pesi**: Una volta calcolati i gradienti, gli algoritmi di ottimizzazione (come la discesa del gradiente), vengono utilizzati per aggiornare i pesi dei neuroni in ogni strato della rete. L'obiettivo è minimizzare la funzione di perdita, in modo che l'output del modello si avvicini il più possibile all'output desiderato. 

