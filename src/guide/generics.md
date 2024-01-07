# Generalizzazioni del comportamento delle reti neurali

Progettare reti neurali significa trovare **patterns** nelle reti che funzionano bene in modo **generico** 
per ogni dato in input.  
Il problema è che abbiamo a che fare con **training set** limitati.  
In generale per evitare errori è bene avere a disposizione training set molto ampi. Troppi dati in input non fanno mai male :)  
Il successo delle odierne reti neurali è dovutoo proprio all'ampia disposizione di dati da usare per l'addestramento.  

Può capitare che una rete si adatti bene ai dati del **train set**, ma che poi commetta molti errori
su altri dati diversi da quelli del training set, es sui dati di valutazione del **dev set**.   
Questa perdita di generalità è chiamata **overfitting** e va combattuta tramite tecniche di **regolarizzazione**.

#### Overfitting

cause:
* La dimensione del set di dati di addestramento è troppo piccola e non contiene campioni di dati sufficienti per rappresentare accuratamente tutti i possibili valori dei dati di input.
* I dati di addestramento contengono grandi quantità di informazioni irrilevanti, chiamate dati rumorosi.
* Il modello viene addestrato troppo a lungo su un singolo set di dati campione.
* La complessità del modello è elevata, quindi il modello riconosce il rumore all'interno dei dati di addestramento

rimedi: 
* Eliminazione: Questa tecnica identifica le caratteristiche più importanti all'interno del set di addestramento ed elimina le caratteristiche irrilevanti.
* Aumento dei dati: Questa tecnica modifica leggermente i dati di esempio ogni volta che il modello li elabora.
* Riduzione delle funzionalità: Questa tecnica prevede la rimozione di alcune caratteristiche dal set di dati. 
* Regolarizzazione: Questa tecnica prevede l'introduzione di parametri di regolarizzazione, noti come fattori di penalità, che controllano i pesi, penalizzando i pesi più grandi e contribuendo a semplificare il modello.
* Ensembling: Questa tecnica combina le previsioni di diversi algoritmi di machine learning separati.
* Arresto anticipato: Questa tecnica interrompe la fase di addestramento prima che il modello di machine learning riconosca il rumore nei dati.



Viceversa, se la rete ha prestazioni scadenti sui dati del training set, allora si parla di **underfitting**.

#### Underfitting

cause:
* Modello molto semplce, con pochi parametri.
* Uso di un **modello lineare** per adattare una **relazione non lineare** tra le caratteristiche di input e la variabile di destinazione. I modelli lineari, come la regressione lineare, presuppongono una relazione lineare tra le caratteristiche e la variabile target. Se la vera relazione non è lineare, il modello non riuscirà a catturare la complessità dei dati

rimedi:
* Aumentare la complessità del modello
* diminuzione della regolarizzazione
* Aumentare la quantità dei dati di adestramento

Altri casi di Overfitting/underfitting ben noti sono:  
* Se l'errore sui dati di training è elevato, c'è sicuramente un problema di underfitting. Il modello ha generalizzato troppo.  
* Se l'errore sui dati di training è accettabile ma l'errore sul **test set** è elevato, c'è un problema di overfitting. Il modello non ha generalizzato abbastanza.



