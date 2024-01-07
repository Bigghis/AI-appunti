# Output Loss Functions
Funzione che serve per mostrare la qualità di una rete neurale.  
Con un singolo numero, riusciamo a misurare la performance della rete.  
Tale numero viene detto **loss**, misura la perdita della rete, cioè l'entità di scostamento tra le previsioni effettuate dalla rete e le previsioni corrette che ci si sarebbe aspettati in realtà.

Ovviamente lo scopo è quello di **minimizzare il valore del loss** rendendolo un numero vicino a 0.  
Vogliamo **loss bassi**, basse perdite!  

In genere in uscita della rete abbiamo la probabilità espressa dalla rete stessa per un dato output.  


**Quindi in output avremo una lista di probabilità, per la lista dei dati di addestramento fornita in input.**


Se consideriamo una rete **bigram**, in output avremo le probabilità del carattere successivo previsto.  
Possiamo avere 27 possibili output (alfabeto) e, se fossero tutti equiprobabili, la probabilità associata ad ognuno
sarebbe 1 / 27 = 0,037 (il 4% circa).  
Avere una probabilità maggiore del 4% per un dato output, vuol dire che la rete ha imparato qualcosa riguardo a quell'output.  
L'ideale sarebbe avere una **probabilità pari a 1 per ogni output** che la rete produce.  

Se per esempio vorremmo come previsione prevista un output di 1.0,
mentre la rete dà in output 0.8, la perdita della rete è 1 - 0.8 = 0.2  
Per far sì che loss sia un numero singolo, vengono **sommate le perdite di tutte le uscite della rete**.  

I del loss vengono ottenuti attraverso alcune funzioni ben gestibili,
errore quadratico medio, negative log likelihood loss, etc. etc.


