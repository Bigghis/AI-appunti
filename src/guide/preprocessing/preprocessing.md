# Preprocessing Input Data

Spesso i dati da dare in pasto alla rete, in input, sono disordinati o archiviati in formati arbitrari
ed è necessario preelaborarli o/e manipolarli in base alle nostre esigenze ed a quelle della rete neurale che stiamo considerando.  
Per esigenze di corretto funzionamento della rete neurale, è necessario trovare dei **sistemi di codifica dei dati**, adeguati allo scopo.  

La **tokenizzazione**, ad esempio, è un sistema per trasformare stringhe di testo in sequenze di numeri.  
Le tokenizzazioni possono essere fatte a livello di singoli caratteri oppure a livello di sottostringhe.  
Esempio: possiamo considerare come **tokens** i 27 caratteri dell'alfabeto, per codificare i singoli  caratteri delle stringhe.  
Oppure possiamo considerare le 50.000 sottostringhe usata da gpt2, tramite la libreria di OpenAI **tiktoken**, che codifica sottostringhe dei testi.  

