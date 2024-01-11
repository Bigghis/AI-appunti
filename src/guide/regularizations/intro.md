# Regularization

La regolarizzazione riguarda la limitazione dei valori che i parametri possono assumere durante il training o anche durante l'ottimizzazione della rete.  
Ricordiamo, per esempio, che **batch normalization** produce un side effect di regolarizzazione perché le sue medie nei buffer limitano i valori dei dati casuali del batch, durante la fase di training.  

In genere si usa regolarizzare reti neurali grandi e profonde in presenza di pochi dati disponibili in input per il training. In questo caso, infatti, le reti neurali non riescono a **generalizzare** perché possono memorizzare facilmente il training set usato che, in quanto piccolo, non è rappresentativo dei dati reali, e quindi produce **overfitting**.
