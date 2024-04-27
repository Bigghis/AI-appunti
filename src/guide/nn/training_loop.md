# Training loop

Generalmente l'implementazione di base di un **training loop** in pytorch prevede una funzione **fit()** che si occupa del training e della valutazione del modello.  

All'interno del training loop viene eseguito per ogni epoch, prima un **addestramento** usando tutti i batch di **valid_dl** e poi un'**inferenza** usando tutti i batch di **valid_dl**.

Nel dettaglio il training loop itera per ogni **epoch** attraverso i seguenti passaggi:  

### Addestramento 
* il modello viene settato in modalità **train**
* vengono presi tutti i batch del training set e su ogni batch viene internamente eseguito il **forward()** del modello, producendo una previsione in output.
* Viene calcolata la **loss function**, mettendo a confronto la previsione prodotta dal modello con l'effettiva e reale label (target) del batch di training
* Viene eseguito il **backward()** per calcolare tutti i gradienti dei parametri del modello
* Viene fatto l'update dei parametri (aggiornamento dei pesi o **step()**) sottraendo i gradienti appena calcolati
* Vengono azzerati i gradienti appena calcolati  

### Inferenza 
Arrivati a questo punto, con i valori dei pesi del modello settati dopo l'iterazione su **TUTTI** i batch di training,
possiamo valutare il modello con i dati di validazione usando quei particolari pesi.
Quindi la stessa logica va applicata in modo simile per il validation dataloader, avendo accortezza di **non calcolare i gradienti**:
* il modello viene settato in modalità **eval**
* vengono presi tutti i batch del validation set e su ogni batch viene internamente eseguito il **forward()** del modello che è stato addestrato con tutti i batch del dataloader di training,  producendo una previsione in output.
* Viene calcolata la **loss function**, mettendo a confronto la previsione prodotta dal modello con l'effettiva e reale label (target) del batch di validazione
* Vengono eseguiti i calcoli per determinare il valore totale della loss function e l'accuratezza del modello e vengono stampati.  

```py
def accuracy(out, yb): return (out.argmax(dim=1)==yb).float().mean() # utility to print accuracy value

def fit(epochs, model, optimizer, loss_fn, train_dl, valid_dl):
    
    for epoch in range(epochs):
        model.train()

        for batch_index, (X_batch, y_batch) in enumerate(train_dl):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():  
            tot_loss, tot_accuracy, count = 0.,0.,0
            
            for X_valid, y_valid in valid_dl:        
                
                val_probs = model(X_valid)
                val_loss = loss_fn(val_probs, y_valid) # labels.view(val_probs.shape)
                
                n = len(X_valid)
                count += n
                
                tot_loss += loss_fn(val_probs, y_valid).item()*n
                tot_accuracy  += accuracy (val_probs, y_valid).item()*n

        print(f'{epoch} loss = {tot_loss/count}, accuracy = {tot_accuracy/count}')

    return tot_loss/count, tot_accuracy/count
```

Per eseguire la funzione **fit()**, consideriamo di avere già un dataloader, un modello, un ottimizzatore e una loss function:

```py
learning_rate = 0.1
train_dl, valid_dl # train dataloader e validation dataloader
model = cnn # cnn è una rete pytorch
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=learning_rate)

n_epochs = 100

fit(n_epochs, model0, optimizer, loss_fn, train_dl, valid_dl)

# print output d'esempio: 
# 0 loss = 0.5227519570827485, accuracy = 0.8274
# 1 loss = 0.2518924878358841, accuracy = 0.9262
# 2 loss = 0.18806885157227515, accuracy = 0.9428
# ....
# 46 loss = 0.06776001966204494, accuracy = 0.9803  # nota come diminuisce il loss ed aumenta l'accuratezza...
```