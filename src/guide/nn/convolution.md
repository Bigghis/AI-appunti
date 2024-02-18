# Convolution

La convoluzione è una tecnica usata soprattutto in **computer vision**.  
Prevede dati di input, moltiplicazioni ed addizioni molto simili alle moltiplicazione di matrici già viste, ed è quindi molto simile ad una rete neurale, come quelle già viste. 

Sappiamo che un'immagine può essere rappresentata, per esempio, con una matrice di rank 2, contenente numeri che rappresentano i pixel con associata informazione del colore.  

Se consideriamo una piccola matrice (es.: 3 x 3), detta **kernel**, di rank 2, e la applichiamo sulla matrice dell'immagine, facendola **scorrere** attraverso l'immagine, ad esempio orizzontalmente, possiamo effettuare un prodotto scalare (**DOT product**) tra queste due matrici.  

L'**operazione di convoluzione** esegue un prodotto scalare tra una porzione 3x3 della matrice-immagine ed il kernel ivi sovrapposto.  

![Convolution scheme](../../images/convolution.png)

Considerando le immagini del db **MNIST**, che contiene immagini di caratteri alfanumerici scritti a mano, possiamo applicare dei kernel e crearne la convoluzione, per riconoscere i tratti caratteristici (linee, angoli) nelle immagini-caratteri, procedendo per tipologia di tratto.  

Possiamo far scorrere un kernel del tipo:
```py
horiz_edge = tensor([ [-1,-1,-1],
                                [ 0, 0, 0],
                                [ 1, 1, 1]]).float()
```
In questo modo il kernel agisce da filtro per le linee orizzontali dell'immagine-carattere, facendole emergere quando l'operazione di convoluzione dà il risultato più alto (**attivazione**), durante lo scorrimento del kernel.  
Quando lo scorrimento sarà completato coprendo l'intera immagine, avremo una **mappa di attivazione** per rilevare le linee orizzontali.  

Possiamo creare un nuovo kernel per filtrare le linee verticali:
```py
vert_edge = tensor([  [ 1, 0, -1],
                                [ 1, 0, -1],
                                [ 1, 0, -1]]).float()
```
che le farà emergere sempre facendolo scorrere orizzontalmente. Si possono creare anche altri kernel per filtrare gli angoli etc.  

## Padding e Stride

Se vogliamo che la mappa di attivazione abbia le stesse dimensioni dell'immagine considerata, c'è la necessità di aggiungere pixel (**padding**) lungo i bordi dell'immagine, per non vanificare il DOT product proprio lungo la riga e colonna del primo e ultimo pixel dell'immagine.  
Si può anche effettuare lo scorrimento non pixel per pixel, ma aumentando il passo, es. due pixel alla volta **stride = 2**.  
In questo modo avremo una mappa di attivazione che considera meno righe e meno colonne della matrice-immagine, creando una mappa di attivazione compressa, che contiene, ad esempio per stride=2, la metà dei risultati dei DOT product per le righe e la metà per le colonne.  

## Pytorch conv2d

La convoluzione in pytorch è implementata tramite **F.conv2d**, che applica la convoluzione all'intero **batch di immagini** in parallelo, applicando più kernel nello stesso tempo.  
E' quindi molto veloce!

Supponendo di avere 4 kernel 3x3, possiamo metterli in un unico tensore:
```py
diag1_edge = tensor([[ 0,-1, 1],
                     [-1, 1, 0],
                     [ 1, 0, 0]]).float()

diag2_edge = tensor([[ 1,-1, 0],
                     [ 0, 1,-1],
                     [ 0, 0, 1]]).float()

# concatenate 4 kernels:
edge_kernels = torch.stack([horiz_edge, vert_edge, diag1_edge, diag2_edge]) # torch.Size([4, 3, 3])
```

per poi eseguirne la convoluzione con il batch:

```py
#example: xb = batch of 64 images
batch_features = F.conv2d(xb, edge_kernels)
```

## Convolutional Neural Network (CNN)
Abbiamo visto che possiamo creare i kernel più disparati e non sappiamo se uno sia migliore di un altro!  
Possiamo, però, farci aiutare dalla discesa del gradiente nel cercare di trovare i numeri adeguati da mettere all'interno di un kernel.
