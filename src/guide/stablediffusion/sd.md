# Stable Diffusion

Modello di A.I. generativa che produce immagini a partire da **prompt di testi e  immagini**.   

Immaginiamo di avere un **modello** addestrato con le immagini 28x28 del DB **MNIST**,  che prende in input una qualsiasi immagine e produce in output la probabilità
che tale immagine sia una cifra numerica scritta a mano.

![sd0](../../images/sd0.png)


In che modo si potrebbe sfruttare tale modello per farlo produrre in output **nuove immagini** che siano delle cifre numeriche scritte a mano?  

Potremmo suddividere una immagine 28x28 su una griglia di 784 pixel totali e, pixel per pixel, potremmo provare a scurire o sbiancare un pixel, dare l'immagine modificata in input al modello e vedere se in output la probabilità che sia una cifra numerica aumenta o diminuisce.  
Possiamo eseguire tale procedimento per tutti i 784 pixel dell'immagine.  

In altri termini, il modello calcolerà **i gradienti** relativi a tutti i pixel dell'immagine rispetto al fatto che 
sia una cifra numerica scritta a mano.

Possiamo considerare gli input come se fossero composti da 2 immagini sovrapposte:
la cifra numerica + **rumore**

![sd1](../../images/sd1.png)

Quindi, di conseguenza, possiamo pensare che il modello preveda in output **quanto rumore** è stato aggiunto alla cifra numerica.  
**La quantità di rumore** ci dice quanto l'output somiglia ad una cifra numerica.
Se c'è poco rumore avremo una cifra numerica, altrimenti no.  

Quello che il modello cerca di fare è restituire in output la quantità di rumore presente in input,
e applicando una **loss function MSE** tra rumore predetto e rumore in input, si riesce a capire di quanto
il modello riesca a prevedere fedelmente le cifre numeriche e quindi a crearle.  
Possiamo prendere le immagini del MNIST, applicare un layer di rumore random sopra di esse e istruire il modello con queste immagini composte, calcolare poi la loss function ed aggiornare i pesi del modello.  

## U-Net
Una volta che il modello è addestrato, possiamo anche dare in input **solo rumore** ed il modello restituirà
in output del rumore che allontana di molto l'ìmmagine in input dall'essere una cifra.  
Sottraendo tale output all'input, avremo qualche pixel che, in effetti, fa sembrare il nuovo input più simile ad un cifra numerica e in output avremo del nuovo rumore da sostituire all'input.  
Effettuando questa procedura più volte, per sottrazione, ricaveremo un'immagine chiara di una cifra numerica!  
Un rete del genere è chiamata **U-Net** e sfrutta internamente la convoluzione.

Qui è mostrato il suo funzionamento per sottrazione di rumore:
![sd2](../../images/sd2.png)

## VAE Autoencoder
Nella realtà non si usano immagini 28x28, ma molto più grandi! Per tenere il numero di calcoli dei gradienti dentro limiti accettabili, possiamo ridurre il numero di pixel dell'immagine, usando **versioni compresse di immagini in input**, sfruttando il funzionamento del **VAE Autoencoder**, che ci assicura che non si abbiano perdite di informazioni relative all'immagine, durante il processo di compressione/decompressione.

Avremo, quindi, una compressione dell'input attraverso un **VAE encoder** che produce un **latent** (tensore di immagine compressa).   
Il latent costituisce l'input della rete U-net che restituirà in output un nuovo latent.  
il latent in output verrà decompresso attraverso un **VAE decoder**.  

![sd3](../../images/sd3.png)