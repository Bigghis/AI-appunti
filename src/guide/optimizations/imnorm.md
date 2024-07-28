# Images Normalization


Spesso si usa normalizzare le immagini in input, per maggiore efficienza durante il training e per scongiurare pericoli di dead neurons ed altri effetti indesiderati.  

Partendo dal presupposto che un'immagine è di una certa altezza x larghezza ed è disposta in 3 channels (RGB), 
può essere immagazzinata in un tensore che rappresenta i pixel scalati in un range [0.0, 1.0]  

Per normalizzare un'immagine si sottrae dal pixel x la media dei pixel e poi si divide per la deviazione standard.  
Per normalizzare le immagini si possono usare delle medie e varianze già disponibile e precalcolate a partire dal database **ImageNET** che contiene milioni e milioni di immagini varie.

le medie e le varianze  dell'imagenet per i 3 channels sono:  
* mean = [0.485, 0.456, 0.406] 
* std = [0.229, 0.224, 0.225]  

In alternativa si possono anche ricavare media e deviazione standard a partire dal proprio dataset di immagini, invece di usare i valori predefiniti.  

Creiamo una funzione da applicare a tutte le immagini del dataset, allo scopo di normalizzarle:
```py
# img. shape = tensor([3, 256, 256]) # immagine 256x256 pixel in 3 channels

def normalize(img):
    imagenet_mean = tensor([0.485, 0.456, 0.406])[:,None,None].to(img.device)
    imagenet_std = tensor([0.229, 0.224, 0.225])[:,None,None].to(img.device)
    return (img - imagenet_mean) / imagenet_std  # operazione di normalizzazione
```

Oppure, torchvision transforms mette a disposizione la funzione Normalize:

```py
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```


