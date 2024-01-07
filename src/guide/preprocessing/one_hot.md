# Codifica One Hot


Una rete neurale non gestisce facilmente gli interi di xs in input, che rappresentano un carattere alfabetico, per cui va fatta una ulteriore codifica,
usando un'**array one hot** che è un array di 27 elementi, composto  
di tutti elementi 0, tranne quello che corrisponde
alla posizione del numero da codificare, che è 1.  
Per cui, ad es. il carattere 'm', corrispondente a 13, corriponde a sua volta al tredicesimo valore dell'array one hot, che avrà valore 1.  

m = 13 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]


```py
# ricordando che abbiamo mapato tutti i caratteri dell'alfabeto (da 0 a 26) in itos,
# il primo elemento del tensore xs è indice 0 che corrisponde all'inizio della parola "."
# il primo elemento di ys corrisponde al carattere successivo VOLUTO in corrispondenza di xs[0]
# che è 'e'

# , per cui vanno codificati in ONE HOT
# e poi dati in pasto alla rete.
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()
xenc

tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0.]] ... .. .. )
```
