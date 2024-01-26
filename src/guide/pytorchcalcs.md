# Pytorch Gradients Calcs

Pytorch calcola automaticamente i gradienti dei parametri rispetto alla loss function.  

Per abilitare il calcolo automatico dei gradienti basta settare il flag **requires_grad**.  
```py
for p in parameters:
    p.requires_grad = True

```

Questo ci assicura che ogni volta che usiamo un tensore all'interno di una funzione, ne viene calcolato in automatico il gradiente.   
Il gradiente calcolato viene salvato nell'attributo **.grad** del tensore.  

es.:

```py
t = torch.tensor([23.0], requires_grad=True)

t -= t.grad * 0.01 # pytorch calcola in automatico il gradiente di t
loss = quad_mse(t)
```

In caso vogliamo inibire il calcolo del gradiente di t, usiamo il contesto **no_grad()**:  
```py

with torch.no_grad():
    t -= t.grad * 0.01 # non viene calcolato il gradiente di t
    loss = quad_mse(t) 
```  

Questo ultimo esempio è un tipico utilizzo di no_grad(), perché  vogliamo aggiornare il valore dei gradienti riducendoli di un tot (es.: 0.01) in ogni loop di training per poi ricalcolare la loss function, **ma NON vogliamo ricalcolare il gradiente di t, Ci interessa solo ridurlo di una piccola quantità!**  

### Loop del processo di training
Per automatizzare il processo di training in loop possiamo eseguire i seguenti passaggi in sequenza :    
* calcolo della loss function  
* esegui **backward()** per calcolare i gradienti rispetto alla loss function
* sottrai una piccola quantità dai gradienti (senza ricalcolare il gradiente!) per far scendere il suo valore (**descent gradient**)
* stampa risultato delle elaborazioni  
```py
for i in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.01
    print(f'step={i}; loss={loss:.2f}')
```

Questo ciclo riduce il valore della loss function dopo 10 passaggi. Applica il principio della **discesa del gradiente**.  E' un esempio di **ottimizzatore**.  