# mean squared error (MSE)
Errore quadratico medio.  
Calcola il valore del loss, sottraendo il valore previsto dalla rete dal valore atteso.
Ci assicura che il loss sia sempre un valore positivo, a causa dell'elevamento a quadrato della sottrazione.
Si considera come loss, cioè, il valore assoluto dell'errore.

```py
# se x sono gli input e y sono gli output previsti della rete:
ypred = [n(x) for x in xs]

# sottraggo valore ottenuto dalla rete da quello previsto ed elevo al quadrato
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```

