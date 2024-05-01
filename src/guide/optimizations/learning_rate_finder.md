# Learning Rate Finder

E' possibile trovare il corretto valore del learning rate da usare in **automatico**.  

### Pytorch learning rate scheduler
Pytorch possiede un ***optim.lr_scheduler** che può essere usato assieme ai vari ottimizzatori per determinare il corretto valore del learning rate.  

Usando un training loop "pytorch like", possiamo sfruttare lo scheduler in questo modo:

```py
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(num_epochs):
    for input, target in dataset:
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()  # <--- find best lr at every epoch loop
```

lo scheduler, partendo dal valore iniziale iniziale (0.1 nell'esempio), aggiorna il learning rate, cercando un valore migliore e passandolo all'ottimizzatore alla successiva iterazione di epoch.  

### Custom learning rate finder

E' ovviamente possibile anche una implementazione custom al fine di trovare il miglior learning rate.  
Possiamo creare svariati algoritmi adatti a questo scopo.  
Consideriamone uno che usa un **moltiplicatore** (es aumenta del 30% ill valore del learning rate per ogni epoch).  
Per ogni epoch:  
* dobbiamo memorizzare sia il valore del loss che quello del learning rate,
* dobbiamo fermare il training loop quando si verifica una certa condizione (es: il loss supera di 3 volte il minimo loss ottenuto)


Possibile implementazione:
```py

learning_rates, losses = [], []
min_learning_rate = 9999999 # inizializza min_learning_rate ad un valore molto alto
max_mult = 1.3 #valore del moltiplicatore (30%)

        self.sched = ExponentialLR(learn.opt, self.gamma)
        self.lrs,self.losses = [],[]
        self.min = math.inf
for epoch in range(num_epochs):
    learning_rates.append(learning_rate) # memorizza il valore attuale del learning rate
    losses.append(loss) # memorizza il valore attuale del loss

    if loss < min_learning_rate: # aggiorna min_learning_rate se necessario
        min_learning_rate = loss
    
    if math.isnan(loss) or (loss > min_learning_rate * max_mult): # se loss è troppo alto..
        break # stop training!
```