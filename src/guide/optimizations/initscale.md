# Calculating init scale (Gain to compensate non linearity activating function)

Le probabilità dei valori di input, all'inizio, dovrebbero seguire quanto più possibile una **distribuzione gaussiana uniforme**.  
Vorremo, quindi ottenere una **media = 0** e una **deviazione standard = 1**.

Immaginando di avere:
```py
import matplotlib
import matplotlib.pyplot as plt


x = torch.randn(1000, 10) # dati di input, 1000 possibili esempi, embeddati in 10 dimensioni
W = torch.randn(10, 200) # 200 neuroni, ognuno accetta 10 input
y = x @ W # attivazione, moltiplicazione per i pesi W

print(x.mean(), x.std())
# tensor(0.0162) tensor(1.0022)  i dati di input sono distribuiti secondo una distribuzione gaussiana normale!

print(y.mean(), y.std())
# tensor(-0.0043) tensor(3.1594) i dati dopo l'attivazione, purtroppo NON sono distribuiti secondo una distribuzione gaussiana normale!
```
Possiamo anche fare un grafico dei dati x, che è una gaussiana:

```py
plt.figure(figsize=(20 ,5))
plt.subplot(121)
plt.hist(x.view(-1).tolist(), 50, density=True)
```

![hist1](../images/gauss2.png)  

e dei dati di y, dopo l'attivazione, cioè dopo la moltiplicazione per i pesi:
```py
plt.subplot(122)
plt.hist(y.view(-1).tolist(), 50, density=True)
```
![hist1](../images/nogauss1.png)  

Che sembra di forma gaussiana ma NON lo è, perché è decentrata e ripida!

Come facciamo per riportare y a forma di gaussiana,  
cioè per riportare una deviazione standard = 1?  

Basta dividere i pesi per la radice quadrata del numero di elementi di input **fan in**, con:  
**fan_in = embedded_size * block_size**:

```py
W = torch.randn(10, 200) / 10**0.5

# otterremo:
print(y.mean(), y.std())
# tensor(0.0044) tensor(1.0029)
```
![hist1](../images/gauss3.png) 

y distribuita come una gaussiana normale ;)

#### Gain per compensare l'effetto della funzione di attivazione non lineare

Abbiamo visto che le funzioni di attivazioni non lineari, come **tanh()**, **relu()**, **sigmoid()**, tendono a schiacciare i loro output
verso valori limite, causando **saturazione**.  
Per compensare quest'effetto si usano dei valori **gain** predeterminati analiticamente.  
Per esempio, per tanh() si usa **5/3**, etc. etc.

Avremo quindi, in aggiunta:
```py
# considerando
embedded_size = 2
block_size = 5

# avremo
W = torch.randn(10, 200) * (5/3) / ((embedded_size * block_size)**0.5)

```

**gain / ((embedded_size * block_size)*\*0.5)** è anche detto **kaiming init** dei pesi di input, in caso di non linearità di tipo tanh(), nella funzione di attivazione.  

Pytorch mette a disposizione la funzione:
```py
# es.: per tanh:
torch.nn.init.calculate_gain('tanh') # 1.666667 = 5/3
```