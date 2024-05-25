# Hooks

Altre callback messe a disposizione da pytorch, sono i cosidetti **hooks**.  
Mentre le callback viste finora possono essere eseguite in corrispondenza di punti specifici del training loop (after_batch, before_fit etc..),
gli hooks possono essere applicati ad un **modulo** o ad un **tensore**.

### Module Hooks
Per un modulo possiamo registrare un hook in 3 punti:

* **forward prehook** eseguito prima del forward (**register_forward_pre_hook()**)
* **forward** eseguito dopo il forward (**register_forward_hook()**)
* **backward** eseguito dopo il backward (**register_backward_hook()**)

le funzioni registrabili devono accettare 3 parametri, il modulo e i suoi input ed output, e ritornare **module_output** modificato o None

```py
def func_hook(module, module_input, module_output) 

# usage ex.:
module.register_forward_hook(func_hook)
```


### Tensor Hooks
Per un tensore possiamo registrare un hook solo per:

* **forward** eseguito dopo il forward, dopo che sono stati calcolati i gradienti del tensore (**register_hook()**)

le funzioni registrabili sono del tipo:

```py
def func_hook(grad) 

# usage ex.:
x_tensor.register_hook(func_hook)
```


**nota** per rimuovere un hook:

```py
d = c.register_hook(func_hook)
d.remove()
```

### Casi d'uso


