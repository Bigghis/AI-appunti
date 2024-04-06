# Rank manipulations

### Aggiungere dimensioni
Possiamo facilmente aggiungere **dimensioni di size 1**, ad un tensore, modificandone il rank:

```py
t = torch.ones(2,3,4,5)
# t.shape = [2,3,4,5]

# prepend a 1 size to the first dimension
t1 = t[None, :]
# t1.shape = [1,2,3,4,5]

# adds a 1 size in second dimension:
t1 = t[;, None, :]
# t1.shape = [2,1,3,4,5]

# adds a 1 size in third dimension:
t1 = t[;, :, None, :]
# t1.shape = [2,3,1,4,5]

# adds a 1 size in fourth dimension:
t1 = t[;, :, :, None, :]
# t1.shape = [2,3,4,1,5]

# adds a 1 size after last dimension:
t1 = t[;, None]
# t1.shape = [2,3,4,5,1]

```

é possibile usare anche **torch.unsqueeze( tensor_t, dim_position)**

### Rimuovere dimensioni

per rimuovere tutte le dimensioni di size 1 da un tensore possiamo usare **torch.squeeze()**

```py
t = torch.ones(3,4,1,1,1,5,1,4,1)
t.squeeze().shape
# torch.Size([3, 4, 5, 4])
```


Per rimuovere una dimensione specifica, modificando il rank di un tensore:

```py
t = torch.ones(5,4,3,2,1)

# removes first dimension:
t1 = t[0, :]
# t1.shape = [4,3,2,1]

# removes second dimension:
t1 = t[;, 0, :]
# t1.shape = [5,3,2,1]

# removes third dimension:
t1 = t[;, :, 0, :]
# t1.shape = [5,4,2,1]
```
