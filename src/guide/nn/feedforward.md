# Feed-forward MLP Layer

i **logits** calcolati attraverso i layer di self-attention non hanno il tempo di pensare su cosa è stato trovato nei token.  
Per "rallentare" il processo di training, per aiutare nella comprensione delle informazioni trovate durante l'attention, viene introdotto un layer **MLP***, un piccolo livello di **feed forward**.  

E' un semplice layer lineare seguito da una attivazione non lineare di tipo ReLU.  
```py
class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
```


Per convenienza può essere anche usato per innestare al suo interno un componente di **Dropout**.  
