# SGD with momentum


il **momentum** è una tecnica per accelerare la discesa del gradiente nella giusta direzione.  
Viene memorizzato il gradiente dei loop passati **per ogni peso (parametro)**, per determinare la giusta direzione di discesa del gradiente.  
La memorizzazione comporta, quindi, un salvataggio dello **stato** dell'optimizer con conseguente ingombro di memoria.  

Si noti che la memorizzazione dello stato, per modelli che hanno molti parametri, può richiedere quantità di memoria non certo trascurabili!

**Il momentum aiuta la loss function a convergere verso i minimi.**  
Perciò ha l'effetto di smorzare le oscillazioni della loss function.

Per esempio si usa un **momentum = 0.5**

# RMSProp (Root Mean Square Propagation)

Metodo affine al momentum è **RMSProp** inventato da Geoffrey Hinton.  
Viene memorizzata la media mobile dei quadrati dei gradienti e viene diviso il learning rate per la radice quadrata di questa media memorizzata, **per ciascun peso** + un piccolo epsilon.  

Nella pratica questo garantisce che si possa adattare un learning rate per ogni peso del modello, smorzando i picchi delle oscillazioni della loss function ed accelerando la stessa in direzione della giusta posizione di discesa.  

Sembra molto efficace se usato per ottimizzare reti ricorrenti **RNN**.

In generale è più efficente e veloce del semplice SGD with momentum, soprattutto nel caso di gradienti sparsi e rumorosi.  


esempio di implementazione

```py
class RMSProp(SGD):
    def __init__(self, params, lr, wd=0., sqr_mom=0.99, eps=1e-5):
        super().__init__(params, lr=lr, wd=wd)
        self.sqr_mom,self.eps = sqr_mom,eps

    def opt_step(self, p):
        if not hasattr(p, 'sqr_avg'): p.sqr_avg = p.grad**2
        p.sqr_avg = p.sqr_avg*self.sqr_mom + p.grad**2*(1-self.sqr_mom)
        p -= self.lr * p.grad/(p.sqr_avg.sqrt() + self.eps)

```     