# calculate cross entropy for classification problem
import numpy as np
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2)
nb_email = 100                       # On part sur 100 proba différentes comme si on avait 100 mails
p = np.linspace(0, 1, 100, endpoint=False)[1:] # imaginons qu'ils aient tous une proba différente
                                                # proba => donc comprise entre 0 et 1

#https://stackoverflow.com/questions/62594562/how-to-exclude-starting-point-from-linspace-function-under-numpy-in-python
# si on considère que les probas sont celles de mails qui sont des spams
ytruth = 1
def ll(p): return -(ytruth*np.log(p))  # Tous ces mails sont des spams (c'est la vérité)
# donc la log loss n'a que ce terme
ax1.plot(p,ll(p),label="cout pour/selon les probabilités pour la classe des spams")
ax1.scatter(0.1,ll(0.1))
ax1.annotate('Point 1', xy=(0.1, ll(0.1)))
ax1.scatter(0.9,ll(0.9))
ax1.annotate('Point 2', xy=(0.9, ll(0.9)))
ax1.legend()

# si on considère que les probas sont celles de mails qui sont des hams
ytruth = 0
def lk(p) : return -(1-ytruth)*np.log(1-p) # Tous ces mails sont des hams  (c'est la vérité)
# donc la log loss n'a que ce terme
ax2.plot(p,lk(p),label="cout pour/selon les probabilités pour la classe des hams")
ax2.scatter(0.1,lk(0.1))
ax2.annotate('Point 3', xy=(0.1, lk(0.1)))
ax2.scatter(0.9,lk(0.9))
ax2.annotate('Point 4', xy=(0.9, lk(0.9)))
ax2.legend()
plt.show()