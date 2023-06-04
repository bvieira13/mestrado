import numpy as np
import matplotlib.pyplot as plt

class absorption_t:
    baseline:float
    melanin:float
    epidermis:float
    dermis:float
    bilirubin:float
    oxyhemoglobin:float
    deoxyhemoglobin:float

class scattering_t:
    mie:float
    rayleigh:float
    dermis:float
    epidermis:float

class concentration_t:
    melanin:float
    bilirubin:float
    oxyhemoglobin:float
    deoxyhemoglobin:float
    
mua = absorption_t()
c = concentration_t()
mus = scattering_t()

# This example was extract from https://omlc.org/news/jan98/skinoptics.html
waveleght = np.linspace(250,1000,375)
c.melanin = 0.1
#mua.baseline = 7.84e8*(waveleght**(-3.255))
mua.baseline = 0.244 + 85.3*np.exp((154-waveleght)/66.2)
mua.melanin = 6.6e11*waveleght**(-3.33)
mua.epidermis = c.melanin*mua.melanin + (1 - c.melanin)*mua.baseline

mus.mie = 2e5*(waveleght**(-1.5))
mus.rayleigh = 2e12*(waveleght**(-4))
mus.dermis = mus.epidermis = mus.mie + mus.rayleigh

plt.figure(figsize=(8,4.5))
plt.semilogy(waveleght,mus.mie)
plt.semilogy(waveleght,mus.rayleigh )
plt.semilogy(waveleght,mus.dermis)
plt.xlim(300,1000)
plt.show()