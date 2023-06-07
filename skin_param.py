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

def kubelka_munk_param(absorption, scattering):
    a = np.divide((absorption + scattering),scattering)
    b = np.sqrt(a**2 - 1)
    return a, b

def load_hb_absorptio_coeff():
    abs_data = (np.log(10)*150/64500)*np.loadtxt('hb_hb02_absorption.txt', usecols=range(0,3))
    hb02_abscoeff = abs_data[:,1]
    hb_abscoeff = abs_data[:,2]
    return hb02_abscoeff, hb_abscoeff

mua = absorption_t()
c = concentration_t()
mus = scattering_t()

# This example was extract from https://omlc.org/news/jan98/skinoptics.html
waveleght = np.linspace(400,1000,301)
c.melanin = 0.1
#mua.baseline = 7.84e8*(waveleght**(-3.255))
mua.baseline = 0.244 + 85.3*np.exp((154-waveleght)/66.2)
mua.melanin = 6.6e11*waveleght**(-3.33)
mua.epidermis = c.melanin*mua.melanin + (1 - c.melanin)*mua.baseline
mua.oxyhemoglobin, mua.deoxyhemoglobin = load_hb_absorptio_coeff()

mus.mie = 2e5*(waveleght**(-1.5))
mus.rayleigh = 2e12*(waveleght**(-4))
mus.dermis = mus.epidermis = mus.mie + mus.rayleigh

plt.figure(figsize=(8,4.5))
plt.semilogy(waveleght,mua.melanin,'k')
plt.semilogy(waveleght,mua.oxyhemoglobin,'r')
plt.semilogy(waveleght,mua.deoxyhemoglobin,'b')
plt.xlim(400,1000)
plt.show()

a_epi, b_epi = kubelka_munk_param(mua.epidermis, mus.dermis)