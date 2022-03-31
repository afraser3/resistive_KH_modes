import numpy as np
from matplotlib import pyplot as plt
import kolmogorov_EVP

plt.rcParams.update({"text.usetex": True})

phases = np.linspace(0, 2.0*np.pi, num=50, endpoint=False)
Pm = 0.9
HB = 1.0  ## 1.0  # 0.55  # 1.0
# HB = 124.7598
Re = 100.0  ## 100.0  # 1000.0  # 100.0
# Re = 7.75342
#HB = 31.4  # 1.0  # 0.55  # 1.0
#Re = 17.8  # 100.0  # 1000.0  # 100.0
Rm = Pm * Re
delta = 0.0
N = 33
xs = np.linspace(0, 2.0*np.pi, num=N, endpoint=False)
fname = 'plots/mode_energies_Pm{}_HB{}_Re{}_N{}.pdf'.format(Pm, HB, Re, N)
ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction
# ns_ishift = np.fft.ifftshift(ns)  # same thing, but with 'standard' FFTW order, i.e., [0, 1, 2, ..., -2, -1]

ks = np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
               np.linspace(0.05, 0.2, num=50, endpoint=False))
# ks = np.linspace(0.1, 0.3)

gammas = kolmogorov_EVP.gamma_over_k(delta, HB, Re, Rm, ks, N)
maxind = np.argmax(gammas)
kmax = ks[maxind]
gammax = gammas[maxind]
lmat = kolmogorov_EVP.Lmat(delta, HB, Re, Rm, kmax, N)
w, v = np.linalg.eig(lmat)
ind1 = np.argmax(-np.imag(w))  # index in w,v of fastest-growing mode
ind2 = np.argmin(np.abs(w + np.conj(w[ind1])))  # index of its conjugate pair
omega1 = w[ind1]
omega2 = w[ind2]
full_mode1 = v[:, ind1]
full_mode2 = v[:, ind2]
phi1 = full_mode1[::2]
psi1 = full_mode1[1::2]
norm1_phi = phi1[int(len(ns)/2)+1]
norm1_psi = psi1[int(len(ns)/2)+1]
phi1 = phi1/norm1_phi
psi1 = psi1/norm1_phi
TE1 = kolmogorov_EVP.energy_from_streamfunc(phi1, kmax, ns) + HB*kolmogorov_EVP.energy_from_streamfunc(psi1, kmax, ns)
phi1 = phi1/np.sqrt(TE1)
psi1 = psi1/np.sqrt(TE1)
KE1 = kolmogorov_EVP.energy_from_streamfunc(phi1, kmax, ns)
ME1 = HB*kolmogorov_EVP.energy_from_streamfunc(psi1, kmax, ns)
phi2 = full_mode2[::2]
psi2 = full_mode2[1::2]
norm2_phi = phi2[int(len(ns)/2)+1]
norm2_psi = psi2[int(len(ns)/2)+1]
phi2 = -phi2/norm2_phi
psi2 = -psi2/norm2_phi
TE2 = kolmogorov_EVP.energy_from_streamfunc(phi2, kmax, ns) + HB*kolmogorov_EVP.energy_from_streamfunc(psi2, kmax, ns)
phi2 = phi2/np.sqrt(TE2)
psi2 = psi2/np.sqrt(TE2)
KE2 = kolmogorov_EVP.energy_from_streamfunc(phi2, kmax, ns)
ME2 = HB*kolmogorov_EVP.energy_from_streamfunc(psi2, kmax, ns)

phi3s = np.zeros((len(phases), len(phi2)), dtype=np.complex128)
psi3s = np.zeros_like(phi3s)
KE3s = np.zeros_like(phases)
ME3s = np.zeros_like(KE3s)
for i, phase in enumerate(phases):
    phi3 = (phi1 + np.exp(1.0j*phase)*phi2)/np.sqrt(2.0)
    psi3 = (psi1 + np.exp(1.0j*phase)*psi2)/np.sqrt(2.0)
    # TE3 = kolmogorov_EVP.energy_from_streamfunc(phi3, kmax, ns) + HB*kolmogorov_EVP.energy_from_streamfunc(psi3, kmax, ns)
    phi3s[i] = phi3  # /np.sqrt(TE3)
    psi3s[i] = psi3  # /np.sqrt(TE3)
    KE3s[i] = kolmogorov_EVP.energy_from_streamfunc(phi3s[i], kmax, ns)
    ME3s[i] = HB*kolmogorov_EVP.energy_from_streamfunc(psi3s[i], kmax, ns)

scale = 1.0
plt.figure(figsize=(scale*15, scale*5))
plt.subplot(1, 3, 1)
plt.plot(phases, KE3s, label=r'$E_K[(\psi_1 + e^{i \theta} \psi_2)/\sqrt{2}]$')
plt.axhline(KE1, c='k', label=r'$E_K[\psi_1]$')
plt.legend()
plt.ylabel(r'Energy')
plt.xlabel(r'$\theta \equiv$ complex phase between modes $\psi_1, \psi_2$')
plt.title(r'Kinetic energy $E_K$')
plt.xlim((0, 2.0*np.pi))
# plt.axhline(KE2)

plt.subplot(1, 3, 2)
plt.plot(phases, ME3s, label=r'$E_M[(A_1 + e^{i \theta} A_2)/\sqrt{2}]$')
plt.axhline(ME1, c='k', label=r'$E_M[A_1]$')
plt.legend()
plt.xlabel(r'$\theta$')
plt.title(r'Magnetic energy $E_M$')
plt.xlim((0, 2.0*np.pi))

plt.subplot(1, 3, 3)
plt.plot(phases, KE3s + ME3s)
plt.axhline(KE1 + ME1, c='k')
plt.xlabel(r'$\theta$')
plt.title(r'Total energy $E_K + E_M$')
plt.xlim((0, 2.0*np.pi))

# plt.savefig(fname)
plt.show()
