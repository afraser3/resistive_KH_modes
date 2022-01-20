import numpy as np
from matplotlib import pyplot as plt
import kolmogorov_EVP

plt.rcParams.update({"text.usetex": True})

Pm1 = 0.1
HB1 = 0.1
Re1 = 100.0
Rm1 = Pm1 * Re1
Pm2 = 0.1
HB2 = 0.6
Re2 = 1000.0
Rm2 = Pm2 * Re2
delta = 0.0
N = 33
xs = np.linspace(0, 2.0*np.pi, num=N, endpoint=False)
fname = 'plots/KH_mode_stresses.pdf'
ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction
# ns_ishift = np.fft.ifftshift(ns)  # same thing, but with 'standard' FFTW order, i.e., [0, 1, 2, ..., -2, -1]

ks = np.append(np.linspace(0.01, 0.275, num=100, endpoint=False),
               np.linspace(0.275, 0.6, num=50))

gammas1 = kolmogorov_EVP.gamma_over_k(delta, HB1, Re1, Rm1, ks, N)
maxind1 = np.argmax(gammas1)
kmax1 = ks[maxind1]
gammax1 = gammas1[maxind1]
lmat1 = kolmogorov_EVP.Lmat(delta, HB1, Re1, Rm1, kmax1, N)
w, v = np.linalg.eig(lmat1)
ind1 = np.argmax(-np.imag(w))  # index in w,v of fastest-growing mode
omega1 = w[ind1]
full_mode1 = v[:, ind1]
phi1 = full_mode1[::2]
psi1 = full_mode1[1::2]
norm1_phi = phi1[int(len(ns)/2)+1]
norm1_psi = psi1[int(len(ns)/2)+1]
phi1 = phi1/norm1_phi
psi1 = psi1/norm1_phi
TE1 = kolmogorov_EVP.energy_from_streamfunc(phi1, kmax1, ns) + HB1*kolmogorov_EVP.energy_from_streamfunc(psi1, kmax1, ns)
phi1 = phi1/np.sqrt(TE1)
psi1 = psi1/np.sqrt(TE1)

gammas2 = kolmogorov_EVP.gamma_over_k(delta, HB2, Re2, Rm2, ks, N)
maxind2 = np.argmax(gammas2)
kmax2 = ks[maxind2]
gammax2 = gammas2[maxind2]
lmat2 = kolmogorov_EVP.Lmat(delta, HB2, Re2, Rm2, kmax2, N)
w, v = np.linalg.eig(lmat2)
ind2 = np.argmax(-np.imag(w))  # index in w,v of fastest-growing mode
omega2 = w[ind2]
full_mode2 = v[:, ind2]
phi2 = full_mode2[::2]
psi2 = full_mode2[1::2]
norm2_phi = phi2[int(len(ns)/2)+1]
norm2_psi = psi2[int(len(ns)/2)+1]
phi2 = -phi2/norm2_phi
psi2 = -psi2/norm2_phi
TE2 = kolmogorov_EVP.energy_from_streamfunc(phi2, kmax2, ns) + HB2*kolmogorov_EVP.energy_from_streamfunc(psi2, kmax2, ns)
phi2 = phi2/np.sqrt(TE2)
psi2 = psi2/np.sqrt(TE2)

tau_u1 = np.real(kolmogorov_EVP.stress_from_mode(phi1, kmax1))
tau_b1 = -HB1*np.real(kolmogorov_EVP.stress_from_mode(psi1, kmax1))
tau_u2 = np.real(kolmogorov_EVP.stress_from_mode(phi2, kmax2))
tau_b2 = -HB2*np.real(kolmogorov_EVP.stress_from_mode(psi2, kmax2))

max1 = np.max(np.abs(np.array([tau_u1, tau_b1, tau_u1 + tau_b1])))
max2 = np.max(np.abs(np.array([tau_u2, tau_b2, tau_u2 + tau_b2])))
stress_max = np.max([max1, max2])

scale = 1.0
plt.figure(figsize=(scale*10, scale*5))
plt.subplot(1, 2, 1)
plt.plot(xs, np.cos(xs), c='k', ls='-', label=r'$dW/dx$')
plt.plot(xs, tau_u1/stress_max, c='C0', ls='-', label=r'$\langle u w \rangle$')
plt.plot(xs, tau_b1/stress_max, c='C0', ls='--', label=r'$-C_B \langle b_x b_z \rangle$')
plt.plot(xs, (tau_u1 + tau_b1)/stress_max, c='C0', ls='-.', label=r'$\langle u w \rangle - C_B \langle b_x b_z \rangle$')
plt.xlabel(r'$x$')
plt.ylabel(r'stress')
plt.title(r'mode 1')
plt.legend()
plt.xlim((0, 2.0*np.pi))
plt.ylim((-1, 1))

plt.subplot(1, 2, 2)
plt.plot(xs, np.cos(xs), c='k', ls='-')
plt.plot(xs, tau_u2/stress_max, c='C1', ls='-')
plt.plot(xs, tau_b2/stress_max, c='C1', ls='--')
plt.plot(xs, (tau_u2 + tau_b2)/stress_max, c='C1', ls='-.')
plt.xlabel(r'$x$')
plt.title(r'mode 2')
plt.xlim((0, 2.0*np.pi))
plt.ylim((-1, 1))

# plt.savefig(fname)
plt.show()
