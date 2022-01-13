import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import kolmogorov_EVP

plt.rcParams.update({"text.usetex": True})

phases = np.linspace(0, 2.0*np.pi, num=8, endpoint=False)
Pm = 0.1
HB = 1.0  # 0.55  # 1.0
Re = 100.0  # 1000.0  # 100.0
Rm = Pm * Re
delta = 0.0
N = 33
xs = np.linspace(0, 2.0*np.pi, num=N, endpoint=False)
fname = 'plots/mode_stresses_Pm{}_HB{}_Re{}.pdf'.format(Pm, HB, Re)
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
phi2 = full_mode2[::2]
psi2 = full_mode2[1::2]
norm2_phi = phi2[int(len(ns)/2)+1]
norm2_psi = psi2[int(len(ns)/2)+1]
phi2 = -phi2/norm2_phi
psi2 = -psi2/norm2_phi
TE2 = kolmogorov_EVP.energy_from_streamfunc(phi2, kmax, ns) + HB*kolmogorov_EVP.energy_from_streamfunc(psi2, kmax, ns)
phi2 = phi2/np.sqrt(TE2)
psi2 = psi2/np.sqrt(TE2)

tau_u1 = np.real(kolmogorov_EVP.stress_from_mode(phi1, kmax))
tau_b1 = -HB*np.real(kolmogorov_EVP.stress_from_mode(psi1, kmax))
tau_u2 = np.real(kolmogorov_EVP.stress_from_mode(phi2, kmax))
tau_b2 = -HB*np.real(kolmogorov_EVP.stress_from_mode(psi2, kmax))

phi3s = np.zeros((len(phases), len(phi2)), dtype=np.complex128)
psi3s = np.zeros_like(phi3s)
tau_u3s = np.zeros((len(phases), len(tau_u1)), dtype=np.float64)
tau_b3s = np.zeros_like(tau_u3s)
for i, phase in enumerate(phases):
    phi3 = (phi1 + np.exp(1.0j*phase)*phi2)/np.sqrt(2.0)
    psi3 = (psi1 + np.exp(1.0j*phase)*psi2)/np.sqrt(2.0)
    # TE3 = kolmogorov_EVP.energy_from_streamfunc(phi3, kmax, ns) + HB*kolmogorov_EVP.energy_from_streamfunc(psi3, kmax, ns)
    phi3s[i] = phi3  # /np.sqrt(TE3)
    psi3s[i] = psi3  # /np.sqrt(TE3)
    tau_u3s[i] = np.real(kolmogorov_EVP.stress_from_mode(phi3s[i], kmax))
    tau_b3s[i] = -HB*np.real(kolmogorov_EVP.stress_from_mode(psi3s[i], kmax))

max1 = np.max(np.abs(np.array([tau_u1, tau_b1])))
max2 = np.max(np.abs(np.array([tau_u2, tau_b2])))
max3 = np.max(np.abs(np.array([tau_u3s, tau_b3s])))
stress_max = np.max([max1, max2, max3])

scale = 1.0
plt.figure(figsize=(scale*15, scale*5))
plt.subplot(1, 3, 1)
plt.plot(xs, np.cos(xs), c='k', ls='-', label=r'$dW/dx$')
plt.plot(xs, tau_u1/stress_max, c='C0', ls='-', label=r'$\langle u w \rangle$')
plt.plot(xs, tau_b1/stress_max, c='C0', ls='--', label=r'$-C_B \langle b_x b_z \rangle$')
plt.xlabel(r'$x$')
plt.ylabel(r'stress')
plt.title(r'mode 1')
plt.legend()
plt.xlim((0, 2.0*np.pi))
plt.ylim((-1, 1))

plt.subplot(1, 3, 2)
plt.plot(xs, np.cos(xs), c='k', ls='-')
plt.plot(xs, tau_u2/stress_max, c='C1', ls='-')
plt.plot(xs, tau_b2/stress_max, c='C1', ls='--')
plt.xlabel(r'$x$')
plt.title(r'mode 2')
plt.xlim((0, 2.0*np.pi))
plt.ylim((-1, 1))

plt.subplot(1, 3, 3)
plt.plot(xs, np.cos(xs), c='k', ls='-')
for i in range(len(phases)):
    plt.plot(xs, tau_u3s[i]/stress_max, c='C{}'.format(i+2), ls='-')
    plt.plot(xs, tau_b3s[i]/stress_max, c='C{}'.format(i+2), ls='--')
plt.xlabel(r'$x$')
plt.title(r'$((\mathrm{mode 1}) + e^{i \theta} (\mathrm{mode 2}))/\sqrt{2}$')
plt.xlim((0, 2.0*np.pi))
plt.ylim((-1, 1))

plt.savefig(fname)
plt.show()
