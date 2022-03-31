import numpy as np
import kolmogorov_EVP
from matplotlib import pyplot as plt

CB1 = 0.1
Re1 = 100.0
Rm1 = 10.0

CB2 = 0.6
Re2 = 1000.0
Rm2 = 100.0

N = 33
ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))

ks = np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
               np.linspace(0.05, 0.6, num=100, endpoint=False))
gams1 = kolmogorov_EVP.gamma_over_k(0.0, CB1, Re1, Rm1, ks, N)
k_max_ind1 = np.argmax(gams1)
kmax1 = ks[k_max_ind1]
gams2 = kolmogorov_EVP.gamma_over_k(0.0, CB2, Re2, Rm2, ks, N)
k_max_ind2 = np.argmax(gams2)
kmax2 = ks[k_max_ind2]

L1 = kolmogorov_EVP.Lmat(0.0, CB1, Re1, Rm1, kmax1, N)
w1, v1 = np.linalg.eig(L1)
gammax_ind1 = np.argmax(-np.imag(w1))
lambda_max1 = 1.0j*w1[gammax_ind1]

L2 = kolmogorov_EVP.Lmat(0.0, CB2, Re2, Rm2, kmax2, N)
w2, v2 = np.linalg.eig(L2)
gammax_ind2 = np.argmax(-np.imag(w2))
lambda_max2 = 1.0j*w2[gammax_ind2]

phi1 = v1[::2, gammax_ind1]
psi1 = v1[1::2, gammax_ind1]
psi1 = psi1 / phi1[int(N/2)]
phi1 = phi1 / phi1[int(N/2)]

phi2 = v2[::2, gammax_ind2]
psi2 = v2[1::2, gammax_ind2]
psi2 = psi2 / phi2[int(N/2 + 1)]
phi2 = phi2 / phi2[int(N/2 + 1)]

plt.subplot(2, 2, 1)
plt.semilogy(ns, np.abs(np.real(phi1)), '.')
plt.semilogy(ns, np.abs(np.imag(phi1)), '.')

plt.subplot(2, 2, 3)
plt.semilogy(ns, np.abs(np.real(psi1)), '.')
plt.semilogy(ns, np.abs(np.imag(psi1)), '.')

plt.subplot(2, 2, 2)
plt.semilogy(ns, np.abs(np.real(phi2)), '.')
plt.semilogy(ns, np.abs(np.imag(phi2)), '.')

plt.subplot(2, 2, 4)
plt.semilogy(ns, np.abs(np.real(psi2)), '.')
plt.semilogy(ns, np.abs(np.imag(psi2)), '.')

plt.show()
