import numpy as np
from matplotlib import pyplot as plt
import kolmogorov_EVP

ks = np.append(np.append(np.linspace(0.0001, 0.05, num=100, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=20))

# HB = 0.51
HB = 0.45
HB = 0.3  # 0.4  # 0.49770235643321115
HB = 0.49770235643321115
HB = 33.2744102
HB = 124.7598
# HB = 0.50001
# Re = 322.0
Re = 1000.0
Re = 8.6679185
Re = 7.75342
Pm = 0.1
# Pm = 2.0
Rm = Pm*Re
delta = 0.0
N = 33

omegas = np.zeros((len(ks), 2*N), dtype=np.complex128)
num_unstables = np.zeros((len(ks)), dtype=int)
contains_resistive_mode = np.zeros((len(ks)), dtype=bool)
contains_ordinary_mode = np.zeros((len(ks)), dtype=bool)
contains_strange_mode = np.zeros((len(ks)), dtype=bool)
for i, k in enumerate(ks):
    L = kolmogorov_EVP.Lmat(delta, HB, Re, Rm, k, N, ideal=False)
    w, v = np.linalg.eig(L)
    inds = np.argsort(np.imag(w))
    omegas[i] = w[inds]
    for gamma in -np.imag(omegas[i]):
        if gamma > 1e-12:
            num_unstables[i] += 1
    if num_unstables[i] > 0:
        for evi in range(num_unstables[i]):
            w_i = w[inds[evi]]
            v_i = v[:, inds[evi]]
            if np.abs(np.real(w_i)) > 1e-12:
                contains_resistive_mode[i] = True
            else:
                phi_i = np.abs(v_i[::2])
                psi_i = np.abs(v_i[1::2])
                if phi_i[int(N/2)] < 1e-10:
                    contains_strange_mode[i] = True
                else:
                    contains_ordinary_mode[i] = True

resistive_omegas = np.ma.masked_where(np.logical_not(np.logical_and(-np.imag(omegas) > 1e-10, np.abs(np.real(omegas)) > 1e-10)), omegas)
other_omegas = np.ma.masked_where(np.logical_not(np.logical_and(-np.imag(omegas) > 1e-10, np.abs(np.real(omegas)) < 1e-10)), omegas)

contains_strange_mode = contains_strange_mode.astype(int)
contains_resistive_mode = contains_resistive_mode.astype(int)
contains_ordinary_mode = contains_ordinary_mode.astype(int)

plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)
for i in range(2*N):
    if np.any(np.logical_not(other_omegas.mask[:, i])):
        plt.plot(ks, -np.imag(other_omegas[:, i]), c='C{}'.format(i))
    if np.any(np.logical_not(resistive_omegas.mask[:, i])):
        plt.plot(ks, -np.imag(resistive_omegas[:, i]), c='C2')
    # if np.any(-np.imag(omegas[:, i]) > 1e-10):
        # plt.plot(ks, -np.imag(omegas[:, i]))
# plt.fill_between(ks, contains_strange_mode, hatch='/', alpha=0)
# plt.fill_between(ks, contains_ordinary_mode, hatch='\\', alpha=0)
plt.ylim(ymin=0)
# plt.xlim((0, ks[-1]))
# plt.xlim((0, 0.4))
plt.xlim((0, 0.05))
plt.ylabel(r'Growth rate')
# plt.axvline(1.0/50)
# plt.axvline(2.0/50)
plt.axvline(2.0/150)
plt.axvline(4.0/150)
plt.xlabel(r'$k_z$')
plt.title(r'$C_B = 1/M_A^2 = 0.4977, \mathrm{Re} = 1000, \mathrm{Rm} = 100$')
plt.subplot(1, 2, 2)
for i in range(2*N):
    if np.any(np.logical_not(other_omegas.mask[:, i])):
        plt.plot(ks, np.real(other_omegas[:, i]), '.', c='C{}'.format(i))
    if np.any(np.logical_not(resistive_omegas.mask[:, i])):
        plt.plot(ks, np.real(resistive_omegas[:, i]), '.', c='C2')
# plt.ylim(ymin=0)
plt.xlim((0, 0.4))
plt.ylabel(r'Frequency')
plt.xlabel(r'$k_z$')
# plt.plot(ks, num_unstables, '.')
# plt.xlim((0, ks[-1]))
plt.subplots_adjust(wspace=0.25)
plt.show()
# plt.savefig('threemode_disp_relation.pdf', bbox_inches='tight')
#omegas = kolmogorov_EVP.omega_over_k(delta, HB, Re, Rm, ks, N, ideal=False)

#plt.plot(ks, -np.imag(omegas))
