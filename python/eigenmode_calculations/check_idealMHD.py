import numpy as np
from matplotlib import pyplot as plt
import kolmogorov_EVP

HB = 0.49770235643321115
kz = 0.1985
Re = 1000.0
Pm = 0.1
Rm = Pm*Re

N1 = 257
N2 = 513
N3 = 1025

omega_visc_1 = np.linalg.eig(kolmogorov_EVP.Lmat(0.0, HB, Re, Rm, kz, N1))[0]
omega_visc_2 = np.linalg.eig(kolmogorov_EVP.Lmat(0.0, HB, Re, Rm, kz, N2))[0]
omega_visc_3 = np.linalg.eig(kolmogorov_EVP.Lmat(0.0, HB, Re, Rm, kz, N3))[0]

omega_invisc_1 = np.linalg.eig(kolmogorov_EVP.Lmat(0.0, HB, Re, Rm, kz, N1, ideal=True))[0]
omega_invisc_2 = np.linalg.eig(kolmogorov_EVP.Lmat(0.0, HB, Re, Rm, kz, N2, ideal=True))[0]
omega_invisc_3 = np.linalg.eig(kolmogorov_EVP.Lmat(0.0, HB, Re, Rm, kz, N3, ideal=True))[0]

plt.subplot(2, 1, 1)
plt.plot(-np.imag(omega_visc_1), np.real(omega_visc_1), '.')
plt.plot(-np.imag(omega_visc_2), np.real(omega_visc_2), 'x')
plt.plot(-np.imag(omega_visc_3), np.real(omega_visc_3), '+')
plt.xlim((-0.0075, 0.0075))

plt.subplot(2, 1, 2)
plt.plot(-np.imag(omega_invisc_1), np.real(omega_invisc_1), '.')
plt.plot(-np.imag(omega_invisc_2), np.real(omega_invisc_2), 'x')
plt.plot(-np.imag(omega_invisc_3), np.real(omega_invisc_3), '+')
plt.xlim((-0.0075, 0.0075))

plt.show()
