import numpy as np
import kolmogorov_EVP


def eq82(CB, Re, Rm, kmax, lambda_max):
    K2 = 1.0 + kmax**2.0
    kz2 = kmax**2.0
    return (kz2*CB + 0.5*kz2*(1 - kz2)/K2 + (lambda_max + kz2/Rm)*(lambda_max + kz2/Re))/(kmax * (2.0*lambda_max + kz2/Rm + K2/Re))


def eq83(CB, Re, Rm, kmax, lambda_max):
    K2 = 1.0 + kmax**2.0
    kz2 = kmax**2.0
    return (kmax * (2.0*lambda_max + kz2/Re + K2/Rm))/(kz2*(0.5 - CB) + (lambda_max + K2/Rm)*(lambda_max + kz2/Rm))


def eq85(CB, Re, Rm, kmax, lambda_max):
    K2 = 1.0 + kmax**2.0
    kz2 = kmax**2.0
    return ((lambda_max + kz2/Re)*(lambda_max + kz2/Rm) - kz2*(0.5 - CB))/(kmax*(2.0*lambda_max + kz2/Re + K2/Rm))


def eq88(CB, Re, Rm, kmax):
    K2 = 1.0 + kmax**2.0
    kz2 = kmax**2.0
    sqrtCB = np.sqrt(CB)
    return [0.5*kmax*(1 + 1.0j*K2*(Re - Rm)/(sqrtCB*kmax*(Re + Rm)**2))/(2.0j*sqrtCB*kmax + K2/Re), 0.5*kmax*(1 - 1.0j*K2*(Re - Rm)/(sqrtCB*kmax*(Re + Rm)**2))/(-2.0j*sqrtCB*kmax + K2/Re)]


def eq89(CB):
    sqrtCB = np.sqrt(CB)
    return [-1.0j/sqrtCB, 1.0j/sqrtCB]


def eq811(CB, Re, Rm, kmax):
    K2 = 1.0 + kmax**2.0
    kz2 = kmax**2.0
    sqrtCB = np.sqrt(CB)
    return [-0.5*kmax*(1 - 1.0j*K2*(Re - Rm)/(sqrtCB*kmax*(Re + Rm)**2))/(2.0j*sqrtCB*kmax + K2/Rm), -0.5*kmax*(1 + 1.0j*K2*(Re - Rm)/(sqrtCB*kmax*(Re + Rm)**2))/(-2.0j*sqrtCB*kmax + K2/Rm)]



CB = 1000.0
Re = 100.0
Rm = 75.0  # 10.0
N = 3

ks = np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
               np.linspace(0.05, 0.2, num=50, endpoint=False))
gams = kolmogorov_EVP.gamma_over_k(0.0, CB, Re, Rm, ks, N)
k_max_ind = np.argmax(gams)
kmax = ks[k_max_ind]

L = kolmogorov_EVP.Lmat(0.0, CB, Re, Rm, kmax, N)
w, v = np.linalg.eig(L)
gammax_ind = np.argmax(-np.imag(w))
lambda_max = 1.0j*w[gammax_ind]
print(gams[k_max_ind], lambda_max)

phi = v[::2, gammax_ind]
psi = v[1::2, gammax_ind]
psi = psi / phi[int(N/2)]
phi = phi / phi[int(N/2)]

print('Comparing to the exact expressions, at the beginning of Appendix A: ')
print('psi_1/psi_0: ')
print(phi[int(N/2)+1]/phi[int(N/2)])
print(eq82(CB, Re, Rm, kmax, lambda_max))

print('A_0/psi_0: ')
print(psi[int(N/2)]*-1.0j/phi[int(N/2)])
print(eq83(CB, Re, Rm, kmax, lambda_max))

print('A_1/A_0: ')
print(psi[int(N/2) + 1]/psi[int(N/2)])
print(eq85(CB, Re, Rm, kmax, lambda_max))

print('Comparing the approximate expressions, which follow in the appendix: ')
print('psi_1/psi_0: ')
print(phi[int(N/2)+1]/phi[int(N/2)])
print(eq88(CB, Re, Rm, kmax))

print('A_0/psi_0: ')
print(psi[int(N/2)]*-1.0j/phi[int(N/2)])
print(eq89(CB))

print('A_1/A_0: ')
print(psi[int(N/2) + 1]/psi[int(N/2)])
print(eq811(CB, Re, Rm, kmax))
