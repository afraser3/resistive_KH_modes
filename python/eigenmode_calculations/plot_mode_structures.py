import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import kolmogorov_EVP

plt.rcParams.update({"text.usetex": True})

Pm = 0.1
HB = 1.0  # 0.55  # 1.0
Re = 100.0  # 1000.0  # 100.0
Rm = Pm * Re
delta = 0.0
N = 33
fname = 'plots/mode_structures_Pm{}_HB{}_Re{}.pdf'.format(Pm, HB, Re)
ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction
# ns_ishift = np.fft.ifftshift(ns)  # same thing, but with 'standard' FFTW order, i.e., [0, 1, 2, ..., -2, -1]

ks = np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
               np.linspace(0.05, 0.2, num=50, endpoint=False))
# ks = np.linspace(0.1, 0.3)

scales = 1

def xz_from_kxkz(phi_kx_ishift, ns_ishift, kz, scalefac=1):
    # given phi(kx,kz) returns phi(x,z) where z is the direction of flow, x is the direction of shear
    # ns is the array of kx values over which the Fourier series of phi is given
    # returns phi(x,z) where the x,z grid is len(ns)*scalefac points in each of x and z, with
    # 0 <= x < 2pi and 0 <= z < 2pi/kz
    #
    # NOT SET UP FOR delta>0 MODES
    #
    # ASSUMES phi_kxkz IS IN STANDARD FFT ORDER, i.e., STARTING WITH kx=0 PART
    if ns_ishift[0] != 0:
        raise ValueError('Need to provide arrays in standard FFT order')
    if int(scalefac*len(ns_ishift)) != scalefac*len(ns_ishift):
        raise ValueError('Need scalefac * len(ns_ishift) to be an integer')
    phi_kxkz_ishift = np.zeros((int(scalefac*len(ns_ishift)), int(scalefac*len(ns_ishift))), dtype=np.complex128)
    phi_kxkz_ishift[:, 1] = phi_kx_ishift
    phi_kx = np.fft.fftshift(phi_kx_ishift)
    # need to do some shifting around here in order to ensure
    # that phi_kxkz(-kx, -kz) = conj[phi_kxkz(kx, kz)]
    phi_kx_flip = phi_kx[::-1]
    phi_kx_ishift_flip = np.fft.ifftshift(phi_kx_flip)
    phi_kxkz_ishift[:, -1] = np.conj(phi_kx_ishift_flip)
    xs = np.linspace(0, 2.0*np.pi, num=int(scalefac*len(ns_ishift)), endpoint=False)
    zs = np.linspace(0, 2.0*np.pi/kz, num=int(scalefac*len(ns_ishift)), endpoint=False)
    phi_xz = np.fft.ifft2(phi_kxkz_ishift)
    return phi_xz, xs, zs


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
# out = kolmogorov_EVP.gamfromparams(delta, HB, Re, Rm, kmax, N, False, True)
# full_mode = out[1]
# print(gammax, out[0])
# print(np.shape(out[1]))
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
phi3 = phi2 * norm1_psi * norm2_phi / (norm2_psi * norm1_phi)
psi3 = psi2 * norm1_psi * norm2_phi / (norm2_psi * norm1_phi)
# phi3 = 1.0j*phi2
# psi3 = 1.0j*psi2
TE3 = kolmogorov_EVP.energy_from_streamfunc(phi3, kmax, ns) + HB*kolmogorov_EVP.energy_from_streamfunc(psi3, kmax, ns)
phi3 = phi3/np.sqrt(TE3)
psi3 = psi3/np.sqrt(TE3)
phi_ishift1 = np.fft.ifftshift(phi1)
psi_ishift1 = np.fft.ifftshift(psi1)
phi_ishift2 = np.fft.ifftshift(phi2)
psi_ishift2 = np.fft.ifftshift(psi2)
phi_ishift3 = np.fft.ifftshift(phi3)
psi_ishift3 = np.fft.ifftshift(psi3)

phi_xz1, xs, zs = xz_from_kxkz(phi_ishift1, np.fft.ifftshift(ns), kmax, scalefac=scales)
xx, zz = np.meshgrid(xs, zs)
psi_xz1 = xz_from_kxkz(psi_ishift1, np.fft.ifftshift(ns), kmax, scalefac=scales)[0]
if (not np.allclose(phi_xz1, np.real(phi_xz1))) and (not np.allclose(psi_xz1, np.real(psi_xz1))):
    raise RuntimeError("phi and/or psi isn't real, something is wrong")
else:
    phi_xz1 = np.real(phi_xz1)
    psi_xz1 = np.real(psi_xz1)
phi_xz2 = xz_from_kxkz(phi_ishift2, np.fft.ifftshift(ns), kmax, scalefac=scales)[0]
psi_xz2 = xz_from_kxkz(psi_ishift2, np.fft.ifftshift(ns), kmax, scalefac=scales)[0]
if (not np.allclose(phi_xz2, np.real(phi_xz2))) and (not np.allclose(psi_xz2, np.real(psi_xz2))):
    raise RuntimeError("phi and/or psi isn't real, something is wrong")
else:
    phi_xz2 = np.real(phi_xz2)
    psi_xz2 = np.real(psi_xz2)
phi_xz3 = xz_from_kxkz(phi_ishift3, np.fft.ifftshift(ns), kmax, scalefac=scales)[0]
psi_xz3 = xz_from_kxkz(psi_ishift3, np.fft.ifftshift(ns), kmax, scalefac=scales)[0]
phi_xz3 = np.real(phi_xz3)
psi_xz3 = np.real(psi_xz3)

mode_amplitudes = np.array([1, 500, 250, 50, 20, 10, 5, 2, 1])
eq_amplitudes = np.ones_like(mode_amplitudes)
eq_amplitudes[0] = 0
###
# phi_xz2 = np.zeros_like(phi_xz2)
# psi_xz2 = np.zeros_like(psi_xz2)
# phi_xz3 = np.zeros_like(phi_xz3)
# psi_xz3 = np.zeros_like(psi_xz3)
###
scale = 0.75
with PdfPages(fname) as pdf:
    for i in range(len(mode_amplitudes)):
        mode_amp = mode_amplitudes[i]
        eq_amp = eq_amplitudes[i]
        plt.figure(figsize=(scale*15, scale*25))
        plt.subplot(2, 5, 1)
        plt.contourf(xs, zs, mode_amp*phi_xz1.T - eq_amp*np.cos(xx))
        plt.ylabel(r'$z$')
        if i == 0:
            plt.title(r'$\psi_1$', fontsize='medium')
        else:
            plt.title(r'$-\cos(x) + {}\psi_1$'.format(mode_amp), fontsize='medium')
        plt.subplot(2, 5, 6)
        plt.contourf(xs, zs, mode_amp*psi_xz1.T + eq_amp*xx)
        plt.ylabel(r'$z$')
        plt.xlabel(r'$x$')
        if i == 0:
            plt.title(r'$A_1$', fontsize='medium')
        else:
            plt.title(r'$x + {}A_1$'.format(mode_amp), fontsize='medium')

        plt.subplot(2, 5, 2)
        plt.contourf(xs, zs, mode_amp*phi_xz2.T - eq_amp*np.cos(xx))
        if i == 0:
            plt.title(r'$\psi_2$', fontsize='medium')
        else:
            plt.title(r'$-\cos(x) + {}\psi_2$'.format(mode_amp), fontsize='medium')
        plt.subplot(2, 5, 7)
        plt.contourf(xs, zs, mode_amp*psi_xz2.T + eq_amp*xx)
        plt.xlabel(r'$x$')
        if i == 0:
            plt.title(r'$A_2$', fontsize='medium')
        else:
            plt.title(r'$x + {}A_2$'.format(mode_amp), fontsize='medium')

        plt.subplot(2, 5, 3)
        plt.contourf(xs, zs, mode_amp*(phi_xz1.T + phi_xz2.T) - eq_amp*np.cos(xx))
        if i == 0:
            plt.title(r'$\psi_1 + \psi_2$', fontsize='medium')
        else:
            plt.title(r'$-\cos(x) + {}(\psi_1 + \psi_2)$'.format(mode_amp), fontsize='medium')
        plt.subplot(2, 5, 8)
        plt.contourf(xs, zs, mode_amp*(psi_xz1.T + psi_xz2.T) + eq_amp*xx)
        plt.xlabel(r'$x$')
        if i == 0:
            plt.title(r'$A_1 + A_2$', fontsize='medium')
        else:
            plt.title(r'$x + {}(A_1 + A_2)$'.format(mode_amp), fontsize='medium')

        plt.subplot(2, 5, 4)
        plt.contourf(xs, zs, mode_amp*(phi_xz1.T + phi_xz3.T) - eq_amp*np.cos(xx))
        if i == 0:
            plt.title(r'$\psi_1 + e^{{i \alpha}}\psi_2$', fontsize='medium')
        else:
            plt.title(r'$-\cos(x) + {}(\psi_1 + e^{{i \alpha}}\psi_2)$'.format(mode_amp), fontsize='medium')
        plt.subplot(2, 5, 9)
        plt.contourf(xs, zs, mode_amp*(psi_xz1.T + psi_xz3.T) + eq_amp*xx)
        plt.xlabel(r'$x$')
        if i == 0:
            plt.title(r'$A_1 + e^{i \alpha}A_2$', fontsize='medium')
        else:
            plt.title(r'$x + {}(A_1 + e^{{i \alpha}}A_2)$'.format(mode_amp), fontsize='medium')

        plt.subplot(2, 5, 5)
        plt.contourf(xs, zs, mode_amp*(phi_xz1.T - phi_xz2.T) - eq_amp*np.cos(xx))
        if i == 0:
            plt.title(r'$\psi_1 - \psi_2$', fontsize='medium')
        else:
            plt.title(r'$-\cos(x) + {}(\psi_1 - \psi_2)$'.format(mode_amp), fontsize='medium')
        plt.subplot(2, 5, 10)
        plt.contourf(xs, zs, mode_amp*(psi_xz1.T - psi_xz2.T) + eq_amp*xx)
        plt.xlabel(r'$x$')
        if i == 0:
            plt.title(r'$A_1 - A_2$', fontsize='medium')
        else:
            plt.title(r'$x + {}(A_1 - A_2)$'.format(mode_amp), fontsize='medium')

        pdf.savefig()
        plt.close()
print('done')

