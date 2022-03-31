import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import kolmogorov_EVP

plt.rcParams.update({"text.usetex": True})

Pm = 0.1
CB = 1.0
Re = 100.0
Rm = Pm * Re

phases = np.linspace(0, 2.0*np.pi, num=8, endpoint=False)

delta = 0.0
N = 33
plot_fname = 'plots/resistive_mode_structures_combinations.pdf'
ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction
# ns_ishift = np.fft.ifftshift(ns)  # same thing, but with 'standard' FFTW order, i.e., [0, 1, 2, ..., -2, -1]

ks = np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
               np.linspace(0.05, 0.2, num=50, endpoint=False))

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


gammas = kolmogorov_EVP.gamma_over_k(delta, CB, Re, Rm, ks, N)
maxind = np.argmax(gammas)
kmax = ks[maxind]
gammax = gammas[maxind]
lmat = kolmogorov_EVP.Lmat(delta, CB, Re, Rm, kmax, N)
w, v = np.linalg.eig(lmat)
ind1 = np.argmax(-np.imag(w))  # index in w,v of fastest-growing mode
ind2 = np.argmin(np.abs(w + np.conj(w[ind1])))  # index of its symmetric pair

omega1 = w[ind1]
full_mode1 = v[:, ind1]
phi1 = full_mode1[::2]
psi1 = full_mode1[1::2]
norm1_phi = phi1[int(len(ns)/2)+1]
# norm1_psi = psi1[int(len(ns)/2)+1]
phi1 = phi1/norm1_phi
psi1 = psi1/norm1_phi
TE1 = kolmogorov_EVP.energy_from_streamfunc(phi1, kmax, ns) + CB*kolmogorov_EVP.energy_from_streamfunc(psi1, kmax, ns)
phi1 = phi1/np.sqrt(TE1)
psi1 = psi1/np.sqrt(TE1)
phi1 = phi1*2000.0
psi1 = psi1*2000.0

omega2 = w[ind2]
full_mode2 = v[:, ind2]
phi2 = full_mode2[::2]
psi2 = full_mode2[1::2]
norm2_phi = phi2[int(len(ns)/2)+1]
# norm2_psi = psi2[int(len(ns)/2)+1]
phi2 = -phi2/norm2_phi
psi2 = -psi2/norm2_phi
TE2 = kolmogorov_EVP.energy_from_streamfunc(phi2, kmax, ns) + CB*kolmogorov_EVP.energy_from_streamfunc(psi2, kmax, ns)
phi2 = phi2/np.sqrt(TE2)
psi2 = psi2/np.sqrt(TE2)
phi2 = phi2*2000.0
psi2 = psi2*2000.0

phi3s = np.zeros((len(phases), len(phi2)), dtype=np.complex128)
psi3s = np.zeros_like(phi3s)
for i, phase in enumerate(phases):
    phi3s[i] = (phi1 + np.exp(1.0j * phase) * phi2) / np.sqrt(2.0)
    psi3s[i] = (psi1 + np.exp(1.0j * phase) * psi2) / np.sqrt(2.0)
phi_ishift3 = np.fft.ifftshift(phi3s, axes=-1)
psi_ishift3 = np.fft.ifftshift(psi3s, axes=-1)

scale = 0.5
#fig = plt.figure(figsize=(scale*25, scale*20))
#gs = fig.add_gridspec(nrows=2, ncols=4)
#gs1 = GridSpec(1, 2, left=0.05, right=0.48, wspace=0.125)
#gs2 = GridSpec(1, 2, left=0.55, right=0.98, wspace=0.125)
fig, axs = plt.subplots(2, len(phases), figsize=(scale*25, scale*20))
cbars1 = []
cbars2 = []
colorbars = False
# titles = [r'$\psi_1 + \psi_2$', r'$\psi_1 + i\psi_2$', r'$\psi_1 - \psi_2$', r'$\psi_1 - i\psi_2$']
titles = [r'$\psi_1 + \psi_2$', r'$\psi_1 + e^{i \pi/4}\psi_2$', r'$\psi_1 + e^{i \pi/2}\psi_2$',
          r'$\psi_1 + e^{3 i \pi/4}\psi_2$', r'$\psi_1 + e^{i \pi}\psi_2$', r'$\psi_1 + e^{5i \pi/4}\psi_2$',
          r'$\psi_1 + e^{3i \pi/2}\psi_2$', r'$\psi_1 + e^{7i \pi/4}\psi_2$']
titles2 = [r'$A_1 + A_2$', r'$A_1 + e^{i \pi/4}A_2$', r'$A_1 + e^{i \pi/2}A_2$',
           r'$A_1 + e^{3 i \pi/4}A_2$', r'$A_1 + e^{i \pi}A_2$', r'$A_1 + e^{5i \pi/4}A_2$',
           r'$A_1 + e^{3i \pi/2}A_2$', r'$A_1 + e^{7i \pi/4}A_2$']
for i in range(len(phases)):
    phi_xz3, xs, zs = xz_from_kxkz(phi_ishift3[i], np.fft.ifftshift(ns), kmax, scalefac=scales)
    # xx, zz = np.meshgrid(xs, zs)
    psi_xz3 = xz_from_kxkz(psi_ishift3[i], np.fft.ifftshift(ns), kmax, scalefac=scales)[0]
    if (not np.allclose(phi_xz3, np.real(phi_xz3))) and (not np.allclose(psi_xz3, np.real(psi_xz3))):
        raise RuntimeError("phi and/or psi isn't real, something is wrong")
    else:
        phi_xz3 = np.real(phi_xz3)
        psi_xz3 = np.real(psi_xz3)
    im = axs[0, i].contourf(xs, zs, phi_xz3.T)
    if colorbars:
        if i == len(phases)-1:
            cbars1 += [fig.colorbar(im, ax=axs[0, i], label=r'$\psi$')]
        else:
            cbars1 += [fig.colorbar(im, ax=axs[0, i])]
    axs[0, i].set_title(titles[i])

    im = axs[1, i].contourf(xs, zs, psi_xz3.T)
    if colorbars:
        if i == len(phases)-1:
            cbars2 += [fig.colorbar(im, ax=axs[1, i], label=r'$A$')]
        else:
            cbars2 += [fig.colorbar(im, ax=axs[1, i])]
    axs[1, i].set_title(titles2[i])

    axs[1, i].set_xlabel(r'$x$', fontsize='large')
    if i == 0:
        axs[0, i].set_ylabel(r'$z$', fontsize='large')
        axs[1, i].set_ylabel(r'$z$', fontsize='large')
    else:
        axs[0, i].tick_params(labelleft=False)
        axs[1, i].tick_params(labelleft=False)

plt.savefig(plot_fname, bbox_inches='tight')
plt.show()
