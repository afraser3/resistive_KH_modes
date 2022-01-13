import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
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

mode_amplitude = 100.0

delta = 0.0
N = 33
plot_fname = 'plots/KH_mode_structures.pdf'
ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction
# ns_ishift = np.fft.ifftshift(ns)  # same thing, but with 'standard' FFTW order, i.e., [0, 1, 2, ..., -2, -1]

ks = np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
               np.linspace(0.05, 0.2, num=50, endpoint=False))
# ks = np.linspace(0.1, 0.3)
ks = np.append(np.linspace(0.01, 0.275, num=100, endpoint=False),
               np.linspace(0.275, 0.6, num=50))

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


gammas1 = kolmogorov_EVP.gamma_over_k(delta, HB1, Re1, Rm1, ks, N)
maxind1 = np.argmax(gammas1)
kmax1 = ks[maxind1]
gammax1 = gammas1[maxind1]
lmat1 = kolmogorov_EVP.Lmat(delta, HB1, Re1, Rm1, kmax1, N)
w, v = np.linalg.eig(lmat1)
ind1 = np.argmax(-np.imag(w))
omega1 = w[ind1]
full_mode1 = v[:, ind1]
phi1 = full_mode1[::2]
psi1 = full_mode1[1::2]
norm1_phi = phi1[int(len(ns)/2)+1]
# norm1_psi = psi1[int(len(ns)/2)+1]
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
ind2 = np.argmax(-np.imag(w))
omega2 = w[ind2]
full_mode2 = v[:, ind2]
phi2 = full_mode2[::2]
psi2 = full_mode2[1::2]
norm2_phi = phi2[int(len(ns)/2)+1]
# norm2_psi = psi2[int(len(ns)/2)+1]
phi2 = -phi2/norm2_phi
psi2 = -psi2/norm2_phi
TE2 = kolmogorov_EVP.energy_from_streamfunc(phi2, kmax2, ns) + HB2*kolmogorov_EVP.energy_from_streamfunc(psi2, kmax2, ns)
phi2 = phi2/np.sqrt(TE2)
psi2 = psi2/np.sqrt(TE2)

phi_ishift1 = np.fft.ifftshift(phi1)
psi_ishift1 = np.fft.ifftshift(psi1)
phi_ishift2 = np.fft.ifftshift(phi2)
psi_ishift2 = np.fft.ifftshift(psi2)

phi_xz1, xs1, zs1 = xz_from_kxkz(phi_ishift1, np.fft.ifftshift(ns), kmax1, scalefac=scales)
xx1, zz1 = np.meshgrid(xs1, zs1)
psi_xz1 = xz_from_kxkz(psi_ishift1, np.fft.ifftshift(ns), kmax1, scalefac=scales)[0]
if (not np.allclose(phi_xz1, np.real(phi_xz1))) and (not np.allclose(psi_xz1, np.real(psi_xz1))):
    raise RuntimeError("phi and/or psi isn't real, something is wrong")
else:
    phi_xz1 = np.real(phi_xz1)
    psi_xz1 = np.real(psi_xz1)
phi_xz2, xs2, zs2 = xz_from_kxkz(phi_ishift2, np.fft.ifftshift(ns), kmax2, scalefac=scales)
xx2, zz2 = np.meshgrid(xs2, zs2)
psi_xz2 = xz_from_kxkz(psi_ishift2, np.fft.ifftshift(ns), kmax2, scalefac=scales)[0]
if (not np.allclose(phi_xz2, np.real(phi_xz2))) and (not np.allclose(psi_xz2, np.real(psi_xz2))):
    raise RuntimeError("phi and/or psi isn't real, something is wrong")
else:
    phi_xz2 = np.real(phi_xz2)
    psi_xz2 = np.real(psi_xz2)

scale = 0.5
fig = plt.figure(figsize=(scale*25, scale*10))
gs1 = GridSpec(1, 2, left=0.05, right=0.48, wspace=0.125)
gs2 = GridSpec(1, 2, left=0.55, right=0.98, wspace=0.125)

ax1 = fig.add_subplot(gs1[0, 0])
im1 = ax1.contourf(xs1, zs1, phi_xz1.T)
cbar1 = fig.colorbar(im1, ax=ax1)
ax1.set_ylabel(r'$z$', fontsize='large')
ax1.set_xlabel(r'$x$', fontsize='large')
# plt.title(r'$\psi$, fastest-growing mode, $Re = {}$, $C_B = {}$'.format(int(Re1), HB1), fontsize='large')
# plt.title(r'Fastest-growing mode, $Re = {}$, $C_B = {}$'.format(int(Re1), HB1), fontsize='large')
ax1.set_title(r'$\psi$, ordinary KH')
ax2 = fig.add_subplot(gs1[0, 1])
im2 = ax2.contourf(xs1, zs1, psi_xz1.T)
cbar2 = fig.colorbar(im2, ax=ax2)
ax2.set_xlabel(r'$x$', fontsize='large')
# plt.title(r'$A$, fastest-growing mode, $Re = {}$, $C_B = {}$'.format(int(Re1), HB1), fontsize='large')
ax2.set_title(r'$A$, ordinary KH')


ax3 = fig.add_subplot(gs2[0, 0])
im3 = ax3.contourf(xs2, zs2, phi_xz2.T)
cbar3 = fig.colorbar(im3, ax=ax3)
ax3.set_title(r'$\psi$, strange KH')
ax3.set_xlabel(r'$x$', fontsize='large')
ax3.set_ylabel(r'$z$', fontsize='large')
ax4 = fig.add_subplot(gs2[0, 1])
im4 = ax4.contourf(xs2, zs2, psi_xz2.T)
ax4.set_title(r'$A$, strange KH')
# plt.ylabel(r'$z$')
ax4.set_xlabel(r'$x$', fontsize='large')
cbar4 = fig.colorbar(im4, ax=ax4)
#plt.colorbar(label=r'$A$')
# plt.title(r'$A$, fastest-growing mode, $Re = {}$, $C_B = {}$'.format(int(Re2), HB2), fontsize='large')
# plt.subplots_adjust(hspace=0.1)

for ax in [ax2, ax4]:
    ax.tick_params(labelleft=False)

# plt.show()
plt.savefig(plot_fname, bbox_inches='tight')
