import numpy as np
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.colors as colors

plt.rcParams.update({"text.usetex": True})

# HB_stars = np.array([0.2, 0.2, 1.0, 1.0])
# Res = np.array([10.0, 500.0, 10.0, 500.0])
# Pm = 0.1
HB_stars = [0.2, 0.2, 0.6]
Res = [10.0, 500.0, 500.0]
Pm = 2.0
N = 33
delta = 0.0
out_dirs = [Path('kolmogorov_ky_kz_scans/Pm{}_HB{}_Re{}_N{}_delta{}'.format(Pm, HB_stars[i], Res[i], N, delta)) for i in range(len(Res))]
# set_nums = [2, 1, 3, 1]
set_nums = [0, 0, 1]
out_files = [out_dirs[i] / 'set_{}.h5'.format(set_nums[i]) for i in range(len(Res))]

plot_fname = 'plots/ky_kz_scans_Pm{}.pdf'.format(Pm)

scale = 0.6
# plt.figure(figsize=(scale*14, scale*10))
plt.figure(figsize=(scale*21, scale*5))
for i in range(len(Res)):
    with h5py.File(out_files[i], mode='r') as file:
        ks = np.array(file['scan_values/ks'])
        ns = np.array(file['scan_values/ns'])
        omegas = np.array(file['results/omega'])
    gammas = -np.imag(omegas)
    gammas = np.ma.masked_less(gammas, 0.0)

    # plt.subplot(2, 2, i+1)
    plt.subplot(1, 3, i + 1)
    plt.contourf(ks, ks, gammas.T)  # , levels=np.geomspace(1e-7, 1e0, 15), norm=colors.LogNorm())

    if i == 0:
        # if i == 0 or i == 2:
        plt.ylabel(r'$k_z$')
    if i == 0 or i == 1:
        # if i == 0 or i == 2:
        plt.colorbar()
    else:
        plt.colorbar(label=r'$\mathrm{Re}[\lambda]$')
    if True:
        # if i == 2 or i == 3:
        plt.xlabel(r'$k_y$')
    plt.title(r'$(Pm, Re, C_B) = ({:3.1f}, {}, {:3.1f})$'.format(Pm, int(Res[i]), HB_stars[i]))

# plt.show()
plt.savefig(plot_fname, bbox_inches='tight')
