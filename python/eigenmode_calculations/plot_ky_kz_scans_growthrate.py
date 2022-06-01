import numpy as np
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.colors as colors

plt.rcParams.update({"text.usetex": True})

HB_star = 0.2
Re = 500.0
Pm = 0.1
N = 33
delta = 0.0
out_dir = Path('kolmogorov_ky_kz_scans/Pm{}_HB{}_Re{}_N{}_delta{}'.format(Pm, HB_star, Re, N, delta))
set_num = 1
out_file = out_dir / 'set_{}.h5'.format(set_num)

plot_fname = 'plots/scan_Pm{}_HB{}_Re{}_N{}_delta{}.pdf'.format(Pm, HB_star, Re, N, delta)

with h5py.File(out_file, mode='r') as file:
    ks = np.array(file['scan_values/ks'])
    ns = np.array(file['scan_values/ns'])
    omegas = np.array(file['results/omega'])
    # contains_resistive_mode = np.array(file['results/contains_resistive_mode'])
    # contains_strange_mode = np.array(file['results/contains_strange_mode'])
    # contains_ordinary_mode = np.array(file['results/contains_ordinary_mode'])

gammas = -np.imag(omegas)
gammas = np.ma.masked_less(gammas, 0.0)

scale = 0.6
plt.figure(figsize=(scale*7, scale*5))
plt.contourf(ks, ks, gammas.T)  # , levels=np.geomspace(1e-7, 1e0, 15), norm=colors.LogNorm())
# plt.xscale('log')
# plt.yscale('log')
plt.colorbar()
# plt.contourf(ks, ks, contains_ordinary_mode.astype(int).T, [-1, 0, 1], hatches=[None, '\\', None], alpha=0)
# plt.contour(ks, ks, contains_ordinary_mode.astype(int).T, [0, 1], colors='k', linewidths=2.0)
# plt.contourf(ks, ks, contains_strange_mode.astype(int).T, [-1, 0, 1], hatches=[None, '/', None], alpha=0)
# plt.contour(ks, ks, contains_strange_mode.astype(int).T, [0, 1], colors='k', linewidths=2.0)
plt.ylabel(r'$k_z$')
plt.xlabel(r'$k_y$')
plt.title(r'$\Re[\lambda]$ for $\mathrm{{Pm}} = {:3.1f}$'.format(Pm))

plt.show()
# plt.savefig(plot_fname, bbox_inches='tight')
