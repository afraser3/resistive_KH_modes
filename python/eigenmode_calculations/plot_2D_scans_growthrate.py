import numpy as np
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.colors as colors

plt.rcParams.update({"text.usetex": True})

Pm = 0.1
N = 33
delta = 0.0
out_dir = Path('kolmogorov_2D_scans/Pm{}_N{}_delta{}'.format(Pm, N, delta))
set_num = 2
out_file = out_dir / 'set_{}.h5'.format(set_num)

plot_fname = 'plots/scan_Pm{}_N{}_delta{}.pdf'.format(Pm, N, delta)

with h5py.File(out_file, mode='r') as file:
    ks = np.array(file['scan_values/ks'])
    HBs = np.array(file['scan_values/HBs'])
    Res = np.array(file['scan_values/Res'])
    ns = np.array(file['scan_values/ns'])
    omegas = np.array(file['results/omega'])
    k_maxs = np.array(file['results/k_max'])
    num_unstables = np.array(file['results/num_unstable'])
    KExs = np.array(file['results/KEx'])
    KEzs = np.array(file['results/KEz'])
    MExs = np.array(file['results/MEx'])
    MEzs = np.array(file['results/MEz'])
    viscous_diss = np.array(file['results/viscous_diss'])
    resistive_diss = np.array(file['results/resistive_diss'])
num_unstables = np.ma.masked_equal(num_unstables, 0)

scale = 0.6
plt.figure(figsize=(scale*7, scale*5))
plt.contourf(Res, HBs, -np.imag(omegas.T), levels=np.geomspace(1e-7, 1e0, 15), norm=colors.LogNorm())
# plt.pcolormesh(Res, HBs, -np.imag(omegas.T), norm=colors.LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
# plt.contour(Res, HBs, num_unstables.T, [0.0, 1.0, 2.0], colors=['k', 'C0', 'C1'])
# plt.contourf(Res, HBs, num_unstables.T, [0.5, 1.5, 2.5], hatches=['//', None, None], alpha=0)
# plt.contour(Res, HBs, num_unstables.T, [0.5, 1.5], colors='k', linewidths=2.0)
plt.contourf(Res, HBs, np.abs(np.real(omegas)).T, [0.0, 1e-10], hatches=['//', None], alpha=0)
plt.contour(Res, HBs, np.abs(np.real(omegas)).T, [0.0, 1e-10], colors='k', linewidths=2.0)
plt.axhline(0.5)
plt.ylabel(r'$C_B$')
plt.xlabel(r'$\mathrm{Re}$')
plt.title(r'$\Re[\lambda]$ for $\mathrm{{Pm}} = {:3.1f}$'.format(Pm))

plt.show()
# plt.savefig(plot_fname, bbox_inches='tight')
