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
set_num = 5
out_file = out_dir / 'set_{}.h5'.format(set_num)

plot_fname = 'plots/scan_phasevelocity_Pm{}_N{}_delta{}.eps'.format(Pm, N, delta)

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
# freqs = np.ma.masked_less_equal(np.abs(np.real(omegas)), 1e-10)
freqs = np.ma.masked_where(np.logical_or(num_unstables < 2, np.abs(np.real(omegas)) < 1e-10), np.abs(np.real(omegas)))
kmaxs = np.ma.masked_where(num_unstables < 1, k_maxs)
scale = 0.6
plt.figure(figsize=(scale*13, scale*5))

plt.subplot(1, 2, 1)
# plt.contourf(Res, HBs, (-np.real(omegas)/k_maxs).T, levels=np.geomspace(1e-7, 1e0, 8), norm=colors.LogNorm())
plt.contourf(Res, HBs, (freqs/k_maxs).T)#, norm=colors.LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.axhline(0.5, c='r', lw=0.5)
# plt.axvline(Res[50], c='k', ls='-')
plt.ylabel(r'$C_B$')
plt.xlabel(r'$Re$')
plt.title(r'$|\mathrm{{Im}}[\lambda]|/k_z$ for $Pm = {:3.1f}$'.format(Pm))

plt.subplot(1, 2, 2)
plt.loglog(HBs, freqs[50]/k_maxs[50], label=r'$|\mathrm{{Im}}[\lambda]|/k_z$')
plt.loglog(HBs, np.sqrt(HBs), ls='--', c='k', label=r'$\sqrt{C_B}$')
plt.xlabel(r'$C_B$')
plt.legend()
plt.xlim((1e-1, 1e2))

plt.savefig(plot_fname, bbox_inches='tight')
# plt.show()
