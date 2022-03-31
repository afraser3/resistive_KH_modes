import numpy as np
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.colors as colors

Pm = 0.1
N = 33
delta = 0.0
out_dir = Path('kolmogorov_2D_scans/Pm{}_N{}_delta{}'.format(Pm, N, delta))
set_num = 2
out_file = out_dir / 'set_{}.h5'.format(set_num)

plot_fname = 'plots/scan_Pm{}_N{}_delta{}.pdf'.format(Pm, N, delta)

get_contains_arrays = False

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
    if get_contains_arrays:
        contains_resistive_mode = np.array(file['results/contains_resistive_mode'])
        contains_strange_mode = np.array(file['results/contains_strange_mode'])
        contains_ordinary_mode = np.array(file['results/contains_ordinary_mode'])

plt.figure(figsize=(10.0, 30.0))
plt.subplot(6, 2, 1)
plt.pcolormesh(Res, HBs, -np.imag(omegas.T), shading='nearest', norm=colors.LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.ylabel(r'$H_B^*$')
plt.title(r'$\gamma$ for $Pm = {:e}$'.format(Pm))

plt.subplot(6, 2, 2)
plt.pcolormesh(Res, HBs, np.abs(np.real(omegas.T)), shading='nearest', norm=colors.LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$|\omega_r|$')

plt.subplot(6, 2, 3)
plt.pcolormesh(Res, HBs, k_maxs.T, shading='nearest', norm=colors.LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$k_{max}$')
plt.ylabel(r'$H_B^*$')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 4)
plt.pcolormesh(Res, HBs, num_unstables.T, shading='nearest')
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
if get_contains_arrays:
    plt.contourf(Res, HBs, contains_resistive_mode.astype(int).T, [0, 1], hatches=['\\', None], alpha=0)
plt.title(r'Num unstable modes')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 5)
plt.pcolormesh(Res, HBs, KExs.T, shading='nearest')
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$KE_x$')
plt.ylabel(r'$H_B^*$')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 6)
plt.pcolormesh(Res, HBs, MExs.T, shading='nearest')
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$ME_x$')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 7)
plt.pcolormesh(Res, HBs, KEzs.T, shading='nearest')
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$KE_z$')
plt.ylabel(r'$H_B^*$')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 8)
plt.pcolormesh(Res, HBs, MEzs.T, shading='nearest')
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$ME_z$')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 9)
plt.pcolormesh(Res, HBs, (KEzs+KExs).T, shading='nearest')
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$KE$')
plt.ylabel(r'$H_B^*$')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 10)
plt.pcolormesh(Res, HBs, (MEzs+MExs).T, shading='nearest')
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'$ME$')
# plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 11)
plt.pcolormesh(Res, HBs, viscous_diss.T, shading='nearest', norm=colors.LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'Viscous dissipation')
plt.ylabel(r'$H_B^*$')
plt.xlabel(r'Reynolds number')

plt.subplot(6, 2, 12)
plt.pcolormesh(Res, HBs, resistive_diss.T, shading='nearest', norm=colors.LogNorm())
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
plt.title(r'Resistive dissipation')
plt.xlabel(r'Reynolds number')

plt.show()
# plt.savefig(plot_fname)
