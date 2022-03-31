import numpy as np
import h5py
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.colors as colors

plt.rcParams.update({"text.usetex": True})

Pm1 = 0.1
Pm2 = 1.0
Pm3 = 2.0
N = 33
delta = 0.0
out_dir1 = Path('kolmogorov_2D_scans/Pm{}_N{}_delta{}'.format(Pm1, N, delta))
out_dir2 = Path('kolmogorov_2D_scans/Pm{}_N{}_delta{}'.format(Pm2, N, delta))
out_dir3 = Path('kolmogorov_2D_scans/Pm{}_N{}_delta{}'.format(Pm3, N, delta))
set_num1 = 3  # 2
set_num2 = 2  # 0
set_num3 = 2  # 0
out_file1 = out_dir1 / 'set_{}.h5'.format(set_num1)
out_file2 = out_dir2 / 'set_{}.h5'.format(set_num2)
out_file3 = out_dir3 / 'set_{}.h5'.format(set_num3)

plot_fname = 'plots/growthrate_2dscan_Pmscan.pdf'

with h5py.File(out_file1, mode='r') as file:
    ks1 = np.array(file['scan_values/ks'])
    HBs1 = np.array(file['scan_values/HBs'])
    Res1 = np.array(file['scan_values/Res'])
    ns1 = np.array(file['scan_values/ns'])
    omegas1 = np.array(file['results/omega'])
    k_maxs1 = np.array(file['results/k_max'])
    num_unstables1 = np.array(file['results/num_unstable'])
    contains_resistive_mode1 = np.array(file['results/contains_resistive_mode'])
    contains_strange_mode1 = np.array(file['results/contains_strange_mode'])
    contains_ordinary_mode1 = np.array(file['results/contains_ordinary_mode'])

with h5py.File(out_file2, mode='r') as file:
    ks2 = np.array(file['scan_values/ks'])
    HBs2 = np.array(file['scan_values/HBs'])
    Res2 = np.array(file['scan_values/Res'])
    ns2 = np.array(file['scan_values/ns'])
    omegas2 = np.array(file['results/omega'])
    k_maxs2 = np.array(file['results/k_max'])
    num_unstables2 = np.array(file['results/num_unstable'])
    contains_resistive_mode2 = np.array(file['results/contains_resistive_mode'])
    contains_strange_mode2 = np.array(file['results/contains_strange_mode'])
    contains_ordinary_mode2 = np.array(file['results/contains_ordinary_mode'])

with h5py.File(out_file3, mode='r') as file:
    ks3 = np.array(file['scan_values/ks'])
    HBs3 = np.array(file['scan_values/HBs'])
    Res3 = np.array(file['scan_values/Res'])
    ns3 = np.array(file['scan_values/ns'])
    omegas3 = np.array(file['results/omega'])
    k_maxs3 = np.array(file['results/k_max'])
    num_unstables3 = np.array(file['results/num_unstable'])
    contains_resistive_mode3 = np.array(file['results/contains_resistive_mode'])
    contains_strange_mode3 = np.array(file['results/contains_strange_mode'])
    contains_ordinary_mode3 = np.array(file['results/contains_ordinary_mode'])

scale = 0.6
# plt.figure(figsize=(scale*21, scale*5))
# plt.subplot(1, 3, 1)
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(scale*21, scale*5))
ax1, ax2, ax3 = axes.flat
im = ax1.contourf(Res1, HBs1, -np.imag(omegas1.T), levels=np.geomspace(1e-5, 1e0, 11), norm=colors.LogNorm())
# ax1.contourf(Res1, HBs1, np.abs(np.real(omegas1)).T, [0.0, 1e-10], hatches=['/', None], alpha=0)
# ax1.contour(Res1, HBs1, np.abs(np.real(omegas1)).T, [0.0, 1e-10], colors='k', linewidths=2.0)
ax1.contourf(Res1, HBs1, contains_ordinary_mode1.astype(int).T, [-1, 0, 1], hatches=[None, '\\', None], alpha=0)
ax1.contour(Res1, HBs1, contains_ordinary_mode1.astype(int).T, [0, 1], colors='k', linewidths=2.0)
ax1.contourf(Res1, HBs1, contains_strange_mode1.astype(int).T, [-1, 0, 1], hatches=[None, '/', None], alpha=0)
ax1.contour(Res1, HBs1, contains_strange_mode1.astype(int).T, [0, 1], colors='k', linewidths=2.0)
ax1.set_ylabel(r'$C_B$')
# plt.xlabel(r'$\mathrm{Re}$')
# ax1.set_title(r'$\mathrm{{Re}}[\lambda]$ for $Pm = {:3.1f}$'.format(Pm1))
ax1.set_title(r'$Pm = {:3.1f}$'.format(Pm1))

# plt.subplot(1, 3, 2)
im = ax2.contourf(Res2, HBs2, -np.imag(omegas2.T), levels=np.geomspace(1e-5, 1e0, 11), norm=colors.LogNorm())
# ax2.contourf(Res2, HBs2, np.abs(np.real(omegas2)).T, [0.0, 1e-10], hatches=['/', None], alpha=0)
# ax2.contour(Res2, HBs2, np.abs(np.real(omegas2)).T, [0.0, 1e-10], colors='k', linewidths=2.0)
ax2.contourf(Res2, HBs2, contains_ordinary_mode2.astype(int).T, [-1, 0, 1], hatches=[None, '\\', None], alpha=0)
ax2.contour(Res2, HBs2, contains_ordinary_mode2.astype(int).T, [0, 1], colors='k', linewidths=2.0)
ax2.contourf(Res2, HBs2, contains_strange_mode2.astype(int).T, [-1, 0, 1], hatches=[None, '/', None], alpha=0)
ax2.contour(Res2, HBs2, contains_strange_mode2.astype(int).T, [0, 1], colors='k', linewidths=2.0)
# ax2.xlabel(r'$\mathrm{Re}$')
# ax2.set_title(r'$\mathrm{{Re}}[\lambda]$ for $Pm = {:3.1f}$'.format(Pm2))
ax2.set_title(r'$Pm = {:3.1f}$'.format(Pm2))

# plt.subplot(1, 3, 3)
im = ax3.contourf(Res3, HBs3, -np.imag(omegas3.T), levels=np.geomspace(1e-5, 1e0, 11), norm=colors.LogNorm())
fig.colorbar(im, ax=axes.ravel().tolist(), label=r'$\mathrm{Re}[\lambda]$')
# ax3.contourf(Res3, HBs3, np.abs(np.real(omegas3)).T, [0.0, 1e-10], hatches=['/', None], alpha=0)
# ax3.contour(Res3, HBs3, np.abs(np.real(omegas3)).T, [0.0, 1e-10], colors='k', linewidths=2.0)
ax3.contourf(Res3, HBs3, contains_ordinary_mode3.astype(int).T, [-1, 0, 1], hatches=[None, '\\', None], alpha=0)
ax3.contour(Res3, HBs3, contains_ordinary_mode3.astype(int).T, [0, 1], colors='k', linewidths=2.0)
ax3.contourf(Res3, HBs3, contains_strange_mode3.astype(int).T, [-1, 0, 1], hatches=[None, '/', None], alpha=0)
ax3.contour(Res3, HBs3, contains_strange_mode3.astype(int).T, [0, 1], colors='k', linewidths=2.0)
# re2 = 1e3
# cb2 = 6e-1  # cb1*(re1/re2)**2.0
# cb1 = 1e0  # cb2*(re2/re1)**2.0
# re1 = np.sqrt(cb2/cb1)*re2
# ax3.plot([re1, re2], [cb1, cb2], '--', c='orange')
# ax3.xlabel(r'$\mathrm{Re}$')
# ax3.set_title(r'$\mathrm{{Re}}[\lambda]$ for $Pm = {:3.1f}$'.format(Pm3))
ax3.set_title(r'$Pm = {:3.1f}$'.format(Pm3))

for ax in axes.flat:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$Re$')
    ax.axhline(0.5, c='r', lw=0.5)

for ax in [ax2, ax3]:
    ax.set_yticklabels([])
    ax.set_yticklabels([], minor=True)

ax1.axvline(np.sqrt(2*1.1/(0.1*0.9)), c='r')

# plt.show()
plt.savefig(plot_fname, bbox_inches='tight')
