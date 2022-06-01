import numpy as np
import kolmogorov_EVP
import h5py
import os
from pathlib import Path

HB_star = 1.0
Re = 10.0
Pm = 0.1
N = 33
delta = 0.0
# kz = np.append(np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
                         # np.linspace(0.05, 0.275, num=50, endpoint=False)),
               # np.linspace(0.275, 0.6, num=20))
# ks = np.geomspace(1e-4, 0.5, num=200, endpoint=True)
# ks = np.linspace(0.0, 0.5, num=200, endpoint=True)
ks = np.linspace(0.0, 0.8, num=200, endpoint=True)
out_dir = Path('kolmogorov_ky_kz_scans/Pm{}_HB{}_Re{}_N{}_delta{}'.format(Pm, HB_star, Re, N, delta))
if not out_dir.exists():  # os.path.exists('{:s}/'.format(out_dir)):
    # os.makedirs('{:s}/'.format(out_dir))
    out_dir.mkdir()
    out_fname = out_dir / 'set_0.h5'
else:
    if (out_dir / 'set_0.h5').exists():
        set_names = list(out_dir.glob('set_*.h5'))
        set_nums = [int(set_name.stem.split('_')[-1]) for set_name in set_names]
        print(set_nums)
        out_fname = out_dir / 'set_{}.h5'.format(np.max(set_nums)+1)
    else:
        out_fname = out_dir / 'set_0.h5'
print('saving to ', out_fname)


ns = np.array(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # these are wavenumbers in shear direction
omegas = np.zeros((len(ks), len(ks)), dtype=np.complex128)
# contains_resistive_mode = np.zeros(np.shape(omegas), dtype=bool)
# contains_ordinary_mode = np.zeros(np.shape(omegas), dtype=bool)
# contains_strange_mode = np.zeros(np.shape(omegas), dtype=bool)

for kyi, ky in enumerate(ks):
    if kyi % 10 == 0:
        print('starting kyi=', kyi)
    for kzi, kz in enumerate(ks[1:], 1):  # skip kz=0
        ws, vs = np.linalg.eig(kolmogorov_EVP.Lmat_3D_noP(delta, HB_star, Re, Re * Pm, ky, kz, N))
        sort_inds = np.argsort(np.imag(ws))
        ws_sort = ws[sort_inds]
        omegas[kyi, kzi] = ws_sort[0]
        if False:
        # for evi in range(len(ws_sort[-np.imag(ws_sort) > 0])):  # for every unstable eigenmode at this k
            w_i = ws[sort_inds[evi]]
            v_i = vs[:, sort_inds[evi]]
            if np.abs(np.real(w_i)) > 1e-12:  # if it's an oscillating mode
                contains_resistive_mode[kyi, kzi] = True
            else:  # otherwise it's one of the other two
                phi_i = np.abs(v_i[::2])
                if phi_i[int(N/2)] < 1e-10:  # if it's a strange mode
                    contains_strange_mode[kyi, kzi] = True
                else:
                    contains_ordinary_mode[kyi, kzi] = True
print('starting save')
with h5py.File(out_fname, mode='w-') as file:
    scan_values_grp = file.create_group('scan_values')
    ks_data = scan_values_grp.create_dataset('ks', data=ks)
    ns_data = scan_values_grp.create_dataset('ns', data=ns)

    results_grp = file.create_group('results')
    omegas_data = results_grp.create_dataset('omega', data=omegas)
    # contains_res_data = results_grp.create_dataset('contains_resistive_mode', data=contains_resistive_mode)
    # contains_ord_data = results_grp.create_dataset('contains_ordinary_mode', data=contains_ordinary_mode)
    # contains_str_data = results_grp.create_dataset('contains_strange_mode', data=contains_strange_mode)
print('done')
