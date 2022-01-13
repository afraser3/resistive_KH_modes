import numpy as np
import kolmogorov_EVP
import h5py
import os
from pathlib import Path

HB_stars = np.geomspace(0.1, 1.0, num=100)
Res = np.geomspace(1.0, 1000.0, num=100)
Pm = 2.0
N = 33
delta = 0.0
ks = np.append(np.append(np.geomspace(0.0001, 0.05, num=100, endpoint=False),
                         np.linspace(0.05, 0.275, num=50, endpoint=False)),
               np.linspace(0.275, 0.6, num=20))
out_dir = Path('kolmogorov_2D_scans/Pm{}_N{}_delta{}'.format(Pm, N, delta))
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
omegas = np.zeros((len(Res), len(HB_stars)), dtype=np.complex128)
num_unstables = np.zeros(np.shape(omegas), dtype=int)
k_maxs = np.zeros(np.shape(omegas), dtype=np.float64)
KEs = np.zeros_like(k_maxs)
KExs = np.zeros_like(k_maxs)
KEzs = np.zeros_like(k_maxs)
MEs = np.zeros_like(k_maxs)
MExs = np.zeros_like(k_maxs)
MEzs = np.zeros_like(k_maxs)
viscous_diss = np.zeros_like(k_maxs)
resistive_diss = np.zeros_like(k_maxs)
ki_maxs = np.zeros_like(num_unstables)
# gammaxs_over_k = np.zeros_like(ks)
contains_resistive_mode = np.zeros(np.shape(omegas), dtype=bool)
contains_ordinary_mode = np.zeros(np.shape(omegas), dtype=bool)
contains_strange_mode = np.zeros(np.shape(omegas), dtype=bool)


def energy_from_streamfunc(field, k, kxs, xy_parts=False):
    """
    Calculates kinetic energy sum[ u**2 + w**2] from streamfunction phi
    (equivalent to magnetic energy from flux function, but need to multiply by HB^* or something).
    Parameters
    ----------
    field : FFT of 1d streamfunction
    k : wavenumber in direction of flow
    kxs : list of wavenumbers in direction of shear

    Returns
    -------
    Kinetic energy
    """
    if xy_parts:
        # first part is in direction of shear, second is direction of flow
        return [np.sum(np.abs(k * field) ** 2.0), np.sum(np.abs(kxs * field) ** 2.0)]
    else:
        return np.sum(np.abs(k * field) ** 2.0 + np.abs(kxs * field) ** 2.0)


def diss_from_streamfunc(field, k, kxs):
    return np.sum(np.abs((kxs ** 2.0) * field - (k ** 2.0) * field) ** 2.0)


for rei, re in enumerate(Res):
    if rei % 10 == 0:
        print('starting rei=', rei)
    for hbi, hb in enumerate(HB_stars):
        # array of eigenvalues at each k, of shape (len(ks), Nev = 2*N)
        # ws_over_k = np.array([np.linalg.eig(kolmogorov_EVP.Lmat(0.0, hb, re, re*pm, k, N))[0] for k in ks])
        # modified now to extract eigenvectors along with eigenvalues
        # btw, "ws" and "vs" are notation from np.linalg.eig
        ws_vs_over_k = [np.linalg.eig(kolmogorov_EVP.Lmat(delta, hb, re, re * Pm, k, N)) for k in ks]
        ws_over_k = np.array([ws_vs_over_k[ki][0] for ki in range(len(ks))])
        vs_over_k = np.array([ws_vs_over_k[ki][1] for ki in range(len(ks))])
        for ki in range(len(ks)):
            w = ws_over_k[ki]
            if np.any(-np.imag(w) > 0):  # if any mode at this k is unstable
                v = vs_over_k[ki]
                sort_inds = np.argsort(np.imag(w))
                w_sort = w[sort_inds]
                for evi in range(len(w_sort[-np.imag(w_sort) > 0])):  # for every unstable eigenmode at this k
                    w_i = w[sort_inds[evi]]
                    v_i = v[:, sort_inds[evi]]
                    if np.abs(np.real(w_i)) > 1e-12:  # if it's an oscillating mode
                        contains_resistive_mode[rei, hbi] = True
                    else:  # otherwise it's one of the other two
                        phi_i = np.abs(v_i[::2])
                        if phi_i[int(N/2)] < 1e-10:  # if it's a strange mode
                            contains_strange_mode[rei, hbi] = True
                        else:
                            contains_ordinary_mode[rei, hbi] = True


        # index in array ws_over_k where most-unstable mode is located
        ws_over_k_argmax = np.unravel_index(np.argmax(-np.imag(ws_over_k)), np.shape(ws_over_k))
        # most-unstable growth rate
        omegas[rei, hbi] = ws_over_k[ws_over_k_argmax]
        # index in array ks of the largest growth rate
        ki_maxs[rei, hbi] = ws_over_k_argmax[0]
        # k value of peak in growth rate
        k_maxs[rei, hbi] = ks[ki_maxs[rei, hbi]]
        # full set of eigenvalues at k_max
        ws_at_peak = ws_over_k[ws_over_k_argmax[0]]
        # number of unstable modes at k_max
        num_unstables[rei, hbi] = len(ws_at_peak[-np.imag(ws_at_peak) > 0])
        # most-unstable eigenmode
        full_mode = vs_over_k[ws_over_k_argmax[0], :, ws_over_k_argmax[1]]
        phi = full_mode[::2]
        psi = full_mode[1::2]
        KE = energy_from_streamfunc(phi, k_maxs[rei, hbi], ns)
        ME = energy_from_streamfunc(psi, k_maxs[rei, hbi], ns)*hb
        TE = KE + ME
        KEx, KEz = np.array(energy_from_streamfunc(phi, k_maxs[rei, hbi], ns, True))
        MEx, MEz = np.array(energy_from_streamfunc(psi, k_maxs[rei, hbi], ns, True))*hb
        KEs[rei, hbi] = KE/TE
        MEs[rei, hbi] = ME/TE
        KExs[rei, hbi] = KEx/TE
        KEzs[rei, hbi] = KEz/TE
        MExs[rei, hbi] = MEx/TE
        MEzs[rei, hbi] = MEz/TE
        viscous_diss[rei, hbi] = diss_from_streamfunc(phi, k_maxs[rei, hbi], ns)/(TE*re)
        resistive_diss[rei, hbi] = diss_from_streamfunc(psi, k_maxs[rei, hbi], ns)*hb/(TE*re*Pm)
print('starting save')
with h5py.File(out_fname, mode='w-') as file:
    scan_values_grp = file.create_group('scan_values')
    ks_data = scan_values_grp.create_dataset('ks', data=ks)
    HBs_data = scan_values_grp.create_dataset('HBs', data=HB_stars)
    Res_data = scan_values_grp.create_dataset('Res', data=Res)
    ns_data = scan_values_grp.create_dataset('ns', data=ns)

    results_grp = file.create_group('results')
    omegas_data = results_grp.create_dataset('omega', data=omegas)
    k_maxs_data = results_grp.create_dataset('k_max', data=k_maxs)
    num_unstables_data = results_grp.create_dataset('num_unstable', data=num_unstables)
    KEx_data = results_grp.create_dataset('KEx', data=KExs)
    KEz_data = results_grp.create_dataset('KEz', data=KEzs)
    MEx_data = results_grp.create_dataset('MEx', data=MExs)
    MEz_data = results_grp.create_dataset('MEz', data=MEzs)
    visc_data = results_grp.create_dataset('viscous_diss', data=viscous_diss)
    res_data = results_grp.create_dataset('resistive_diss', data=resistive_diss)
    contains_res_data = results_grp.create_dataset('contains_resistive_mode', data=contains_resistive_mode)
    contains_ord_data = results_grp.create_dataset('contains_ordinary_mode', data=contains_ordinary_mode)
    contains_str_data = results_grp.create_dataset('contains_strange_mode', data=contains_strange_mode)
print('done')

# omegas = np.zeros((len(Res), len(HB_stars)), dtype=np.complex128)
# num_unstables = np.zeros(np.shape(omegas), dtype=np.int)
# k_maxs = np.zeros(np.shape(omegas), dtype=np.float64)
# KEs = np.zeros_like(k_maxs)
# KExs = np.zeros_like(k_maxs)
# KEzs = np.zeros_like(k_maxs)
# MEs = np.zeros_like(k_maxs)
# MExs = np.zeros_like(k_maxs)
# MEzs = np.zeros_like(k_maxs)
# viscous_diss = np.zeros_like(k_maxs)
# resistive_diss = np.zeros_like(k_maxs)