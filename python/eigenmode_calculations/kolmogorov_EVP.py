"""Solves the eigenvalue problem for KH instability of sinusoidal flow

Sets up the eigenvalue problem corresponding to the KH instability of a
sinusoidal flow profile in MHD with a uniform, flow-aligned magnetic 
field. Includes various functions for finding just the most-unstable
growth rate, or the complex frequency, or a scan over k, or a helper
function for using root-finding algorithms for finding the flow speed
such that the growth rate matches some value.

Can be used with viscosity/resistivity (ideal=False) or without 
(ideal=True).

Note that most methods return frequencies or
growth rates that are normalized to the finger width/speed, like in 
Fig 3 of Harrington & Garaud.

Exceptions
----------
KrangeError(Exception)
    Used for handling the situation where you want the most-
    unstable growth rate for a given range of ks, but the most-
    unstable growth rate occurs at one of the edges of that rage,
    and thus you likely have not found a true local maximum.

Methods
-------
Deln(k,n,delta)
    Calculates Delta_n from my LaTeX notes (for setting up KH EVP)
Lmat(delta, M2, Re, Rm, k, N, ideal=False)
    Constructs the linear operator whose eigenvalues are the
    complex frequencies for the KH eigenmodes, i.e., the modes
    are taken to go like f(t) ~ exp[i omega t], and the eigenvalues
    of this matrix are the various omega for different modes.
gamfromL(L)
    Calls numpy.linalg.eig to get the eigenvalues. Returns the
    growth rate gamma = -np.imag(omega) of the most unstable mode
omegafromL(L)
    Returns the complex frequency instead (should just merge these
    two functions)
gamma_over_k(delta, M2, Re, Rm, ks, N, ideal=False)
    Calls gamfromL for each k in array ks, returns the resulting
    array of growth rates gamma[ks]
omega_over_k(...)
    Same as above but complex frequencies
gammax_kscan(delta, M2, Re, Rm, ks, N, ideal=False, badks_except=False)
    Same as gamma_over_k but returns just the most unstable growth
    rate over the provided range of ks. If the result is positive
    and badks_except=True, it will check to see if the maximum
    occurred at either the highest or lowest k in ks, and throw an
    error if it did. This is so that you can make sure your local
    maximum isn't just a maximum because of the range you chose.
KHparams_from_fingering(w, lhat, HB, Pr, DB)
    Returns H_B^* (equivalent to 1/M_A^2), Re, and Rm defined in
    terms of the finger's speed, width, and fluid parameters.
gammax_minus_lambda(w, lamhat, lhat, HB, Pr, DB, delta, ks, N, 
            ideal=False, badks_exception=False, CH=1.66)
    Just a helper function around gammax_kscan. Instead of
    returning gamma, it returns gamma*w_f*l_f - C_H*lambda_f, i.e.,
    equations 30 and 31 in Harrington & Garaud except without the
    fit, and written in the form F(stuff) = 0 so that saturation is
    given by the roots of F.

"""
# TODO: rename KH growth rate from gamma to sigma for consistency w/ HG18
import numpy as np
import fingering_modes
import scipy


class KrangeError(Exception):
    pass


def KHparams_from_fingering(w, lhat, HB, Pr, DB):
    M2 = HB / w ** 2.0
    Re = w / (Pr * lhat)
    Rm = w / (DB * lhat)
    return [M2, Re, Rm]


def Deln(k, n, delta, finger_norm=False, k0=1.0):  # \Delta_n in my notes. So simple, probably shouldn't be a function
    if finger_norm:
        return k ** 2.0 + k0**2.0 * (n + delta)**2.0
    else:
        return k ** 2.0 + (n + delta) ** 2.0


def Lmat(delta, M2, Re, Rm, k, N, ideal=False):
    """Returns the linear operator for the KH instability

    Note that the eigenvalues are the complex frequencies, and the
    eigenvectors are the streamfunction and flux function mixed together,
    with the nth entry being the streamfunction at some wavenumber, and
    the n+1th being the flux function at a wavenumber.

    The eigenvalue problem this corresponds to is normalized to the flow speed and length scale, and background field.

    Parameters
    ----------
    delta : float
        This should be in the range 0 <= delta <= 0.5 and indicates
        the periodicity of the KH mode relative to the wavelength
        of the sinusoidal shear flow. See LaTeX notes; should
        probably be left at delta=0.0
    M2 : float
        The parameter H_B^* in Harrington & Garaud
    Re : float
        Reynolds number
    Rm : float
        Magnetic Reynolds number
    k : float
        Wavenumber in direction of flow
    N : int (ODD NUMBER)
        Numerical resolution in direction of shear
    ideal : Bool, default=False
        Whether or not to set viscosity, resistivity -> 0
        (if True then Re and Rm don't matter)

    Returns
    -------
    L : 2N x 2N numpy array
        Matrix whose eigenvalues are complex frequencies of KH modes
    """
    diss = 1.0 - ideal  # =0 for ideal=True, =1 for ideal=False
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # the n over which we sum the Fourier series
    ms = list(range(-N + 1, N + 1, 1))  # this is going to be twice as long so we can loop over each n twice, once for phi and once for psi

    # the following few lines just sets up arrays of Delta_n
    delns = [Deln(k, n, delta) for n in ns]
    delns_m = np.zeros_like(ms, dtype=np.float64)
    for i, m in enumerate(ms):
        if m % 2 == 0:
            delns_m[i] = Deln(k, m / 2, delta)
        else:
            delns_m[i] = Deln(k, (m - 1) / 2, delta)

    M = 2 * N
    L = np.zeros((M, M), dtype=np.complex128)

    # first fill in the entries that aren't near the edges
    for i, m in enumerate(ms):
        deltan = delns_m[i]
        # deltanp1 = delns_m[i+2]
        # deltanm1 = delns_m[i-2]
        if i > 1 and i < len(ms) - 2:  # avoid entries near edges
            deltanp1 = delns_m[i + 2]
            deltanm1 = delns_m[i - 2]
            if m % 2 == 0:  # phi entries
                n = m / 2
                # phi_n, phi_n part
                L[i, i] = (1.0j) * (diss / Re) * deltan
                # phi_n, psi_n part
                L[i, i + 1] = M2 * k
                # phi_n, phi_n+1
                L[i, i + 2] = -k * (1 - deltanp1) / (2.0j * deltan)
                if not np.isfinite(L[i, i + 2]):
                    # Pretty sure I can get rid of this now -- I was debugging 0/0 errors, which I think only happen
                    # if you try to solve the system at k=0, which isn't interesting. And if it is, then the way to
                    # go about it is to multiply both sides of the linear system by Delta_n, and solve as a
                    # generalized eigenvalue problem
                    print(-k * (1 - deltanp1))
                    print(2.0j * deltan)
                # phi_n, phi_n-1
                L[i, i - 2] = k * (1 - deltanm1) / (2.0j * deltan)
            else:  # psi entries
                # psi_n, psi_n
                L[i, i] = (1.0j) * deltan * diss / Rm
                # psi_n, phi_n
                L[i, i - 1] = k
                # psi_n, psi_n+1
                L[i, i + 2] = k / (2.0j)
                # psi_n, psi_n-1
                L[i, i - 2] = -k / (2.0j)
    # now do the edges
    # first, the most negative phi
    L[0, 0] = (1.0j) * delns_m[0] * diss / Re
    L[0, 1] = M2 * k
    L[0, 2] = -k * (1 - delns_m[2]) / (2.0j * delns_m[0])
    # most negative psi
    L[1, 1] = (1.0j) * delns_m[1] * diss / Rm
    L[1, 0] = k
    L[1, 3] = k / (2.0j)
    # most positive phi
    L[-2, -2] = (1.0j) * delns_m[-2] * diss / Re
    L[-2, -1] = M2 * k
    L[-2, -4] = k * (1 - delns_m[-4]) / (2.0j * delns_m[-2])
    # most positive psi
    L[-1, -1] = (1.0j) * delns_m[-1] * diss / Rm
    L[-1, -2] = k
    L[-1, -3] = -k / (2.0j)
    return L


def Lmat_finger_norm(HB, DB, Pr, R0, tau, A_phi, A_T, A_C, k0, kz, delta, N, ideal=False, zero_T_C_diss=False):
    diss = 1.0 - ideal  # =0 for ideal=True, =1 for ideal=False
    diss_TC = 1.0 - zero_T_C_diss
    M = int(4*N)  # size of matrix
    L = np.zeros((M, M), dtype=np.complex128)
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))  # e.g. [-4, -3, -2, ..., 4] for N=9
    ns_long, field_iter = np.meshgrid(ns, range(4))
    ns_long = ns_long.T.flatten()  # [-4, -4, -4, -4, -3, -3, -3, -3, ...] so four copies of ns
    field_iter = field_iter.T.flatten()  # [0,1,2,3,0,1,2,3,...] useful for looping over the 4 physical fields
    delns_m = Deln(kz, ns_long, delta, True, k0)  # \Delta_n from my notes, but four copies just like ns_long

    for i, n in enumerate(ns_long):
        delta_n = delns_m[i]  # \Delta_n
        if n != ns[0] and n != ns[-1]:  # avoid filling in the edges of the matrix for now
            delta_nm1 = delns_m[i-4]  # \Delta_{n-1}
            delta_np1 = delns_m[i+4]  # \Delta_{n+1}
            if field_iter[i] == 0:  # phi entries
                L[i, i] = 1.0j * Pr * delta_n * diss  # phi_n, phi_n part
                L[i, i + 1] = HB * kz  # phi_n, psi_n
                L[i, i + 2] = Pr * (n + delta) * k0 / delta_n  # phi_n, T_n
                L[i, i + 3] = -Pr * (n + delta) * k0 / delta_n  # phi_n, C_n
                L[i, i + 4] = -A_phi * kz * k0 * (k0**2.0 - delta_np1) / (2.0j * delta_n)  # phi_n, phi_{n+1}
                L[i, i - 4] = A_phi * kz * k0 * (k0**2.0 - delta_nm1) / (2.0j * delta_n)  # phi_n, phi_{n-1}
            if field_iter[i] == 1:  # psi entries
                L[i, i] = 1.0j * DB * delta_n * diss  # psi_n, psi_n part
                L[i, i - 1] = kz  # psi_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # psi_n, psi_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # psi_n, psi_{n-1}
            if field_iter[i] == 2:  # T entries
                L[i, i] = 1.0j * delta_n * diss_TC  # T_n, T_n part
                L[i, i - 2] = (n + delta) * k0  # T_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # T_n, T_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # T_n, T_{n-1}
                L[i, i + 2] = -A_T * kz * k0 / 2.0  # T_n, phi_{n+1}
                L[i, i - 6] = L[i, i + 2]  # T_n, phi_{n-1}
            if field_iter[i] == 3:  # C entries
                L[i, i] = 1.0j * tau * delta_n * diss_TC  # C_n, C_n part
                L[i, i - 3] = (n + delta) * k0 / R0  # C_n, phi_n
                L[i, i + 4] = A_phi * kz * k0 / 2.0j  # C_n, C_{n+1}
                L[i, i - 4] = -L[i, i + 4]  # C_n, C_{n-1}
                L[i, i + 1] = -A_C * kz * k0 / 2.0  # C_n, phi_{n+1}
                L[i, i - 7] = L[i, i + 1]  # C_n, phi_{n-1}
    # Now fill in the edges
    # First, most negative phi part
    L[0, 0] = 1.0j * Pr * delns_m[0] * diss  # phi_-N, phi_-N
    L[0, 1] = HB * kz  # phi_-N, psi_-N
    L[0, 2] = Pr * (ns[0] + delta) * k0 / delns_m[0]  # phi_-N, T_-N
    L[0, 3] = -Pr * (ns[0] + delta) * k0 / delns_m[0]  # phi_-N, C_-N
    L[0, 4] = -A_phi * kz * k0 * (k0**2.0 - delns_m[4]) / (2.0j * delns_m[0])  # phi_-N, phi_{-N + 1}
    # Most positive phi part
    L[-4, -4] = 1.0j * Pr * delns_m[-4] * diss  # phi_N, phi_N
    L[-4, -3] = HB * kz  # phi_N, psi_N
    L[-4, -2] = Pr * (ns[-1] + delta) * k0 / delns_m[-4]  # phi_N, T_N
    L[-4, -1] = -Pr * (ns[-1] + delta) * k0 / delns_m[-4]  # phi_N, C_N
    L[-4, -8] = A_phi * kz * k0 * (k0**2.0 - delns_m[-8]) / (2.0j * delns_m[-4])  # phi_N, phi_{N-1}
    # Most negative psi part
    L[1, 0] = kz  # psi_-N, phi_-N
    L[1, 1] = 1.0j * DB * delns_m[1] * diss  # psi_-N, psi_-N
    L[1, 5] = A_phi * kz * k0 / 2.0j  # psi_-N, psi_{-N + 1}
    # Most positive psi part
    L[-3, -4] = kz  # psi_N, phi_N
    L[-3, -3] = 1.0j * DB * delns_m[-3] * diss  # psi_N, psi_N
    L[-3, -7] = -A_phi * kz * k0 / 2.0j  # psi_N, psi_{N-1}
    # Most negative T part
    L[2, 0] = (ns[0] + delta) * k0  # T_-N, phi_-N
    L[2, 2] = 1.0j * delns_m[2] * diss_TC  # T_-N, T_-N
    L[2, 4] = -A_T * kz * k0 / 2.0  # T_-N, phi_{-N + 1}
    L[2, 6] = A_phi * kz * k0 / 2.0j  # T_-N, T_{-N + 1}
    # Most positive T part
    L[-2, -4] = (ns[-1] + delta) * k0  # T_N, phi_N
    L[-2, -2] = 1.0j * delns_m[-2] * diss_TC  # T_N, T_N
    L[-2, -8] = -A_T * kz * k0 / 2.0  # T_N, phi_{N - 1}
    L[-2, -6] = -A_phi * kz * k0 / 2.0j  # T_N, T_{N - 1}
    # Most negative C part
    L[3, 0] = (ns[0] + delta) * k0 / R0  # C_-N, phi_-N
    L[3, 3] = 1.0j * delns_m[3] * tau * diss_TC  # C_-N, C_-N
    L[3, 4] = -A_C * kz * k0 / 2.0  # C_-N, phi_{-N + 1}
    L[3, 7] = A_phi * kz * k0 / 2.0j  # C_-N, C_{-N + 1}
    # Most positive C part
    L[-1, -4] = (ns[-1] + delta) * k0 / R0  # C_N, phi_N
    L[-1, -1] = 1.0j * delns_m[-1] * tau * diss_TC  # C_N, C_N
    L[-1, -8] = -A_T * kz * k0 / 2.0  # C_N, phi_{N - 1}
    L[-1, -5] = -A_phi * kz * k0 / 2.0j  # C_N, C_{N - 1}
    return L


def Lmat_Mmat_3D(delta, M2, Re, Rm, ky, kz, N, ideal=False):
    diss = 1.0 - ideal
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))
    L = np.zeros((7*N, 7*N), dtype=np.complex128)
    M = np.zeros_like(L)  # not the same as the M from the 2D function. This is the "mass matrix" for the generalized eigenvalue problem

    for i, n in enumerate(ns):
        for j in [0, 1, 2, 4, 5, 6]:  # quick loop over fields to construct M matrix
            M[7*i+j, 7*i+j] = 1.0  # make it the identity for all but the p (j = 3) parts
        delta_n = ky**2.0 + kz**2.0 + (n + delta)**2.0
        # start with the u_n rows
        if n > ns[0]:
            L[7*i, 7*(i-1)] = -kz/2.0j  # u_{n-1} column
        if n < ns[-1]:
            L[7*i, 7*(i+1)] = kz/2.0j  # u_{n+1} column
        L[7*i, 7*i] = diss*1.0j*delta_n/Re  # u_n column
        L[7*i, 7*i+3] = -(n + delta)  # p_n column
        L[7*i, 7*i+4] = M2*kz  # b_{x,n} column
        L[7*i, 7*i+6] = -M2*(n + delta)  # b_{z,n} column

        # v_n rows
        if n > ns[0]:
            L[7*i+1, 7*(i-1)+1] = -kz/2.0j  # v_{n-1} column
        if n < ns[-1]:
            L[7*i+1, 7*(i+1)+1] = kz/2.0j  # v_{n+1} column
        L[7*i+1, 7*i+1] = diss*1.0j*delta_n/Re  # v_n column
        L[7*i+1, 7*i+3] = -ky  # p_n column
        L[7*i+1, 7*i+5] = M2*kz  # b_{y,n} column
        L[7*i+1, 7*i+6] = -M2*ky  # b_{z,n} column

        # w_n rows
        if n > ns[0]:
            L[7*i+2, 7*(i-1)] = 0.5j  # u_{n-1} column
        if n < ns[-1]:
            L[7*i+2, 7*(i+1)] = 0.5j  # u_{n+1} column
        if n > ns[0]:
            L[7*i+2, 7*(i-1)+2] = -kz/2.0j  # w_{n-1} column
        if n < ns[-1]:
            L[7*i+2, 7*(i+1)+2] = kz/2.0j  # w_{n+1} column
        L[7*i+2, 7*i+2] = diss*1.0j*delta_n/Re  # w_n column
        L[7*i+2, 7*i+3] = -kz  # p_n column

        # p_n rows
        L[7*i+3, 7*i] = (n + delta)  # u_n column
        L[7*i+3, 7*i+1] = ky  # v_n column
        L[7*i+3, 7*i+2] = kz  # w_n column

        # b_{x,n} rows
        L[7*i+4, 7*i] = kz  # u_n column
        if n > ns[0]:
            L[7*i+4, 7*(i-1)+4] = -kz/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[7*i+4, 7*(i+1)+4] = kz/2.0j  # b_{x,n+1} column
        L[7*i+4, 7*i+4] = diss*1.0j*delta_n/Rm  # b_{x,n} column

        # b_{y,n} rows
        L[7*i+5, 7*i+1] = kz  # v_n column
        if n > ns[0]:
            L[7*i+5, 7*(i-1)+5] = -kz/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[7*i+5, 7*(i+1)+5] = kz/2.0j  # b_{y,n+1} column
        L[7*i+5, 7*i+5] = diss*1.0j*delta_n/Rm  # b_{y,n} column

        # b_{z,n} rows
        L[7*i+6, 7*i] = -(n + delta)  # u_n column
        L[7*i+6, 7*i+1] = -ky  # v_n column
        if n > ns[0]:
            L[7*i+6, 7*(i-1)+4] = (n + delta)/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[7*i+6, 7*(i+1)+4] = -(n + delta)/2.0j  # b_{x,n+1} column
        if n > ns[0]:
            L[7*i+6, 7*(i-1)+5] = ky/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[7*i+6, 7*(i+1)+5] = -ky/2.0j  # b_{y,n+1} column
        L[7*i+6, 7*i+6] = diss*1.0j*delta_n/Rm  # b_{z,n} column
    return L, M


# The following noP functions were various attempts to turn the generalized eigenvalue
# problem (for 3D perturbations on planar shear flow) into an ordinary eigenvalue problem
# by using div(u)=0 to eliminate the pressure, in hopes of getting a more numerically-stable
# matrix. As I recall, the 3rd one worked well... or maybe it was the 2nd?
def Lmat_3D_noP(delta, M2, Re, Rm, ky, kz, N, ideal=False):
    diss = 1.0 - ideal
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))
    L = np.zeros((6*N, 6*N), dtype=np.complex128)

    for i, n in enumerate(ns):
        delta_n = ky**2.0 + kz**2.0 + (n + delta)**2.0
        # start with the u_n rows
        if n > ns[0]:
            L[6*i, 6*(i-1)] = -kz/2.0j  # u_{n-1} column
        if n < ns[-1]:
            L[6*i, 6*(i+1)] = kz/2.0j  # u_{n+1} column
        L[6*i, 6*i] = diss*1.0j*delta_n/Re  # u_n column
        # L[6*i, 6*i+3] = -(n + delta)  # p_n column
        L[6*i, 6*i+3] = M2*kz  # b_{x,n} column
        L[6*i, 6*i+5] = -M2*(n + delta)  # b_{z,n} column
        pfac = (n + delta) / delta_n
        if n > ns[0]:
            L[6*i, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            L[6*i, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            L[6*i, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
        if n < ns[-1]:
            L[6*i, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            L[6*i, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            L[6*i, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
        L[6*i, 6*i+3] += -pfac * M2 * kz * (n + delta)
        L[6*i, 6*i+4] += -pfac * M2 * ky * kz
        L[6*i, 6*i+5] += pfac * M2 * (ky**2.0 + (n + delta)**2.0)
        # L[6*i, 6*i+5] -= pfac * M2 * (n + delta)

        # v_n rows
        if n > ns[0]:
            L[6*i+1, 6*(i-1)+1] = -kz/2.0j  # v_{n-1} column
        if n < ns[-1]:
            L[6*i+1, 6*(i+1)+1] = kz/2.0j  # v_{n+1} column
        L[6*i+1, 6*i+1] = diss*1.0j*delta_n/Re  # v_n column
        # L[6*i+1, 6*i+3] = -ky  # p_n column
        L[6*i+1, 6*i+4] = M2*kz  # b_{y,n} column
        L[6*i+1, 6*i+5] = -M2*ky  # b_{z,n} column
        pfac = ky / delta_n
        if n > ns[0]:
            L[6*i+1, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            L[6*i+1, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            L[6*i+1, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
        if n < ns[-1]:
            L[6*i+1, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            L[6*i+1, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            L[6*i+1, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
        L[6*i+1, 6*i+3] += -pfac * M2 * kz * (n + delta)
        L[6*i+1, 6*i+4] += -pfac * M2 * ky * kz
        L[6*i+1, 6*i+5] += pfac * M2 * (ky**2.0 + (n + delta)**2.0)
        # L[6*i+1, 6*i+5] -= pfac * M2 * ky

        # w_n rows
        if n > ns[0]:
            L[6*i+2, 6*(i-1)] = 0.5j  # u_{n-1} column
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)] = 0.5j  # u_{n+1} column
        if n > ns[0]:
            L[6*i+2, 6*(i-1)+2] = -kz/2.0j  # w_{n-1} column
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)+2] = kz/2.0j  # w_{n+1} column
        L[6*i+2, 6*i+2] = diss*1.0j*delta_n/Re  # w_n column
        # L[6*i+2, 6*i+3] = -kz  # p_n column
        pfac = kz / delta_n
        if n > ns[0]:
            L[6*i+2, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            L[6*i+2, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            L[6*i+2, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            L[6*i+2, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            L[6*i+2, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
        L[6*i+2, 6*i+3] += -pfac * M2 * kz * (n + delta)
        L[6*i+2, 6*i+4] += -pfac * M2 * ky * kz
        L[6*i+2, 6*i+5] += pfac * M2 * (ky**2.0 + (n + delta)**2.0)
        # L[6*i+2, 6*i+5] -= pfac * M2 * kz

        # b_{x,n} rows
        L[6*i+3, 6*i] = kz  # u_n column
        if n > ns[0]:
            L[6*i+3, 6*(i-1)+3] = -kz/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[6*i+3, 6*(i+1)+3] = kz/2.0j  # b_{x,n+1} column
        L[6*i+3, 6*i+3] = diss*1.0j*delta_n/Rm  # b_{x,n} column

        # b_{y,n} rows
        L[6*i+4, 6*i+1] = kz  # v_n column
        if n > ns[0]:
            L[6*i+4, 6*(i-1)+4] = -kz/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[6*i+4, 6*(i+1)+4] = kz/2.0j  # b_{y,n+1} column
        L[6*i+4, 6*i+4] = diss*1.0j*delta_n/Rm  # b_{y,n} column

        # b_{z,n} rows
        L[6*i+5, 6*i] = -(n + delta)  # u_n column
        L[6*i+5, 6*i+1] = -ky  # v_n column
        if n > ns[0]:
            L[6*i+5, 6*(i-1)+3] = (n + delta)/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[6*i+5, 6*(i+1)+3] = -(n + delta)/2.0j  # b_{x,n+1} column
        if n > ns[0]:
            L[6*i+5, 6*(i-1)+4] = ky/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[6*i+5, 6*(i+1)+4] = -ky/2.0j  # b_{y,n+1} column
        L[6*i+5, 6*i+5] = diss*1.0j*delta_n/Rm  # b_{z,n} column
    return L


def Lmat_3D_noP2(delta, M2, Re, Rm, ky, kz, N, ideal=False):
    diss = 1.0 - ideal
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))
    L = np.zeros((6*N, 6*N), dtype=np.complex128)

    for i, n in enumerate(ns):
        delta_n = ky**2.0 + kz**2.0 + (n + delta)**2.0
        # start with the u_n rows
        if n > ns[0]:
            L[6*i, 6*(i-1)] = -kz/2.0j  # u_{n-1} column
        if n < ns[-1]:
            L[6*i, 6*(i+1)] = kz/2.0j  # u_{n+1} column
        L[6*i, 6*i] = diss*1.0j*delta_n/Re  # u_n column
        # L[6*i, 6*i+3] = -(n + delta)  # p_n column
        L[6*i, 6*i+3] = M2*kz  # b_{x,n} column
        L[6*i, 6*i+5] = -M2*(n + delta)  # b_{z,n} column
        pfac = (n + delta) / delta_n
        if n > ns[0]:
            L[6*i, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            L[6*i, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            L[6*i, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
        if n < ns[-1]:
            L[6*i, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            L[6*i, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            L[6*i, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
        # L[6*i, 6*i+3] += -M2 * kz * (n + delta)
        # L[6*i, 6*i+4] += -M2 * ky * kz
        # L[6*i, 6*i+5] += M2 * (ky**2.0 + (n + delta)**2.0)
        L[6*i, 6*i+5] += M2 * (n + delta)

        # v_n rows
        if n > ns[0]:
            L[6*i+1, 6*(i-1)+1] = -kz/2.0j  # v_{n-1} column
        if n < ns[-1]:
            L[6*i+1, 6*(i+1)+1] = kz/2.0j  # v_{n+1} column
        L[6*i+1, 6*i+1] = diss*1.0j*delta_n/Re  # v_n column
        # L[6*i+1, 6*i+3] = -ky  # p_n column
        L[6*i+1, 6*i+4] = M2*kz  # b_{y,n} column
        L[6*i+1, 6*i+5] = -M2*ky  # b_{z,n} column
        pfac = ky / delta_n
        if n > ns[0]:
            L[6*i+1, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            L[6*i+1, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            L[6*i+1, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
        if n < ns[-1]:
            L[6*i+1, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            L[6*i+1, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            L[6*i+1, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
        # L[6*i+1, 6*i+3] += -M2 * kz * (n + delta)
        # L[6*i+1, 6*i+4] += -M2 * ky * kz
        # L[6*i+1, 6*i+5] += M2 * (ky**2.0 + (n + delta)**2.0)
        L[6*i+1, 6*i+5] += M2 * ky

        # w_n rows
        if n > ns[0]:
            L[6*i+2, 6*(i-1)] = 0.5j  # u_{n-1} column
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)] = 0.5j  # u_{n+1} column
        if n > ns[0]:
            L[6*i+2, 6*(i-1)+2] = -kz/2.0j  # w_{n-1} column
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)+2] = kz/2.0j  # w_{n+1} column
        L[6*i+2, 6*i+2] = diss*1.0j*delta_n/Re  # w_n column
        # L[6*i+2, 6*i+3] = -kz  # p_n column
        pfac = kz / delta_n
        if n > ns[0]:
            L[6*i+2, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            L[6*i+2, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            L[6*i+2, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            L[6*i+2, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            L[6*i+2, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
        # L[6*i+2, 6*i+3] += -M2 * kz * (n + delta)
        # L[6*i+2, 6*i+4] += -M2 * ky * kz
        # L[6*i+2, 6*i+5] += M2 * (ky**2.0 + (n + delta)**2.0)
        L[6*i+2, 6*i+5] += M2 * kz

        # b_{x,n} rows
        L[6*i+3, 6*i] = kz  # u_n column
        if n > ns[0]:
            L[6*i+3, 6*(i-1)+3] = -kz/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[6*i+3, 6*(i+1)+3] = kz/2.0j  # b_{x,n+1} column
        L[6*i+3, 6*i+3] = diss*1.0j*delta_n/Rm  # b_{x,n} column

        # b_{y,n} rows
        L[6*i+4, 6*i+1] = kz  # v_n column
        if n > ns[0]:
            L[6*i+4, 6*(i-1)+4] = -kz/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[6*i+4, 6*(i+1)+4] = kz/2.0j  # b_{y,n+1} column
        L[6*i+4, 6*i+4] = diss*1.0j*delta_n/Rm  # b_{y,n} column

        # b_{z,n} rows
        L[6*i+5, 6*i] = -(n + delta)  # u_n column
        L[6*i+5, 6*i+1] = -ky  # v_n column
        if n > ns[0]:
            L[6*i+5, 6*(i-1)+3] = (n + delta)/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[6*i+5, 6*(i+1)+3] = -(n + delta)/2.0j  # b_{x,n+1} column
        if n > ns[0]:
            L[6*i+5, 6*(i-1)+4] = ky/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[6*i+5, 6*(i+1)+4] = -ky/2.0j  # b_{y,n+1} column
        L[6*i+5, 6*i+5] = diss*1.0j*delta_n/Rm  # b_{z,n} column
    return L


def Lmat_3D_noP3(delta, M2, Re, Rm, ky, kz, N, ideal=False):
    diss = 1.0 - ideal
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))
    L = np.zeros((6*N, 6*N), dtype=np.complex128)

    for i, n in enumerate(ns):
        delta_n = ky**2.0 + kz**2.0 + (n + delta)**2.0
        # start with the u_n rows
        if n > ns[0]:
            L[6*i, 6*(i-1)] = -kz/2.0j  # u_{n-1} column
        if n < ns[-1]:
            L[6*i, 6*(i+1)] = kz/2.0j  # u_{n+1} column
        L[6*i, 6*i] = diss*1.0j*delta_n/Re  # u_n column
        # L[6*i, 6*i+3] = -(n + delta)  # p_n column
        L[6*i, 6*i+3] = M2*kz  # b_{x,n} column
        L[6*i, 6*i+5] = -M2*(n + delta)  # b_{z,n} column
        pfac = (n + delta) / delta_n
        if n > ns[0]:
            # L[6*i, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            # L[6*i, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            # L[6*i, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
            L[6*i, 6*(i-1)] += pfac * -1.0j * kz
        if n < ns[-1]:
            # L[6*i, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            # L[6*i, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            # L[6*i, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
            L[6*i, 6*(i+1)] += pfac * -1.0j * kz

        # L[6*i, 6*i+3] += -M2 * kz * (n + delta)
        # L[6*i, 6*i+4] += -M2 * ky * kz
        # L[6*i, 6*i+5] += M2 * (ky**2.0 + (n + delta)**2.0)
        L[6*i, 6*i+5] += M2 * (n + delta)

        # v_n rows
        if n > ns[0]:
            L[6*i+1, 6*(i-1)+1] = -kz/2.0j  # v_{n-1} column
        if n < ns[-1]:
            L[6*i+1, 6*(i+1)+1] = kz/2.0j  # v_{n+1} column
        L[6*i+1, 6*i+1] = diss*1.0j*delta_n/Re  # v_n column
        # L[6*i+1, 6*i+3] = -ky  # p_n column
        L[6*i+1, 6*i+4] = M2*kz  # b_{y,n} column
        L[6*i+1, 6*i+5] = -M2*ky  # b_{z,n} column
        pfac = ky / delta_n
        if n > ns[0]:
            # L[6*i+1, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            # L[6*i+1, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            # L[6*i+1, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
            L[6*i+1, 6*(i-1)] += pfac * -1.0j * kz
        if n < ns[-1]:
            # L[6*i+1, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            # L[6*i+1, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            # L[6*i+1, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
            L[6*i+1, 6*(i+1)] += pfac * -1.0j * kz
        # L[6*i+1, 6*i+3] += -M2 * kz * (n + delta)
        # L[6*i+1, 6*i+4] += -M2 * ky * kz
        # L[6*i+1, 6*i+5] += M2 * (ky**2.0 + (n + delta)**2.0)
        L[6*i+1, 6*i+5] += M2 * ky

        # w_n rows
        if n > ns[0]:
            L[6*i+2, 6*(i-1)] = 0.5j  # u_{n-1} column
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)] = 0.5j  # u_{n+1} column
        if n > ns[0]:
            L[6*i+2, 6*(i-1)+2] = -kz/2.0j  # w_{n-1} column
        if n < ns[-1]:
            L[6*i+2, 6*(i+1)+2] = kz/2.0j  # w_{n+1} column
        L[6*i+2, 6*i+2] = diss*1.0j*delta_n/Re  # w_n column
        # L[6*i+2, 6*i+3] = -kz  # p_n column
        pfac = kz / delta_n
        if n > ns[0]:
            # L[6*i+2, 6*(i-1)] += pfac * (kz/2.0j) * (n + delta + 1.0)
            # L[6*i+2, 6*(i-1)+1] += pfac * (kz/2.0j) * ky
            # L[6*i+2, 6*(i-1)+2] += pfac * (kz/2.0j) * kz
            L[6*i+2, 6*(i-1)] += pfac * -1.0j * kz
        if n < ns[-1]:
            # L[6*i+2, 6*(i+1)] += pfac * (kz/2.0j) * (1.0 - n - delta)
            # L[6*i+2, 6*(i+1)+1] += -pfac * (kz/2.0j) * ky
            # L[6*i+2, 6*(i+1)+2] += -pfac * (kz/2.0j) * kz
            L[6*i+2, 6*(i+1)] += pfac * -1.0j * kz
        # L[6*i+2, 6*i+3] += -M2 * kz * (n + delta)
        # L[6*i+2, 6*i+4] += -M2 * ky * kz
        # L[6*i+2, 6*i+5] += M2 * (ky**2.0 + (n + delta)**2.0)
        L[6*i+2, 6*i+5] += M2 * kz

        # b_{x,n} rows
        L[6*i+3, 6*i] = kz  # u_n column
        if n > ns[0]:
            L[6*i+3, 6*(i-1)+3] = -kz/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[6*i+3, 6*(i+1)+3] = kz/2.0j  # b_{x,n+1} column
        L[6*i+3, 6*i+3] = diss*1.0j*delta_n/Rm  # b_{x,n} column

        # b_{y,n} rows
        L[6*i+4, 6*i+1] = kz  # v_n column
        if n > ns[0]:
            L[6*i+4, 6*(i-1)+4] = -kz/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[6*i+4, 6*(i+1)+4] = kz/2.0j  # b_{y,n+1} column
        L[6*i+4, 6*i+4] = diss*1.0j*delta_n/Rm  # b_{y,n} column

        # b_{z,n} rows
        L[6*i+5, 6*i] = -(n + delta)  # u_n column
        L[6*i+5, 6*i+1] = -ky  # v_n column
        if n > ns[0]:
            L[6*i+5, 6*(i-1)+3] = (n + delta)/2.0j  # b_{x,n-1} column
        if n < ns[-1]:
            L[6*i+5, 6*(i+1)+3] = -(n + delta)/2.0j  # b_{x,n+1} column
        if n > ns[0]:
            L[6*i+5, 6*(i-1)+4] = ky/2.0j  # b_{y,n-1} column
        if n < ns[-1]:
            L[6*i+5, 6*(i+1)+4] = -ky/2.0j  # b_{y,n+1} column
        L[6*i+5, 6*i+5] = diss*1.0j*delta_n/Rm  # b_{z,n} column
    return L


def Lmat_checkerboard(delta_x, delta_y, Ax, Ay, M2, Re, Rm, kz, Nx, Ny, ideal=False):
    diss = 1.0 - ideal
    ns = list(range(-int((Nx - 1) / 2), int((Nx + 1) / 2), 1))
    ms = list(range(-int((Ny - 1) / 2), int((Ny + 1) / 2), 1))

    L = np.zeros((6*Nx*Ny, 6*Nx*Ny), dtype=np.complex128)
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            delta_nm = kz**2.0 + (n + delta_x)**2.0 + (m + delta_y)**2.0
            u_n_m = 6*j + 6*Ny*i  # the index corresponding to u_{n,m}
            u_nm1_m = 6*j + 6*Ny*(i-1)  # the index corresponding to u_{n-1,m}
            u_np1_m = 6*j + 6*Ny*(i+1)  # the index corresponding to u_{n+1,m}
            u_n_mm1 = 6*(j-1) + 6*Ny*i  # the index corresponding to u_{n,m-1}
            u_n_mp1 = 6*(j+1) + 6*Ny*i  # the index corresponding to u_{n,m+1}
            # start with the u_{n,m} rows
            if n > ns[0]:
                L[u_n_m, u_nm1_m] = -kz * Ax / 2.0j  # u_{n-1,m} column
            if n < ns[-1]:
                L[u_n_m, u_np1_m] = kz * Ax / 2.0j  # u_{n+1,m} column
            if m > ms[0]:
                L[u_n_m, u_n_mm1] = -kz * Ay / 2.0j  # u_{n,m-1} column
            if m < ms[-1]:
                L[u_n_m, u_n_mp1] = kz * Ay / 2.0j  # u_{n,m+1} column
            L[u_n_m, u_n_m + 3] = M2 * kz  # b_{x,n,m} column
            L[u_n_m, u_n_m + 5] = -M2 * (n + delta_x)  # b_{z,n,m} column
            L[u_n_m, u_n_m] = diss*delta_nm*1.0j/Re  # u_{n,m} column
            pfac = -(n + delta_x) / delta_nm  # now add in the contributions from pressure
            if n > ns[0]:
                L[u_n_m, u_nm1_m] += pfac * 1.0j * kz * Ax  # u_{n-1,m}
            if n < ns[-1]:
                L[u_n_m, u_np1_m] += pfac * 1.0j * kz * Ax  # u_{n+1,m}
            if m > ms[0]:
                L[u_n_m, u_n_mm1+1] += pfac * 1.0j * kz * Ay  # v_{n,m-1}
            if m < ms[-1]:
                L[u_n_m, u_n_mp1+1] += pfac * 1.0j * kz * Ay  # v_{n,m+1}
            L[u_n_m, u_n_m+5] -= pfac*M2*delta_nm  # b_{z,n,m}

            # now do v_{n,m} rows
            if n > ns[0]:
                L[u_n_m+1, u_nm1_m+1] = -kz * Ax / 2.0j  # v_{n-1,m} column
            if n < ns[-1]:
                L[u_n_m+1, u_np1_m+1] = kz * Ax / 2.0j  # v_{n+1,m} column
            if m > ms[0]:
                L[u_n_m+1, u_n_mm1+1] = -kz * Ay / 2.0j  # v_{n,m-1} column
            if m < ms[-1]:
                L[u_n_m+1, u_n_mp1+1] = kz * Ay / 2.0j  # v_{n,m+1} column
            L[u_n_m+1, u_n_m+4] = M2 * kz  # b_{y,n,m} column
            L[u_n_m+1, u_n_m+5] = -M2 * (m + delta_y)  # b_{z,n,m} column
            L[u_n_m+1, u_n_m+1] = diss*delta_nm*1.0j/Re  # v_{n,m} column
            pfac = -(m + delta_y) / delta_nm  # now add in the contributions from pressure
            if n > ns[0]:
                L[u_n_m+1, u_nm1_m] += pfac * 1.0j * kz * Ax  # u_{n-1,m}
            if n < ns[-1]:
                L[u_n_m+1, u_np1_m] += pfac * 1.0j * kz * Ax  # u_{n+1,m}
            if m > ms[0]:
                L[u_n_m+1, u_n_mm1+1] += pfac * 1.0j * kz * Ay  # v_{n,m-1}
            if m < ms[-1]:
                L[u_n_m+1, u_n_mp1+1] += pfac * 1.0j * kz * Ay  # v_{n,m+1}
            L[u_n_m+1, u_n_m+5] -= pfac*M2*delta_nm  # b_{z,n,m}

            # w_{n,m} rows
            if n > ns[0]:
                L[u_n_m+2, u_nm1_m+2] = -kz * Ax / 2.0j  # w_{n-1,m} column
            if n < ns[-1]:
                L[u_n_m+2, u_np1_m+2] = kz * Ax / 2.0j  # w_{n+1,m} column
            if m > ms[0]:
                L[u_n_m+2, u_n_mm1+2] = -kz * Ay / 2.0j  # w_{n,m-1} column
            if m < ms[-1]:
                L[u_n_m+2, u_n_mp1+2] = kz * Ay / 2.0j  # w_{n,m+1} column
            if n > ns[0]:
                L[u_n_m+2, u_nm1_m] = -Ax / 2.0j  # u_{n-1,m} column
            if n < ns[-1]:
                L[u_n_m+2, u_np1_m] = -Ax / 2.0j  # u_{n+1,m} column
            if m > ms[0]:
                L[u_n_m+2, u_n_mm1+1] = -Ay / 2.0j  # v_{n,m-1} column
            if m < ms[-1]:
                L[u_n_m+2, u_n_mp1+1] = -Ay / 2.0j  # v_{n,m-1} column
            L[u_n_m+2, u_n_m+2] = diss*delta_nm*1.0j/Re  # w_{n,m} column
            pfac = -kz / delta_nm  # now add in the contributions from pressure
            if n > ns[0]:
                L[u_n_m+2, u_nm1_m] += pfac * 1.0j * kz * Ax  # u_{n-1,m}
            if n < ns[-1]:
                L[u_n_m+2, u_np1_m] += pfac * 1.0j * kz * Ax  # u_{n+1,m}
            if m > ms[0]:
                L[u_n_m+2, u_n_mm1+1] += pfac * 1.0j * kz * Ay  # v_{n,m-1}
            if m < ms[-1]:
                L[u_n_m+2, u_n_mp1+1] += pfac * 1.0j * kz * Ay  # v_{n,m+1}
            L[u_n_m+2, u_n_m+5] -= pfac*M2*delta_nm  # b_{z,n,m}

            # b_{x,n,m} rows
            L[u_n_m+3, u_n_m] = kz  # u_{n,m} column
            if n > ns[0]:
                L[u_n_m+3, u_nm1_m+3] = -kz * Ax / 2.0j  # b_{x,n-1,m} column
            if n < ns[-1]:
                L[u_n_m+3, u_np1_m+3] = kz * Ax / 2.0j  # b_{x,n+1,m} column
            if m > ms[0]:
                L[u_n_m+3, u_n_mm1+3] = -kz * Ay / 2.0j  # b_{x,n,m-1} column
            if m < ms[-1]:
                L[u_n_m+3, u_n_mp1+3] = kz * Ay / 2.0j  # b_{x,n,m-1} column
            L[u_n_m+3, u_n_m+3] = diss * delta_nm * 1.0j / Rm  # b_{x,n,m} column

            # b_{y,n,m} rows
            L[u_n_m+4, u_n_m+1] = kz  # v_{n,m} column
            if n > ns[0]:
                L[u_n_m+4, u_nm1_m+4] = -kz * Ax / 2.0j  # b_{y,n-1,m} column
            if n < ns[-1]:
                L[u_n_m+4, u_np1_m+4] = kz * Ax / 2.0j  # b_{y,n+1,m} column
            if m > ms[0]:
                L[u_n_m+4, u_n_mm1+4] = -kz * Ay / 2.0j  # b_{y,n,m-1} column
            if m < ms[-1]:
                L[u_n_m+4, u_n_mp1+4] = kz * Ay / 2.0j  # b_{y,n,m-1} column
            L[u_n_m+4, u_n_m+4] = diss * delta_nm * 1.0j / Rm  # b_{y,n,m} column

            # b_{z,n,m} rows
            L[u_n_m+5, u_n_m] = -(n + delta_x)  # u_{n,m} column
            L[u_n_m+5, u_n_m+1] = -(m + delta_y)  # v_{n,m} column
            if n > ns[0]:
                L[u_n_m+5, u_nm1_m+3] = (n + delta_x) * Ax / 2.0j  # b_{x,n-1,m} column
                L[u_n_m+5, u_nm1_m+4] = (m + delta_y) * Ax / 2.0j  # b_{y,n-1,m} column
            if n < ns[-1]:
                L[u_n_m+5, u_np1_m+3] = -(n + delta_x) * Ax / 2.0j  # b_{x,n+1,m} column
                L[u_n_m+5, u_np1_m+4] = -(m + delta_y) * Ax / 2.0j  # b_{y,n+1,m} column
            if m > ms[0]:
                L[u_n_m+5, u_n_mm1+3] = (n + delta_x) * Ay / 2.0j  # b_{x,n,m-1} column
                L[u_n_m+5, u_n_mm1+4] = (m + delta_y) * Ay / 2.0j  # b_{y,n,m-1} column
            if m < ms[-1]:
                L[u_n_m+5, u_n_mp1+3] = -(n + delta_x) * Ay / 2.0j  # b_{x,n,m+1} column
                L[u_n_m+5, u_n_mp1+4] = -(m + delta_y) * Ay / 2.0j  # b_{y,n,m+1} column
            L[u_n_m+5, u_n_m+5] = diss * delta_nm * 1.0j / Rm
    return L


def gamfromL(L, withmode=False):
    w, v = np.linalg.eig(L)
    if withmode:
        ind = np.argmax(-np.imag(w))
        return [-np.imag(w[ind]), v[:, ind]]
    else:
        return np.max(-np.imag(w))


def stress_from_mode(phi_kx_shift, kz):
    N = len(phi_kx_shift)
    ns = list(range(-int((N - 1) / 2), int((N + 1) / 2), 1))
    dx_phi_kx_shift = phi_kx_shift * np.array(ns) * 1.0j
    phi_kx = np.fft.ifftshift(phi_kx_shift)
    dx_phi_kx = np.fft.ifftshift(dx_phi_kx_shift)
    phi_x = np.fft.ifft(phi_kx)
    dx_phi_x = np.fft.ifft(dx_phi_kx)
    tau = -2.0*np.real(dx_phi_x * -1.0j*kz*np.conj(phi_x))
    return tau


def omegafromL(L):
    w, v = np.linalg.eig(L)
    wsort = w[np.argsort(-np.imag(w))]
    return wsort[-1]


def gamfromparams(delta, M2, Re, Rm, k, N, ideal, withmode=False):
    # TODO: fully replace gamfromL with this
    L = Lmat(delta, M2, Re, Rm, k, N, ideal)
    return gamfromL(L, withmode)


def sigma_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N, stratified=True, withmode=False):
    """
    Returns fastest growing mode's growth rate for Kolmogorov flow problem, set up so that you can input parameters
    in PADDIM units (except for k_star) rather than normalized to the sinusoidal flow

    Parameters
    ----------
    delta : Floquet parameter, determines periodicity in x of eigenmodes relative to sinusoidal base flow -- set to 0
    w : Amplitude of sinusoidal base flow in PADDIM units
    HB : Lorentz force coefficient in PADDIM units
    DB : Magnetic diffusion coefficient in PADDIM units
    Pr : Thermal Prandtl number
    tau : compositional diffusion coefficient / thermal diffusion coefficient
    R0 : density ratio
    k_star : KH wavenumber *normalized to finger wavenumber* (this is confusing, in hindsight, given previous parames)
    N : Spectral resolution for calculating KH modes
    stratified : Boolean flag for whether or not to include T and C in calculation of KH modes
    withmode : Boolean flag for whether or not to return

    Returns
    -------
    sigma : (if stratified = True) growth rate of fastest-growing KH mode (at that wavenumber) in PADDIM units
    sigma^* : (if stratified = False) same, but in units normalized to the sinusoidal base flow -- TODO: fix this?
    """
    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
    if stratified:
        kz = k_star * lhat  # k_star is supposed to be kz normalized to finger width
        A_phi = w / lhat
        A_T = lhat * A_phi / (lamhat + l2hat)
        A_C = lhat * A_phi / (R0 * (lamhat + tau * l2hat))
        L = Lmat_finger_norm(HB, DB, Pr, R0, tau, A_phi, A_T, A_C, lhat, kz, delta, N)
        return gamfromL(L, withmode)
    else:
        M2, Re, Rm = KHparams_from_fingering(w, lhat, HB, Pr, DB)
        L = Lmat(delta, M2, Re, Rm, k_star, N)
        return gamfromL(L, withmode)


def omega_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N, stratified=True):
    lamhat, l2hat = fingering_modes.gaml2max(Pr, tau, R0)
    lhat = np.sqrt(l2hat)
    if stratified:
        kz = k_star * lhat  # k_star is supposed to be kz normalized to finger width
        A_phi = w / lhat
        A_T = lhat * A_phi / (lamhat + l2hat)
        A_C = lhat * A_phi / (R0 * (lamhat + tau * l2hat))
        L = Lmat_finger_norm(HB, DB, Pr, R0, tau, A_phi, A_T, A_C, lhat, kz, delta, N)
        return omegafromL(L)
    else:
        M2, Re, Rm = KHparams_from_fingering(w, lhat, HB, Pr, DB)
        L = Lmat(delta, M2, Re, Rm, k_star, N)
        return omegafromL(L)


def sigma_over_k_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_stars, N, stratified=True):
    return [sigma_from_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_star, N,
                                        stratified=stratified) for k_star in k_stars]


def gamma_over_k(delta, M2, Re, Rm, ks, N, ideal=False):
    return [gamfromL(Lmat(delta, M2, Re, Rm, k, N, ideal)) for k in ks]


def omega_over_k(delta, M2, Re, Rm, ks, N, ideal=False):
    return [omegafromL(Lmat(delta, M2, Re, Rm, k, N, ideal)) for k in ks]


def energy_from_streamfunc(field, k, kxs, xy_parts=False):
    """
    Calculates kinetic energy sum[ u**2 + w**2] from streamfunction phi
    (equivalent to magnetic energy from flux function, but need to multiply by HB^* AKA (Alfven Mach number)^-2).
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


def omega_over_k_3d(delta, M2, Re, Rm, kys, kzs, N, ideal=False):
    omegas = np.zeros((len(kys), len(kzs)), dtype=np.complex128)
    for kyi, ky in enumerate(kys):
        for kzi, kz in enumerate(kzs):
            L, M = Lmat_Mmat_3D(delta, M2, Re, Rm, ky, kz, N, ideal)
            w, v = scipy.linalg.eig(L, b=M, overwrite_a=True, overwrite_b=True)
            w_finite = w[np.isfinite(w)]
            omegas[kyi, kzi] = w_finite[np.argmax(-np.imag(w_finite))]
    return omegas


def omega_over_k_3d_nop(delta, M2, Re, Rm, kys, kzs, N, ideal=False):
    omegas = np.zeros((len(kys), len(kzs)), dtype=np.complex128)
    for kyi, ky in enumerate(kys):
        for kzi, kz in enumerate(kzs):
            L = Lmat_3D_noP3(delta, M2, Re, Rm, ky, kz, N, ideal)
            w, v = np.linalg.eig(L)
            w_finite = w[np.isfinite(w)]
            omegas[kyi, kzi] = w_finite[np.argmax(-np.imag(w_finite))]
    return omegas


def omega_over_k_checkerboard(delta_x, delta_y, Ax, Ay, M2, Re, Rm, kzs, Nx, Ny, ideal=False, sparse=False):
    omegas = np.zeros((len(kzs)), dtype=np.complex128)
    for kzi, kz in enumerate(kzs):
        L = Lmat_checkerboard(delta_x, delta_y, Ax, Ay, M2, Re, Rm, kz, Nx, Ny, ideal)
        if not sparse:
            w, v = np.linalg.eig(L)
        else:
            w, v = scipy.sparse.linalg.eigs(L, which='SI', k=1)
        w_finite = w[np.isfinite(w)]
        omegas[kzi] = w_finite[np.argmax(-np.imag(w_finite))]
    return omegas


def gammax_kscan(delta, M2, Re, Rm, ks, N, ideal=False, sparse=False, badks_except=False, get_kmax=False, threeD=False,
                 checkerboard=False, delta_x=0.0, delta_y=0.0, Ax=0.5, Ay=0.5, Nx=0, Ny=0):
    if not threeD and not checkerboard:
        gammas = gamma_over_k(delta, M2, Re, Rm, ks, N, ideal)
        ind = np.argmax(gammas)
        gammax = gammas[ind]
        if badks_except and gammax > 0.0:  # ASSUMING USER DOESN'T CARE ABOUT GAMMA_MAX IF IT'S NEGATIVE
            if gammax == gammas[0]:
                raise KrangeError  # ('the k-range needs to be extended downwards')
            if gammax == gammas[-1]:
                raise ValueError('the k-range needs to be extended upwards (or check your resolution)')
        if get_kmax:
            return [np.max(gammas), ks[ind]]
        else:
            return np.max(gammas)
    if threeD:
        gammas = -np.imag(omega_over_k_3d_nop(delta, M2, Re, Rm, ks, ks, N, ideal))
        inds = np.unravel_index(np.argmax(gammas), gammas.shape)
        if inds[0] > 0:
            print("Hey! Squire's theorem broke for M2, Re, Rm = {}, {}, {}".format(M2, Re, Rm))
            print("fastest-growing mode is at ky, kz = {}, {}".format(ks[inds[0]], ks[inds[1]]))
        gammax = gammas[inds[0], inds[1]]
        if badks_except and gammax > 0.0:  # ASSUMING USER DOESN'T CARE ABOUT GAMMA_MAX IF IT'S NEGATIVE
            if ks[inds[1]] == ks[0]:
                raise KrangeError  # ('the k-range needs to be extended downwards')
            if ks[inds[1]] == ks[-1]:
                raise ValueError('the k-range needs to be extended upwards (or check your resolution)')
        if get_kmax:
            return gammax, [ks[inds[0]], ks[inds[1]]]
        else:
            return gammax
    if checkerboard:
        if Nx == 0:
            nx = N
        else:
            nx = Nx
        if Ny == 0:
            ny = N
        else:
            ny = Ny
        gammas = -np.imag(omega_over_k_checkerboard(delta_x, delta_y, Ax, Ay, M2, Re, Rm, ks, nx, ny, ideal, sparse))
        ind = np.argmax(gammas)
        gammax = gammas[ind]
        if badks_except and gammax > 0.0:  # ASSUMING USER DOESN'T CARE ABOUT GAMMA_MAX IF IT'S NEGATIVE
            if ks[ind] == ks[0]:
                raise KrangeError  # ('the k-range needs to be extended downwards')
            if ks[ind] == ks[-1]:
                raise ValueError('the k-range needs to be extended upwards (or check your resolution)')
        if get_kmax:
            return [gammax, ks[ind]]
        else:
            return gammax


def sigma_max_kscan_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_stars, N, badks_except=False, get_kmax=False,
                                     stratified=True):
    sigmas = sigma_over_k_fingering_params(delta, w, HB, DB, Pr, tau, R0, k_stars, N, stratified=stratified)
    ind = np.argmax(sigmas)
    sigma_max = sigmas[ind]
    if badks_except and sigma_max > 0.0:  # ASSUMING USER DOESN'T CARE ABOUT GAMMA_MAX IF IT'S NEGATIVE
        if sigma_max == sigmas[0]:  # I don't remember why I separated them like this
            print(w, sigma_max)
            raise KrangeError  # ('the k-range needs to be extended downwards')
        if sigma_max == sigmas[-1]:
            raise ValueError('the k-range needs to be extended upwards (or check your resolution)')
    if get_kmax:
        return [sigmas[ind], k_stars[ind]]
    else:
        return sigmas[ind]


def gammax_minus_lambda(w, lamhat, lhat, HB, Pr, DB, delta, ks, N, ideal=False, sparse=False, badks_exception=False, CH=1.66,
                        threeD=False, checkerboard=False, delta_x=0.0, delta_y=0.0, Ax=0.5, Ay=0.5, Nx=0, Ny=0,
                        stratified=False, tau=-1, R0=0):
    # a silly helper function that returns sigma - lambda rather than sigma
    # so that I can use root-finding packages to search for zeros of this
    # function

    # NOTE THIS MULTIPLIES sigma BY w_f l_f
    # (or rather -- gammax_kscan returns what HG19 calls sigma/(w_f l_f),
    # and so this quantity is multiplied by w_f l_f to get their sigma)
    #
    # NOTE THAT THE INPUT CH REFERS TO THE CH YOU SEE
    # IN EQ 31 IN HARRINGTON & GARAUD 2019

    if not stratified:
        M2, Re, Rm = KHparams_from_fingering(w, lhat, HB, Pr, DB)
    krange = np.copy(ks)

    # The following while/try/except is for repeating the k scan if the max
    # occurs at the edges of the range of k sampled
    count = 0
    while True:
        count += 1
        try:
            if not stratified:
                out = gammax_kscan(delta, M2, Re, Rm, krange, N, ideal, sparse, badks_exception, threeD=threeD,
                                   checkerboard=checkerboard, delta_x=delta_x, delta_y=delta_y, Ax=Ax, Ay=Ay, Nx=Nx,
                                   Ny=Ny) * w * lhat - CH * lamhat
            else:
                if tau == -1 or R0 == 0:
                    raise ValueError("Need to pass tau to gammax_minus_lambda")
                # Note: no need to multiply by w*lhat, sigma should already be in PADDIM units
                out = sigma_max_kscan_fingering_params(delta, w, HB, DB, Pr, tau, R0, krange, N,
                                                       badks_except=badks_exception) - CH * lamhat
            break
        # except KrangeError:  # Only happens if badks_exception is True
            # krange = np.linspace(0.5 * krange[0], 1.5 * krange[0], num=len(krange))
            # print("modifying ks", count)
            # if count > 4:
                # raise
        except ValueError:
            # the max occurs at the upper end of ks so seldomly
            # that I never bothered to implement this part
            print("w = ", w)  # for debugging
            raise
    return out
