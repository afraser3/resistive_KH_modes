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

"""
# TODO: rename KH growth rate from gamma to sigma for consistency w/ HG18
import numpy as np
import scipy


class KrangeError(Exception):
    pass


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


def gammax_kscan(delta, M2, Re, Rm, ks, N, ideal=False, badks_except=False, get_kmax=False,):
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
