"""Various functions for working with fingering instability

Provides tools for things like calculating fingering growth rate

Imports
-------
Some standard numpy & scipy stuff, as well as:
kolmogorov_EVP: a module for solving the KH eigenvalue problem for the
        parasites on top of the fingering modes

Methods
-------
rfromR(R0, tau)
    Calculates the reduced density ratio r
lamguess(Pr, tau, R0)
    Provides an initial guess for the most-unstable fingering growth rate

    (lam is short for lambda, which can't be used as a variable name
    in python)

k2guess(Pr, tau, R0)
    Provides initial guess for corresponding wavenumber (squared)
eq1(lam, k2, Pr, tau, R0)
    Evaluates the characteristic polynomial for fingering instability,
    returns 0 iff lam, k2 are a valid growth rate/wavenumber^2 pair
eq2(lam, k2, Pr, tau, R0)
    Evaluates to 0 iff d(lam)/d(k2) = 0
fun(x, Pr, tau, R0, passk1=False)
    A vector equivalent of eq1 and eq2. If x = [lam, k2], fun
    just returns [eq1, eq2]
jac(x, Pr, tau, R0, passk1=False)
    The jacobian matrix for fun wrt lam and k2
gaml2max(Pr, tau, R0)
    Uses scipy.optimize's rootsolving tools to find the roots x=[lam,k2]
    of [eq1, eq2] and returns them -- i.e., returns the most-unstable
    growth rate of the fingering instability and the corresponding
    wavenumber squared

Beyond gaml2max there's some junk I wrote as an aside that isn't important
"""
import numpy as np
from scipy import linalg
from scipy import optimize as opt


# import kolmogorov_EVP

def rfromR(R0, tau):
    return (R0 - 1.0) / (-1.0 + 1.0 / tau)


def lamguess(Pr, tau, R0):
    r = rfromR(R0, tau)
    if r < tau:
        return np.sqrt(Pr) - Pr * np.sqrt(1.0 + tau / Pr)
    else:
        if r > 0.5:
            return 2.0 * Pr * (tau / Pr) * ((1.0 / 3.0) * (1.0 - r)) ** (3.0 / 2.0) / (
                    1.0 - (1.0 - r) * (1.0 + tau / Pr) / 3.0)
        else:
            return np.sqrt(Pr * tau / r) - Pr * np.sqrt(1 + tau / Pr)


def k2guess(Pr, tau, R0):
    r = rfromR(R0, tau)
    if r < tau:
        return (1.0 + tau / Pr) ** (-0.5) - np.sqrt(Pr) * (1.0 + (tau / Pr) * (1.0 + tau / Pr) ** (-2.0))
    else:
        if r > 0.5:
            return np.sqrt((1.0 - r) / 3.0)
        else:
            return np.sqrt((1.0 + tau / Pr) ** (-0.5) - 2.0 * np.sqrt(r * tau / Pr) * (1.0 + tau / Pr) ** (-5.0 / 2.0))


def eq1(lam, k2, Pr, tau, R0):
    b2 = k2 * (1.0 + Pr + tau)
    b1 = k2 ** 2.0 * (tau * Pr + Pr + tau) + Pr * (1.0 - 1.0 / R0)
    b0 = k2 ** 3.0 * tau * Pr + k2 * Pr * (tau - 1.0 / R0)
    return lam ** 3.0 + b2 * lam ** 2.0 + b1 * lam + b0


def eq2(lam, k2, Pr, tau, R0):
    c2 = 1.0 + Pr + tau
    c1 = 2.0 * k2 * (tau * Pr + tau + Pr)
    c0 = 3.0 * k2 ** 2.0 * tau * Pr + Pr * (tau - 1.0 / R0)
    return c2 * lam ** 2.0 + c1 * lam + c0


def fun(x, Pr, tau, R0, passk1=False):  # returns f(x) where f = [eq1, eq2] and x = [lam, k2]
    if passk1:  # if x[1] is k instead of k^2
        return [eq1(x[0], x[1] ** 2.0, Pr, tau, R0), eq2(x[0], x[1] ** 2.0, Pr, tau, R0)]
    else:
        return [eq1(x[0], x[1], Pr, tau, R0), eq2(x[0], x[1], Pr, tau, R0)]


def jac(x, Pr, tau, R0, passk1=False):  # jacobian of fun(x)
    lam = x[0]
    if passk1:  # is x[1] k or k^2?
        k2 = x[1] ** 2.0
    else:
        k2 = x[1]
    b2 = k2 * (1.0 + Pr + tau)
    db2dk2 = 1.0 + Pr + tau  # derivative of b2 wrt k2
    b1 = k2 ** 2.0 * (tau * Pr + Pr + tau) + Pr * (1.0 - 1.0 / R0)
    db1dk2 = 2.0 * k2 * (tau * Pr + Pr + tau)
    b0 = k2 ** 3.0 * tau * Pr + k2 * Pr * (tau - 1.0 / R0)
    db0dk2 = 3.0 * k2 ** 2.0 * tau * Pr + Pr * (tau - 1.0 / R0)

    j11 = 3.0 * lam ** 2.0 + 2.0 * b2 * lam + b1  # d(eq1)/dlam
    j12 = lam ** 2.0 * db2dk2 + lam * db1dk2 + db0dk2  # d(eq1)/dk2
    if passk1:
        j12 = j12 * 2.0 * x[1]  # d(eq1)/dk = d(eq1)/dk2 * dk2/dk

    c2 = 1.0 + Pr + tau
    c1 = 2.0 * k2 * (tau * Pr + tau + Pr)
    dc1dk2 = c1 / k2
    c0 = 3.0 * k2 ** 2.0 * tau * Pr + Pr * (tau - 1.0 / R0)
    dc0dk2 = 6.0 * k2 * tau * Pr

    j21 = 2.0 * c2 * lam + c1
    j22 = lam * dc1dk2 + dc0dk2
    if passk1:
        j22 = j12 * 2.0 * x[1]
    return [[j11, j12], [j21, j22]]


def gaml2max(Pr, tau, R0):  # uses scipy.optimize.root with the above functions to find lambda_FGM and l^2_FGM
    sol = opt.root(fun, [lamguess(Pr, tau, R0), k2guess(Pr, tau, R0)], args=(Pr, tau, R0), jac=jac, method='hybr')
    x = sol.x
    if sol.x[1] < 0:  # if a negative k^2 is returned, then try again but solve for k instead of k^2
        sol = opt.root(fun, [lamguess(Pr, tau, R0), np.sqrt(k2guess(Pr, tau, R0))], args=(Pr, tau, R0, True), jac=jac,
                       method='hybr')
        test = fun(sol.x, Pr, tau, R0, True)
        if np.allclose(test, np.zeros_like(test)) == False:
            raise ValueError("fingering_modes.gaml2max is broken!")
        x = sol.x
        x[1] = x[1] ** 2.0  # whatever calls gaml2max expects k^2, not k
    # sol = opt.root(fun, [lamguess(Pr, tau, R0), 4.0*k2guess(Pr, tau, R0)], args=(Pr, tau, R0), jac=jac, method='hybr')
    # if sol.x[1]<0:
    # raise ValueError("fingering_modes.gaml2max settled on a negative l2!")
    # return sol.x
    return x


###################################
###################################


def Bz_disp_roots(HB, Pr, tau, R0, DB, kx, kz, max_real=True):  # roots of dispersion relation for vertical uniform B_0
    k2 = kx**2.0 + kz**2.0
    alpha = Pr*kx**2.0 / (R0 * k2)
    a4 = 1
    a3 = (Pr + DB + tau + 1)*k2
    a2 = k2**2.0 * (tau + (tau + 1)*(Pr + DB) + Pr*DB) + kz**2.0*HB - alpha * (1 - R0)
    a1 = (tau + 1)*k2*(Pr*DB*k2**2.0 + kz**2.0*HB) + (Pr + DB)*tau*k2**3.0 - alpha*(1 - R0*tau + DB*(1-R0))*k2
    a0 = tau*k2**2.0*(Pr*DB*k2**2.0 + kz**2.0*HB) - alpha*(1 - R0*tau)*DB*k2**2.0
    roots = np.roots([a4, a3, a2, a1, a0])
    if max_real:
        ind = np.argmax(np.real(roots))
        return roots[ind]
    else:
        return roots


def Bz_disp_roots_kxscan(HB, Pr, tau, R0, DB, kxs, kz):  # do a kx scan, return the max
    return np.max(np.array([Bz_disp_roots(HB, Pr, tau, R0, DB, kx, kz) for kx in kxs]))


def stabilizing_HB(Pr, tau, R0, DB, kxs, kz, HBmax=10.0**5.0):  # given these parameters, which HB gives lambda=0?
    # could probably calculate d(lambda)/d(HB) to use NR if we wanted to
    if kz == 0.0:
        raise ValueError("HB doesn't stabilize kz=0 modes.")
    if R0 < 1.0 or R0 > 1.0/tau:
        raise ValueError("For instability, need 1 < R0 < 1/tau")
    if tau > 1.0:
        raise ValueError("Script written assuming tau < 1")
    sol = opt.root_scalar(Bz_disp_roots_kxscan, args=(Pr, tau, R0, DB, kxs, kz), bracket=[0.0, HBmax])
    return sol.root


def Mmat():
    M = np.identity(9, dtype=np.complex128)
    M[3, 3] = 0.0
    return M


def Lmat(kx, ky, kz, Pr, tau, R0, HB, DB):  # the B_0 = \hat{x} case first
    k = np.array([kx, ky, kz])
    k2 = np.dot(k, k)  # kx**2.0 + ky**2.0 + kz**2.0
    L = np.zeros_like(Mmat())
    for i in range(3):
        L[i, i] = k2 * Pr
        L[i, 3] = 1.0j * k[i]
        L[3, i] = k[i]
        L[-1 - i, -1 - i] = DB * k2
    L[1, 6] = 1.0j * HB * ky
    L[1, 7] = -1.0j * HB * kx
    L[2, 4] = -Pr
    L[2, 5] = Pr
    L[2, 6] = 1.0j * HB * kz
    L[2, 8] = -1.0j * HB * kx
    L[4, 2] = 1
    L[4, 4] = k2
    L[5, 2] = 1.0 / R0
    L[5, 5] = k2 * tau
    L[6, 0] = -1.0j * kx
    L[7, 1] = -1.0j * kx
    L[8, 2] = -1.0j * kx
    return -L
