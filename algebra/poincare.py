from lie_learn.representations.SO3.pinchon_hoggan.pinchon_hoggan_dense import Jd, rot_mat
import numpy as np
import scipy.linalg
import cvxpy as cvx
import itertools
from scipy.linalg import inv
from copy import copy
from tqdm.auto import tqdm


def boost(nx, ny, nz, beta):
    gamma = 1/np.sqrt(1-beta**2)
    return np.array([
        [gamma, -gamma*beta*nx, -gamma*beta*ny, -gamma*beta*nz],
        [-gamma*beta*nx, 1+(gamma-1)*nx**2, (gamma-1)*nx*ny, (gamma-1)*nx*nz],
        [-gamma*beta*ny, (gamma-1)*ny*nx, 1+(gamma-1)*ny**2, (gamma-1)*ny*nz],
        [-gamma*beta*nz, (gamma-1)*nz*nx, (gamma-1)*nz*ny, 1+(gamma-1)*nz**2]
    ])

def random_rotation():
    # print('generate random rotation')
    α, β, 𝜸 = np.random.randn(3)
    # print('α, β, 𝜸 = {}'.format((α, β, 𝜸)))
    R = rot_mat(alpha=α, beta=β, gamma=𝜸, J=Jd[1], l=1)
    return R

def random_group_element(vmax=1):
    print('generate random rotation')
    α, β, 𝜸 = np.random.randn(3)
    print('α, β, 𝜸 = {}'.format((α, β, 𝜸)))
    R = scipy.linalg.block_diag(
        1,
        rot_mat(alpha=α, beta=β, gamma=𝜸, J=Jd[1], l=1)
    )
    # print(R)

    print('generate random boost')
    v = np.random.randn(3)
    n = v / np.sqrt(np.sum(v**2))
    print('v = {}'.format(v))
    B = boost(*n, min(vmax, np.tanh(np.sqrt(np.sum(v**2)))))
    # print(B)
    L = np.matmul(R, B)

    print('generate random translation')
    b = np.random.randn(4)

    print('Poincaré Group Element')
    print('x^μ -> L^μν x^ν + b^μ')
    print('L^μν = {}\nb^μ = {}'.format(np.round(L, 3), np.round(b, 3)))
    return L, b

#
# def spin_matrices(n):
#     N = int(2*n+1)
#     a = np.arange(-n, +n+1, 1)
#     A = np.arange(0, N)
#     J1, J2, J3 = np.zeros((N, N), 'complex'), np.zeros((N, N), 'complex'), np.zeros((N, N), 'complex')
#
#     J1[A[:-1], A[:-1]+1] = 1/2 * np.sqrt((n-a[:-1])*(n+a[:-1]+1))
#     J1[A[1:], A[1:]-1] = 1/2 * np.sqrt((n+a[1:])*(n-a[1:]+1))
#
#     J2[A[:-1], A[:-1]+1] = 1/(2j) * np.sqrt((n-a[:-1])*(n+a[:-1]+1))
#     J2[A[1:], A[1:]-1] = -1/(2j) * np.sqrt((n+a[1:])*(n-a[1:]+1))
#
#     # may need to swap order, see
#     # http://easyspin.org/easyspin/documentation/spinoperators.html
#     J3[A, A] = -a
#     return (J1, J2, J3)

import qutip as qt
def spin_matrices(n):
    return np.stack([np.array(J.data.todense()) for J in qt.jmat(n)], axis=0)


# Jax, Jay, Jaz = qt.jmat(a)
# Jbx, Jby, Jbz = qt.jmat(b)

def pi(m, n):
    # eq. 5.6.14-15 Weinberg Volume I Foundations
    A1, A2, A3 = (
        np.einsum('pb,ea->epab',
            np.eye(int(2*n+1)), J/2.0
        ).reshape(int(2*m+1)*int(2*n+1), int(2*m+1)*int(2*n+1))
        for J in spin_matrices(m)
    )
    B1, B2, B3 = (
        np.einsum('pb,ea->peba',
            np.eye(int(2*m+1)), J.conj()/2.0
        ).reshape(int(2*n+1)*int(2*m+1), int(2*n+1)*int(2*m+1))
        for J in spin_matrices(n)
    )
    return ((A1, A2, A3), (B1, B2, B3))


def irrep_lie_algebra_gens_so31(m, n):
    sigma_ax, sigma_ay, sigma_az = (-1j * sigma for sigma in spin_matrices(m))
    sigma_bx, sigma_by, sigma_bz = (-1j * sigma for sigma in spin_matrices(n))

    dim_a = int(2*(m) + 1)
    dim_b = int(2*(n) + 1)
    rep_dim = dim_a*dim_b
    A1 = np.einsum('ae,bp->abep', sigma_ax, np.eye(dim_b)).reshape(rep_dim,rep_dim)
    A2 = np.einsum('ae,bp->abep', sigma_ay, np.eye(dim_b)).reshape(rep_dim,rep_dim)
    A3 = np.einsum('ae,bp->abep', sigma_az, np.eye(dim_b)).reshape(rep_dim,rep_dim)
    A = np.array([A1, A2, A3])

    B1 = np.einsum('ae,bp->abep', np.eye(dim_a), sigma_bx).reshape(rep_dim,rep_dim).conj()
    B2 = np.einsum('ae,bp->abep', np.eye(dim_a), sigma_by).reshape(rep_dim,rep_dim).conj()
    B3 = np.einsum('ae,bp->abep', np.eye(dim_a), sigma_bz).reshape(rep_dim,rep_dim).conj()
    B = np.array([B1, B2, B3])

    J = (A+B)
    K = (A-B)/-1j

    return J, K


four_repr = (
    np.array([
        [
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,-1],
            [0,0,1,0]
        ],
        [
            [0,0,0,0],
            [0,0,0,1],
            [0,0,0,0],
            [0,-1,0,0]
        ],
        [
            [0,0,0,0],
            [0,0,-1,0],
            [0,1,0,0],
            [0,0,0,0]
        ]
    ]),
    np.array([
        [
            [0,1,0,0],
            [1,0,0,0],
            [0,0,0,0],
            [0,0,0,0]
        ],
        [
            [0,0,1,0],
            [0,0,0,0],
            [1,0,0,0],
            [0,0,0,0]
        ],
        [
            [0,0,0,1],
            [0,0,0,0],
            [0,0,0,0],
            [1,0,0,0]
        ]
    ])
)


three_repr_2d = (
    np.array([
        [
            [0,0,0,],
            [0,0,-1,],
            [0,1,0,]
        ],
        [
            [0,1,0,],
            [1,0,0,],
            [0,0,0,]
        ],
        [
            [0,0,1,],
            [0,0,0,],
            [1,0,0,]
        ]
    ])
)
