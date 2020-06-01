import numpy as np
from tqdm.auto import tqdm
import scipy.linalg
import itertools
from functools import reduce


def cg_n_ary(
    gens_list,
    dims_list,
    gens_out,
    dim_out
):
    num_params = dim_out * reduce(lambda x, y: x*y, dims_list+[dim_out])
    for *i_in_arr, k in itertools.product(*([range(d) for d in dims_list] + [range(dim_out)])):
        print(i_in_arr+[k])


def clebsch_gordan(gens_1, gens_2, gens_p, num_examples=None, zero=1e-3):

    dim_1 = gens_1[0].shape[0]
    dim_2 = gens_2[0].shape[0]
    dim_p = gens_p[0].shape[0]

    assert gens_1.shape[0] == gens_2.shape[0]
    assert gens_2.shape[0] == gens_p.shape[0]
    num_gens = gens_1.shape[0]

    if num_examples is None:
        # determine sufficient num examples to overdetermine system
        num_examples = dim_1*dim_2 + 24

    print('sample constraints')
    A = []
    for ex in tqdm(range(num_examples)):
        for (u, v), alpha in itertools.product(
            [
                (
                    # np.random.normal(size=(dim_1, 2)).view(np.complex128).reshape(dim_1),
                    # np.random.normal(size=(dim_2, 2)).view(np.complex128).reshape(dim_2)
                    np.random.normal(size=(dim_1)),
                    np.random.normal(size=(dim_2))
                ) for _ in range(1)
            ],
            [
                np.random.normal(size=(num_gens,2)).view(np.complex128).reshape(num_gens,1,1)
                for _ in range(1)
            ]
        ):

            rho_g_1 = scipy.linalg.expm((
                alpha * gens_1
            ).sum(axis=0))
            rho_g_2 = scipy.linalg.expm((
                alpha * gens_2
            ).sum(axis=0))
            rho_g_p = scipy.linalg.expm((
                alpha * gens_p
            ).sum(axis=0))

            for d in range(dim_p):
                l_coeffs = np.zeros((dim_1, dim_2, dim_p), dtype='complex128')
                r_coeffs = np.zeros((dim_1, dim_2, dim_p), dtype='complex128')
                for s, t, r, q in itertools.product(
                    range(dim_1),
                    range(dim_1),
                    range(dim_2),
                    range(dim_2)
                ):
                    l_coeffs[s,r,d] += rho_g_1[s][t] * u[t] * rho_g_2[r][q] * v[q]

                for g, h, f in itertools.product(
                    range(dim_1),
                    range(dim_2),
                    range(dim_p)
                ):
                    r_coeffs[g,h,f] += rho_g_p[d][f] * u[g] * v[h]

                A.append((l_coeffs - r_coeffs).reshape(-1))


    # print('convert to matrix form')
    # b = []
    # LmR = []
    # for constraint in tqdm(list(prob.constraints.values())):
    #     b.append(constraint.constant)
    #     LmR.append([
    #         constraint[prob.variablesDict()['C_{}'.format(str((i,j,k)).replace(' ', '_'))]]
    #         for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_p))
    #     ])
    # A = np.array(LmR)
    A = np.array(A)
    print(f'A shape = {A.shape}')
    print('solve nullspace')
    # ns = scipy.linalg.null_space(A, rcond=5e-11)
    P, D, Q  = np.linalg.svd(A)
    print(P.shape, D.shape, Q.shape)
    # ns = Q[-1:].T
    print(f'smallest 10 singular values: {D.round(6)[-10:]}')
    print(f'cutoff {zero}')
    ns = Q[np.abs(D) <= zero].T.conj()
    print(f"nullspace dimension {ns.shape[1]}")
    # there should be exactly 1D nullspace
    if not ns.shape[1]:
        print('No coefficients found: nullspace basis shape {}'.format(ns.shape))
        return 0

    # # flatten nullvector -- this contains the correct CGs,
    # # which are arbitrary up to a phase
    # ns = ns.reshape(-1)

    # remove phase information / normalize
    for j in range(ns.shape[1]):
        ns[:,j] /= ns[:,j][np.argmax(np.abs(ns[:,j]))]

    print('reshape')
    # C = np.zeros((dim_1, dim_2, dim_p), dtype='complex128')
    # for ii, (i, j, k) in enumerate(itertools.product(range(dim_1), range(dim_2), range(dim_p))):
    #     C[i,j,k] = ns[ii]

    # I do not trust interoperability of np reshape and itertools product

    # print(ns.reshape(dim_1, dim_2, dim_p) - C)
    # print(ns.reshape(dim_1, dim_2, dim_p) == C)
    # assert ns.reshape(dim_1, dim_2, dim_p) == C
    Cs = np.swapaxes(ns, 0, 1).reshape(ns.shape[1], dim_1, dim_2, dim_p)

    return Cs


def clebsch_gordan_r(r1, r2, rp, num_examples=None, zero=1e-3):

    dim_1 = r1.dim
    dim_2 = r2.dim
    dim_p = rp.dim

    assert r1.num_gens == r2.num_gens
    assert r2.num_gens == rp.num_gens
    num_gens = r1.num_gens

    if num_examples is None:
        # determine sufficient num examples to overdetermine system
        num_examples = dim_1*dim_2 + 20

    print('sample constraints')
    A = []
    for ex in tqdm(range(num_examples)):
        for (u, v), alpha in itertools.product(
            [
                (
                    np.random.normal(size=(dim_1)),
                    np.random.normal(size=(dim_2))
                ) for _ in range(1)
            ],
            [
                np.random.normal(size=(num_gens,2)).view(np.complex128).reshape(num_gens,1,1)
                for _ in range(1)
            ]
        ):

            rho_g_1 = r1.matrix_expm_np(alpha)
            rho_g_2 = r2.matrix_expm_np(alpha)
            rho_g_p = rp.matrix_expm_np(alpha)

            for d in range(dim_p):
                l_coeffs = np.zeros((dim_1, dim_2, dim_p), dtype='complex128')
                r_coeffs = np.zeros((dim_1, dim_2, dim_p), dtype='complex128')
                for s, t, r, q in itertools.product(
                    range(dim_1),
                    range(dim_1),
                    range(dim_2),
                    range(dim_2)
                ):
                    l_coeffs[s,r,d] += rho_g_1[s][t] * u[t] * rho_g_2[r][q] * v[q]

                for g, h, f in itertools.product(
                    range(dim_1),
                    range(dim_2),
                    range(dim_p)
                ):
                    r_coeffs[g,h,f] += rho_g_p[d][f] * u[g] * v[h]

                A.append((l_coeffs - r_coeffs).reshape(-1))

    A = np.array(A)
    print(f'A shape = {A.shape}')
    print('solve nullspace')
    P, D, Q  = np.linalg.svd(A)
    print(P.shape, D.shape, Q.shape)

    print(f'smallest 10 singular values: {D.round(6)[-10:]}')
    print(f'cutoff {zero}')
    ns = Q[np.abs(D) <= zero].T.conj()
    print(f"nullspace dimension {ns.shape[1]}")
    if not ns.shape[1]:
        print('No coefficients found: nullspace basis shape {}'.format(ns.shape))
        return 0

    # remove phase information / normalize
    for j in range(ns.shape[1]):
        ns[:,j] /= ns[:,j][np.argmax(np.abs(ns[:,j]))]

    Cs = np.swapaxes(ns, 0, 1).reshape(ns.shape[1], dim_1, dim_2, dim_p)

    return Cs


def test_cg_r(C, r1, r2, rp, num_examples=2):
    dim_1 = r1.dim
    dim_2 = r2.dim
    dim_p = rp.dim

    assert r1.num_gens == r2.num_gens
    assert r2.num_gens == rp.num_gens
    num_gens = r1.num_gens

    error = 0
    for ex in range(num_examples):
        # alpha = np.random.randn(num_gens).reshape(num_gens,1,1)
        alpha = np.random.normal(size=(num_gens,2)).view(np.complex128).reshape(num_gens,1,1)
        u = np.random.randn(dim_1)
        v = np.random.randn(dim_2)

        # rho_g_1 = scipy.linalg.expm((
        #     alpha * gens_1
        # ).sum(axis=0))
        # rho_g_2 = scipy.linalg.expm((
        #     alpha * gens_2
        # ).sum(axis=0))
        # rho_g_p = scipy.linalg.expm((
        #     alpha * gens_p
        # ).sum(axis=0))

        rho_g_1 = r1.matrix_expm_np(alpha)
        rho_g_2 = r2.matrix_expm_np(alpha)
        rho_g_p = rp.matrix_expm_np(alpha)

        left_side_sums = [
            sum(
                C[s][r][d] * rho_g_1[s][t] * u[t] * rho_g_2[r][q] * v[q]
                for s, t, r, q in itertools.product(
                    range(dim_1),
                    range(dim_1),
                    range(dim_2),
                    range(dim_2)
                )
            )
            for d in range(dim_p)
        ]

        right_side_sums = [
            sum(
                rho_g_p[d][f] * C[g][h][f] * u[g] * v[h]
                for g, h, f in itertools.product(
                    range(dim_1),
                    range(dim_2),
                    range(dim_p)
                )
            )
            for d in range(dim_p)
        ]

        error += sum(
            np.sum(np.abs(l - r))
            for l, r in zip(left_side_sums, right_side_sums)
        )
    return error / num_examples


def test_cg(C, gens_1, gens_2, gens_p, num_examples=2):
    dim_1 = gens_1[0].shape[0]
    dim_2 = gens_2[0].shape[0]
    dim_p = gens_p[0].shape[0]

    assert gens_1.shape[0] == gens_2.shape[0]
    assert gens_2.shape[0] == gens_p.shape[0]
    num_gens = gens_1.shape[0]
    #
    # J_1, K_1 = irrep_lie_algebra_gens_so31(m_1, n_1)
    # J_2, K_2 = irrep_lie_algebra_gens_so31(m_2, n_2)
    # J_p, K_p = irrep_lie_algebra_gens_so31(m_p, n_p)

    error = 0
    for ex in range(num_examples):
        # alpha = np.random.randn(num_gens).reshape(num_gens,1,1)
        alpha = np.random.normal(size=(num_gens,2)).view(np.complex128).reshape(num_gens,1,1)
        u = np.random.randn(dim_1)
        v = np.random.randn(dim_2)

        rho_g_1 = scipy.linalg.expm((
            alpha * gens_1
        ).sum(axis=0))
        rho_g_2 = scipy.linalg.expm((
            alpha * gens_2
        ).sum(axis=0))
        rho_g_p = scipy.linalg.expm((
            alpha * gens_p
        ).sum(axis=0))

        left_side_sums = [
            sum(
                C[s][r][d] * rho_g_1[s][t] * u[t] * rho_g_2[r][q] * v[q]
                for s, t, r, q in itertools.product(
                    range(dim_1),
                    range(dim_1),
                    range(dim_2),
                    range(dim_2)
                )
            )
            for d in range(dim_p)
        ]

        right_side_sums = [
            sum(
                rho_g_p[d][f] * C[g][h][f] * u[g] * v[h]
                for g, h, f in itertools.product(
                    range(dim_1),
                    range(dim_2),
                    range(dim_p)
                )
            )
            for d in range(dim_p)
        ]

        error += sum(
            np.sum(np.abs(l - r))
            for l, r in zip(left_side_sums, right_side_sums)
        )
    return error / num_examples


def clebsch_gordan_table_lorentz(
    m_n_array=[(0,0), (1/2,1/2), (1,1)]#, (3/2,3/2)] # last must be highest-dimensional
):
    C = np.zeros(
        3*(len(m_n_array),) + 3*(int((2*m_n_array[-1][0] + 1) * (2*m_n_array[-1][1] + 1)),),
        dtype='complex128'
    )
    for (ri1, (m_1, n_1)), (ri2, (m_2, n_2)), (ri3, (m_3, n_3)) in itertools.product(
        enumerate(m_n_array),
        enumerate(m_n_array),
        enumerate(m_n_array)
    ):
        print('{} ⓧ {} → {}'.format((m_1, n_1), (m_2, n_2), (m_3, n_3)))
        # if m_1+n_1+m_2+n_2+m_3+n_3 >= 5:
        #     print('skipping due to problem size')
        #     continue
        C[ri1,ri2,ri3,:int((2*m_1+1)*(2*n_1+1)),:int((2*m_2+1)*(2*n_2+1)),:int((2*m_3+1)*(2*n_3+1))] = connection_np(
            m_1, n_1, m_2, n_2, m_3, n_3
        )
        print('testing connection')
        error = test_cg(
            C[ri1,ri2,ri3,:int((2*m_1+1)*(2*n_1+1)),:int((2*m_2+1)*(2*n_2+1)),:int((2*m_3+1)*(2*n_3+1))],
            m_1, n_1, m_2, n_2, m_3, n_3,
            num_examples=13
        )
        assert error < 1e-6, error
        print('error = {}'.format(error))
    return C


def complex_noise(shape):
    return np.random.normal(
        size=shape + (2,)
    ).view(np.complex128).reshape(shape)

def clebsch_sv_ratio(gens_1, gens_2, gens_p, num_examples=None, sv_small=-1, sv_large=-2, zero=1e-3):
    # np.random.seed(123)
    dim_1 = gens_1[0].shape[0]
    dim_2 = gens_2[0].shape[0]
    dim_p = gens_p[0].shape[0]

    assert gens_1.shape[0] == gens_2.shape[0]
    assert gens_2.shape[0] == gens_p.shape[0]
    num_gens = gens_1.shape[0]

    if num_examples is None:
        # determine sufficient num examples to overdetermine system
        num_examples = dim_1*dim_2 + 10

    print('sample constraints')
    A = []
    for ex in tqdm(range(num_examples)):
        for (u, v), alpha in itertools.product(
            [
                (
                    # np.random.normal(size=(dim_1, 2)).view(np.complex128).reshape(dim_1),
                    # np.random.normal(size=(dim_2, 2)).view(np.complex128).reshape(dim_2)
                    np.random.normal(size=(dim_1)),
                    np.random.normal(size=(dim_2))
                ) for _ in range(1)
            ],
            [
                (
                    np.random.normal(size=(num_gens,2)) * np.concatenate(
                        (
                            np.ones((num_gens,1), dtype=float),
                            np.zeros((num_gens,1), dtype=float)
                        ),
                        -1
                    )
                ).view(np.complex128).reshape(num_gens,1,1)
                for _ in range(1)
            ]
        ):
            rho_g_1 = scipy.linalg.expm((
                alpha * gens_1
            ).sum(axis=0))
            rho_g_2 = scipy.linalg.expm((
                alpha * gens_2
            ).sum(axis=0))
            rho_g_p = scipy.linalg.expm((
                alpha * gens_p
            ).sum(axis=0))

            for d in range(dim_p):
                l_coeffs = np.zeros((dim_1, dim_2, dim_p), dtype='complex128')
                r_coeffs = np.zeros((dim_1, dim_2, dim_p), dtype='complex128')
                for s, t, r, q in itertools.product(
                    range(dim_1),
                    range(dim_1),
                    range(dim_2),
                    range(dim_2)
                ):
                    l_coeffs[s,r,d] += rho_g_1[s][t] * u[t] * rho_g_2[r][q] * v[q]

                for g, h, f in itertools.product(
                    range(dim_1),
                    range(dim_2),
                    range(dim_p)
                ):
                    r_coeffs[g,h,f] += rho_g_p[d][f] * u[g] * v[h]

                A.append((l_coeffs - r_coeffs).reshape(-1))
    # print(A)


    # print('convert to matrix form')
    # b = []
    # LmR = []
    # for constraint in tqdm(list(prob.constraints.values())):
    #     b.append(constraint.constant)
    #     LmR.append([
    #         constraint[prob.variablesDict()['C_{}'.format(str((i,j,k)).replace(' ', '_'))]]
    #         for i, j, k in itertools.product(range(dim_1), range(dim_2), range(dim_p))
    #     ])
    # A = np.array(LmR)
    A = np.array(A)
    print(f'A shape = {A.shape}')
    print('solve nullspace')
    # ns = scipy.linalg.null_space(A, rcond=5e-11)
    P, D, Q  = np.linalg.svd(A)
    print(P.shape, D.shape, Q.shape)

    print(f'smallest 10 singular values (numpy): {D[-10:]}')
    return np.abs(D)[sv_large]/np.abs(D)[sv_small]
