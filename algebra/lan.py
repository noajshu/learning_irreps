import sys
from tools import cg_lib
from tools import tensor as cplx_lib
import itertools
import numpy as np
import scipy.linalg
from opt_einsum import contract

from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)
USE_CACHE = False
if USE_CACHE:
    clebsch_gordan_cached = memory.cache(cg_lib.clebsch_gordan_r)
    test_clebsch_gordan_cached = memory.cache(cg_lib.test_cg_r)
else:
    clebsch_gordan_cached = cg_lib.clebsch_gordan_r
    test_clebsch_gordan_cached = cg_lib.test_cg_r

import torch
from torch import nn
from tools.tensor import device, dtype

prec = 1e-2

max_num_cg = 2000

def matrix_exponential(M):
    return cplx_lib.np_to_cplx(scipy.linalg.expm(cplx_lib.cplx_to_np(M)))

def structure_constants(gens):
    f = np.zeros((gens.shape[0], gens.shape[0], gens.shape[0]), dtype=np.complex128)
    for i, j, k in itertools.product(*(3*[range(gens.shape[0])])):
        f[i,j,k] = np.einsum('st,st->', gens[k].conj(), gens[i] @ gens[j] - gens[j] @ gens[i]) / np.einsum('st,st->', gens[k].conj(), gens[k])
    return f

class LieAlgebra:
    def __init__(self, dim, base_field='complex'):
        self.dim = dim
        assert base_field == 'complex', 'only algebras over the complex field are supported'
        self.base_field = base_field

    def __eq__(self, other):
        if isinstance(other, LieAlgebra):
            return (self.dim == other.dim) and (self.base_field == other.base_field)
        return False

    def random_element(self):
        return torch.randn(self.dim, 2, device=device, dtype=dtype)

class LieGroup:
    def __init__(self, algebra):
        self.algebra = algebra

class LieAlgebraRepresentation:
    def __init__(self, gens):
        if type(gens) is np.ndarray:
            print('converting numpy ndarray generators to torch cplx')
            gens = cplx_lib.np_to_cplx(gens)

        assert gens.shape[1] == gens.shape[2]
        self.num_gens = gens.shape[0]
        self.dim = gens.shape[1]
        self.gens = gens
        self.gens_np = cplx_lib.cplx_to_np(gens)
        self.algebra = LieAlgebra(len(self.gens))

    def matrix(self, algebra_element):
        return torch.einsum(
            'dx,dsty,xyz->stz',
            algebra_element,
            self.gens,
            cplx_lib.cplx_contractor(2)
        )

    def matrix_np(self, algebra_element):
        return np.sum(algebra_element.reshape(-1,1,1) * self.gens_np, axis=0)

    def matrix_expm(self, algebra_element):
        return matrix_exponential(self.matrix(algebra_element))

    def matrix_expm_np(self, algebra_element):
        return scipy.linalg.expm(self.matrix_np(algebra_element))

def tensor_product_np(matrices):
    assert len(matrices) >= 1
    result = matrices[0]
    for i in range(1, len(matrices)):
        result = np.einsum('st,rj->srtj', result, matrices[i]).reshape(result.shape[0]*matrices[i].shape[0], result.shape[1]*matrices[i].shape[1])
    return result

def tensor_product(matrices):
    assert len(matrices) >= 1
    result = matrices[0]
    for i in range(1, len(matrices)):
        print(
            result.shape,
            matrices[i].shape,
            cplx_lib.cplx_contractor(2).shape
        )
        result = torch.einsum(
            'stv,rjw,vwx->srtjx', result, matrices[i], cplx_lib.cplx_contractor(2)).reshape(result.shape[0]*matrices[i].shape[0], result.shape[1]*matrices[i].shape[1], 2)
    return result

class LieAlgebraTensorProductRepresentation:
    def __init__(self, representations):
        assert len(representations) >= 1

        for r in representations:
            if r.algebra != representations[0].algebra:
                raise ValueError(f'inconsistent algebras for tensor product of representations: {r.algebra} and {representations[0].algebra}')

        self.representations = representations
        self.algebra = self.representations[0].algebra
        self.num_gens = self.representations[0].num_gens
        self.dim = 1
        for rep in self.representations:
            self.dim *= rep.dim

    def matrix_np(self, algebra_element):
        raise NotImplentedError('because we exponentiate before tensor product, the algebra representation matrix is not well-defined')
        # return tensor_product_np([r.matrix_np(algebra_element) for r in self.representations])

    def matrix_expm_np(self, algebra_element):
        # return scipy.linalg.expm(self.matrix_np(algebra_element))
        return tensor_product_np([r.matrix_expm_np(algebra_element) for r in self.representations])


    def matrix_expm(self, algebra_element):
        return tensor_product([r.matrix_expm(algebra_element) for r in self.representations])



class LieAlgebraRepresentationDirectSum:
    def __init__(self, representations):
        assert all(
            rep.algebra == representations[0].algebra
            for rep in representations
        ), 'all representations must be of the same Lie Algebra'

        self.representations = representations
        self.num_reps = len(representations)
        self.cg_coeff = None
        self.max_dim = max(rep.dim for rep in self.representations)
        self.algebra = representations[0].algebra
        self.max_num_cg = 0

    def get_max_num_cg(self):
        cg = self.cg()
        return cg.shape[0]

    def matrices_expm(self, elem):
        'qst'
        O = torch.zeros(
            self.num_reps,
            self.max_dim,
            self.max_dim,
            2,
            device=device, dtype=dtype
        )
        for l, r in enumerate(self.representations):
             O[l,:r.dim,:r.dim,:] = r.matrix_expm(elem)
        return O

    def cg(self):
        if self.cg_coeff is None:
            self.cg_coeff = torch.zeros(
                max_num_cg,
                *(3*(self.num_reps, self.max_dim)),
                2, # for cplx type
                dtype=dtype, device=device
            )
            for i, j, k in itertools.product(*(3*[range(self.num_reps)])):
                while True:
                    try:
                        r1, r2, r3 = self.representations[i], self.representations[j], self.representations[k]
                        print(f'computing CG coeffs for {r1.dim} (x) {r2.dim} --> {r3.dim}')
                        print(f'i.e. i={i} (x) j={j} --> k={k}')
                        cg_coeffs = clebsch_gordan_cached(
                            r1, r2, r3
                        )
                        if np.any(cg_coeffs != 0):
                            for g, cg in enumerate(cg_coeffs):
                                test_error = test_clebsch_gordan_cached(
                                    cg,
                                    r1,
                                    r2,
                                    r3
                                )
                                if test_error < prec:
                                    self.cg_coeff[g][i,:r1.dim,j,:r2.dim,k,:r3.dim] = cplx_lib.np_to_cplx(cg)
                                    print('CG passed error test!')
                                else:
                                    print('CG error test failed')
                                    raise ValueError(
                                        f'CG error test yields {test_error} '
                                        'Since the Clebsch-Gordan coefficients are obtained '
                                        'with a randomized algorithm, you may need to try again.'
                                    )
                            self.max_num_cg = max(self.max_num_cg, cg_coeffs.shape[0])
                        else:
                            print(f'no nonzero CG coeffs for {r1.dim} (x) {r2.dim} --> {r3.dim}')
                        break
                    except ValueError:
                        print('Encountered error with CG coeffs, re-deriving')
                        pass
            print(f'observed at most {self.max_num_cg} distinct cg coeffs')
            self.cg_coeff = self.cg_coeff[:self.max_num_cg]
        return self.cg_coeff


# class MinimumDegreeNonzeroLieGroupRepresentationTensorPowerDirectSumDecompositionElements:
#     def __init__(self, rep_coll):
#         self.rep_coll = rep_coll
#
#     def filters(self, X):
#         # # GIVEN a base representation (likely tangible)
#         # # decompose the tensor powers of V until
#         # # nonzero elements of all representations
#         # # are obtained
#         # for l in range(self.rep_coll.num_reps):
#         # TODO: for now just return V



class LieGroupEquivariantLayer(nn.Module):
    def __init__(
        self,
        rep_coll,
        num_channels
    ):
        super().__init__()
        self.rep_coll = rep_coll
        self.num_channels = num_channels
        self.W = nn.Parameter(data=torch.randn(
            # self.rep_coll.num_reps,
            # self.rep_coll.num_reps,
            self.rep_coll.num_reps,
            self.num_channels,
            self.rep_coll.get_max_num_cg(),
            self.num_channels,
            2,
            device=device, dtype=dtype
        ) / (self.num_channels+1)**2)
        # self.P = nn.Parameter(data=torch.randn(
        #     self.rep_coll.num_reps,
        #     self.num_channels,
        #     self.num_channels,
        #     2, device=device, dtype=dtype
        # ) / 1e2)
        self.f = nn.Parameter(data=torch.randn(
            self.rep_coll.num_reps,
            self.rep_coll.get_max_num_cg(), 2,
            device=device, dtype=dtype
        ) / (self.rep_coll.num_reps+1)**2)


    def forward(self, dX, l, V):
        # dX4 = contract(
        #     backend='torch',
        #     memory_limit=250000000
        # )
        filters = torch.zeros(
            V.shape[0], V.shape[1], V.shape[1],
            self.rep_coll.num_reps, self.rep_coll.max_dim, 2,
            device=device, dtype=dtype
        )
        filters[:,:,:,l,:self.rep_coll.representations[l].dim,:] = dX
        # filters[:,:,:,l,:self.rep_coll.representations[l].dim,:] = (dX[...,0].pow(2)).exp()
        filters = filters + contract(
            'glsmtqrv,xijlsw,xijmty,qgz,vwyzk->xijqrk',
            self.rep_coll.cg(),
            filters,
            filters,
            self.f,
            cplx_lib.cplx_contractor(4),
            backend='torch',
            memory_limit='max_input'
        )
        # multiply by scalar fct (analogous to RBF)
        # print(f'filter exp arg {(-((filters[:,:,:,0,0,0] - 1).pow(2) + filters[:,:,:,0,0,1].pow(2)))}')
        # filters = contract(
        #     'xijlsw,xij->xijlsw',
        #     filters,
        #     (1 + (filters[:,:,:,0,0,:].pow(2).sum(-1) - 1).abs()).pow(-3),
        #     backend='torch',
        #     memory_limit=250000000
        # )
        # print('filter preview')
        # print(filters[0,0,1,0,0,:])
        # print(filters[0,0,1,1,:3,:])
        # print(X.shape)
        # filters[:,:,:,0,:self.rep_coll.representations[l].dim,:] = torch.eye(*X.shape[:-2])
        # print(dX[3][2][4] == X[3][4] - X[3][2])
        # return torch.einsum(
        #     'glsmtqrv,xijlsw,xjmctu,lmqcgdy,vwuyz->xiqdrz',
        #     self.rep_coll.cg(),
        #     filters,
        #     V,
        #     self.W,
        #     cplx_lib.cplx_contractor(4)
        # )
        # temp = contract(
        #     'glsmtqrv,xijlsw,xjmctu,lmqcgdy,vwuyz->xiqdrz',
        #     self.rep_coll.cg(),
        #     filters,
        #     V,
        #     self.W,
        #     cplx_lib.cplx_contractor(4),
        #     backend='torch',
        #     memory_limit=250000000
        # )
        # return temp
        # return contract(
        #     'xjlcsu,xjmctv,glsmtqrw,uvwz->xjqcrz',
        #     temp, temp, self.rep_coll.cg(),
        #     cplx_lib.cplx_contractor(3),
        #     backend='torch',
        #     memory_limit=250000000
        # )

        output = contract(
            # 'glsmtqrv,xijlsw,xjmctu,lmqcgdy,vwuyz->xiqdrz',
            # 'glsmtqrv,xijlsw,xjmctu,qcgdy,vwuyz->xiqdrz',
            'glsmtqrv,xijlsw,xjmctu,qcgdy,vwuyz->xiqdrz',
            self.rep_coll.cg(),
            filters,
            V,
            self.W,
            cplx_lib.cplx_contractor(4),
            backend='torch',
            memory_limit=250000000
        )
        # print('slice previews:')
        # print(output[0,0,0,:,0,:])
        # print(output[0,0,1,:,:3,:])
        return output
        #  + contract(
        #     'xjmctu,mcdv,uvz->xjmdtz',
        #     V,
        #     self.P,
        #     cplx_lib.cplx_contractor(2),
        #     backend='torch',
        #     memory_limit=250000000
        # )


class LieGroupEquivariantNeuralNetwork(nn.Module):
    def __init__(
        self,
        rep_coll,
        num_layers=3,
        num_channels=10
    ):
        super().__init__()
        self.rep_coll = rep_coll
        self.num_layers = num_layers
        self.num_channels = num_channels

        self.layers = nn.ModuleList([
            LieGroupEquivariantLayer(rep_coll, num_channels)
            for k in range(num_layers)
        ])

    def test_equivariance(
        self, l,
        num_test_samples=32,
        num_test_points=32,
        num_group_elements=4,
    ):
        X_rep = self.rep_coll.representations[l]

        X = torch.randn(num_test_samples, num_test_points, X_rep.dim, 2, device=device, dtype=dtype)
        V = torch.randn(
            # xjmctu
            num_test_samples,
            num_test_points,
            self.rep_coll.num_reps,
            self.num_channels,
            self.rep_coll.max_dim,
            2, device=device, dtype=dtype
        )

        print('self-test of group equivariance')
        for elem in [
            self.rep_coll.algebra.random_element()
            for _ in range(num_group_elements)
        ]:
            O = self.rep_coll.matrices_expm(elem)
            X_tf = torch.einsum(
                'stu,xitv,uvw->xisw',
                X_rep.matrix_expm(elem),
                X,
                cplx_lib.cplx_contractor(2)
            )
            V_tf = torch.einsum(
                'lstu,xilctv,uvw->xilcsw',
                O,
                V,
                cplx_lib.cplx_contractor(2)
            )
            V_out = self.forward(X, l, V)
            V_out_tf = torch.einsum(
                'lstu,xilctv,uvw->xilcsw',
                O,
                V_out,
                cplx_lib.cplx_contractor(2)
            )
            V_tf_out = self.forward(X_tf, l, V_tf)
            error_abs = (V_out_tf - V_tf_out).abs().sum()
            print('absolute error', error_abs.item())
            error_rel = error_abs / V_tf_out.abs().sum()
            print('relative error', error_rel.item())
            if error_rel.item() > prec:
                return False
        return True


    def forward(self, X, l, V):
        dX = -(
            X.view(*X.shape[:-2], 1, self.rep_coll.representations[l].dim, 2).add(
                -X.view(*X.shape[:-3], 1, *X.shape[-3:])
            )
        )
        for k in range(self.num_layers):
            V = self.layers[k](dX, l, V)
        return V


if __name__ == '__main__':
    irreps = np.load('irreps.npy', allow_pickle=True)[()]
    from gal_lib import three_repr_2d

    representations = LieAlgebraRepresentationDirectSum([

        # Tangible representations
        LieAlgebraRepresentation(irreps['homog_galilean_2d'][1]), # Scalar values
        LieAlgebraRepresentation(three_repr_2d), # space-time 3-vectors (t,x,y)

        # Intangible representations
        LieAlgebraRepresentation(irreps['homog_galilean_2d'][6])

    ])

    model = LieGroupEquivariantNeuralNetwork(representations, 5, 5)

    assert model.test_equivariance(0)
    assert model.test_equivariance(1)
    assert model.test_equivariance(2)
