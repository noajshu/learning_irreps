import torch
import itertools
import numpy as np
import scipy.linalg

import os
if os.environ.get('DEVICE'):
    device = torch.device(os.environ.get('DEVICE'))
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dtype = torch.float32

def cplx_contractor(n):
    c = torch.zeros(*[2 for dim in range(n+1)], dtype=dtype, device=device)
    for bs in itertools.product(*(n*[[0,1]])):
        c[bs][sum(bs)%2] = (-1)**((sum(bs)-(sum(bs)%2))/2)
    return c

def cplx_matmul(A, B):
    return torch.einsum('ijs,jkp,spt->ikt', A, B, cplx_contractor(2))

def cplx_bracket(A, B):
    return cplx_matmul(A, B) - cplx_matmul(B, A)

def cplx_to_np(A):
    return A[...,0].cpu().numpy() + 1j*A[...,1].cpu().numpy()

def bracket(A, B):
    return A.mm(B) - B.mm(A)

def np_to_cplx(A):
    return torch.stack(
        (
            torch.tensor(A.real, device=device, dtype=dtype),
            torch.tensor(A.imag, device=device, dtype=dtype),
        ),
        dim=-1
    )

const_i = torch.tensor([0,1], device=device, dtype=dtype)
def i_times(A):
    return torch.einsum('spt,s,...p->...t', cplx_contractor(2), const_i, A)

def swap(i, j):
    return torch.stack(
        (
            torch.tensor(
                np.eye(4)[[{i:j, j:i}.get(k, k) for k in range(4)]],
                dtype=dtype, device=device
            ),
            torch.zeros(4,4, dtype=dtype, device=device),
        ),
        dim=-1
    )


def perm_parity(lst):
    '''
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity


def levicivita(lst):
    if len(set(lst)) != len(lst):
        return 0
    return perm_parity(lst)

def levi_nonzero(lst):
    missing = {0,1,2}.difference(set(lst))
    if len(missing) != 1:
        return 0, -1
    index = list(missing)[0]
    coeff = levicivita(lst + [index])
    return coeff, index


def random_walk(gens):
    n = gens[0].shape[0]
    x_0 = np.random.normal(size=(n,))
    X = [x_0]
    for i in range(n):
        X.append((
            scipy.linalg.expm(sum(np.random.normal() * g for g in gens)) @ X[-1].reshape(n, 1)
        ).reshape(n,))
    return np.array(X).T


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = value.replace('\'', 'p')
    import unicodedata
    import re
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('utf-8')
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)
    return value


if __name__ == '__main__':
    # tests
    A = torch.randn(3, 3, 2, device=device, dtype=dtype)
    B = torch.randn(3, 3, 2, device=device, dtype=dtype)
    assert np.sum(np.abs(cplx_to_np(A) @ cplx_to_np(B) - cplx_to_np(cplx_matmul(A, B)))) < 1e-5
