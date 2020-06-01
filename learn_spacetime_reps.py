import torch
import numpy as np
import itertools
from tools import cg_lib
from enum import Enum, auto
from algebra.poincare import irrep_lie_algebra_gens_so31
import tools.tensor
tools.tensor.dtype = torch.float64
from tools.tensor import dtype, device, cplx_contractor, \
    cplx_to_np, np_to_cplx, i_times, bracket, \
    levi_nonzero, random_walk
dtype = torch.float64


torch.manual_seed(528491)

class Group(Enum):
    lorentz = auto()
    galilean = auto()
    rotations = auto()

def derive_explicit_matrix_reps_retry(**kwargs):
    result = None
    while result is None:
        result = derive_explicit_matrix_reps(**kwargs)
    return result


def derive_explicit_matrix_reps(n=4, group=Group.lorentz,
    num_space_dims=3, init_gens=None, verbose=False,
    convergence_tol=1e-9, lr_cutoff=1e-13, use_exact_sol=False
):
    # if n != 4:
    #     exit(0)
    print(f'Deriving explicit {n}-dimensional matrix reps for the {group} group in {num_space_dims} spatial dimensions')

    logs = {
        'loss': [],
        'min_norm': [],
        'lr': []
    }
    if group is Group.lorentz:
        if num_space_dims == 3:
            J_1 = torch.randn(n, n, device=device, dtype=dtype, requires_grad=True)
            K_1 = torch.randn(n, n, device=device, dtype=dtype, requires_grad=True)
            K_2 = torch.randn(n, n, device=device, dtype=dtype, requires_grad=True)

        elif num_space_dims == 2:
            K_1 = torch.randn(n, n, device=device, dtype=dtype, requires_grad=True)
            J_3 = torch.randn(n, n, device=device, dtype=dtype, requires_grad=True)

        else:
            raise ValueError(f'bad num_space_dims {num_space_dims}')

    elif group is Group.rotations:
        if num_space_dims == 3:
            # j_1_diag = torch.randn(n, 2, device=device, dtype=dtype, requires_grad=True)
            J_1 = torch.randn(n, n, device=device, dtype=dtype, requires_grad=True)
            J_2 = torch.randn(n, n, device=device, dtype=dtype, requires_grad=True)
        else:
            raise ValueError(f'so{num_space_dims} not supported for so')
    if group is Group.lorentz:
        if num_space_dims == 3:
            optimizer = torch.optim.Adam([J_1, K_1, K_2], lr=1e-1)
        elif num_space_dims == 2:
            optimizer = torch.optim.Adam([K_1, J_3], lr=1e-1)
    elif group is Group.rotations:
        if num_space_dims == 3:
            optimizer = torch.optim.Adam([J_1, J_2], lr=1e-1)
        else:
            raise ValueError(f'so{num_space_dims} not supported for so')
    else:
        raise ValueError(f'unknown group: {group}')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True,
        cooldown=100,
        patience=100,
        min_lr=1e-16, eps=1e-16
    )
    t = 0
    while True:
        optimizer.zero_grad()
        loss = 0

        if group is Group.lorentz:
            if num_space_dims == 3:
                # J_1 = torch.zeros(n, n, 2, device=device, dtype=dtype)
                # J_1[range(n), range(n)] = j_1_diag

                # K_1 = torch.zeros(n, n, 2, device=device, dtype=dtype)
                # K_1[range(n), range(n)] = k_1_diag

                K_3 = bracket(J_1, K_2)
                J_3 = -bracket(K_1, K_2)
                J_2 = -bracket(J_1, J_3)

            elif num_space_dims == 2:
                K_2 = bracket(J_3, K_1)
                J_1 = J_2 = K_3 = 1000*torch.ones(K_2.size(), device=device, dtype=dtype)

        elif group is Group.rotations:
            if num_space_dims == 3:
                # J_1 = torch.zeros(n, n, 2, device=device, dtype=dtype)
                # J_1[range(n), range(n),0] = j_1_diag[...,0]
                # J_1_re = torch.zeros(*J_1.shape, device=device, dtype=dtype)
                # J_1_re[...,0] = J_1[...,0]
                # J_2_re = torch.zeros(*J_2.shape, device=device, dtype=dtype)
                # J_2_re[...,0] = J_2[...,0]
                # J_3_re = -i_times(bracket(J_1_re, J_2_re))
                # J_3_re = bracket(J_1_re, J_2_re)
                J_3 = bracket(J_1, J_2)
            else:
                raise ValueError(f'so{num_space_dims} not supported for so')

            K = 1000*torch.ones(3, *J_1.size(), device=device, dtype=dtype)

        if group is Group.lorentz:
            J = torch.stack([J_1, J_2, J_3], dim=0)
            K = torch.stack([K_1, K_2, K_3], dim=0)

            if num_space_dims == 3:
                # since for 2D we don't have overlap between Js and Ks
                for i in range(3):
                    loss = loss + (bracket(J[i], K[i]) - 0).abs().mean()
        else:
            J = torch.stack([J_1, J_2, J_3], dim=0)

        for (i, j) in itertools.product(range(3), range(3)):
            if i==j:
                continue

            eps, k = levi_nonzero([i, j])

            # Lorentz, Galilean, and so algebra
            # [J_i, J_j] = i epsilon_ijk J_k
            if num_space_dims == 3:
                if group is not Group.rotations:
                    error = (bracket(J[i], J[j]) - eps * J[k]).abs().mean()
                    # print(f'i, j, k = {i, j, k}, error = {error.item()}')
                    loss = loss + error
                else:
                    # print(f'i, j, k = {i, j, k}, error = {(bracket(J[i], J[j]) - eps * (J[k])).abs().mean().item()}')
                    loss = loss + (bracket(J[i], J[j]) - eps * J[k]).abs().mean()

            if group is Group.lorentz and (num_space_dims == 3 or (i == 2 and j != 2)):
                # we only have J_3 in the 2+1D case
                # [J_i, K_j] = i epsilon_ijk K_k
                error = (bracket(J[i], K[j]) - eps * K[k]).abs().mean()
                loss = loss + error

            if group is Group.lorentz:
                if num_space_dims == 3 or (i != 2 and j != 2):
                    # [K_i, K_j] = -epsilon_ijk J_k
                    error = (bracket(K[i], K[j]) - -eps * J[k]).abs().mean()
                    loss = loss + error

        if group is Group.lorentz:
            min_norm = torch.min(torch.min(J.pow(2).sum((-1,-2))), torch.min(K.pow(2).sum((-1,-2))))
            # print(min_norm.item())
        elif group is Group.rotations:
            min_norm = torch.min(J.pow(2).sum((-1,-2)))
        else:
            raise ValueError(f'unknown group: {group}')

        # print(f'error_loss = {loss}')
        min_norm = min_norm.clamp(max=1)
        loss = loss / min_norm
        # loss = loss + 1 / min_norm
        # if group is Group.rotations:
        #     if n > 3:
        #         sv_penalty = simplecg.projector_sv(
        #             3,
        #             n**2, simplecg.tprep(J),
        #             1, simplecg.rep(torch.zeros(3,1,1,device=device,dtype=dtype)),
        #             # 3, simplecg.rep(simplecg.T),
        #             rank=-2
        #         ).clamp(max=1000)**2
        #         print(f'sv penalty = {sv_penalty.item()}')
        #         loss = loss / sv_penalty

        logs['loss'].append(loss.item())
        logs['min_norm'].append(min_norm.item())
        logs['lr'].append(optimizer.param_groups[0]['lr'])

        if t % 100 == 0:
            print(f'lr = {optimizer.param_groups[0]["lr"]} min_norm = {min_norm.item()} loss = {loss.item()}', flush=True)

        loss.backward()

        optimizer.step()
        scheduler.step(loss)

        if loss.item() <= convergence_tol:
            # if group == 'so':
            #     if num_space_dims == 3 and n == 3:
            #         w, v = np.linalg.eig(cplx_to_np(J.detach()))
            #         print(w.round(4))
            #         ranks = np.linalg.matrix_rank(cplx_to_np(J.detach()))
            #         if list(ranks) != [2,2,2]:
            #             print(f'Wrong Ranks: {ranks} Retry')
            #             return None

            # print('appears convergent: saving to save_convergent.npy')
            # np.save('save_convergent.npy', J[...,0].detach().cpu().numpy())

            # print(J, K)
            if group is Group.lorentz:
                if num_space_dims == 3:
                    gens0 = np.concatenate(
                        [J.detach().cpu().numpy(), K.detach().cpu().numpy()],
                        axis=0
                    )
                    gens1 = gens0
                    gens2 = np.concatenate(
                        irrep_lie_algebra_gens_so31(0, 0),
                        axis=0
                    )
                elif num_space_dims == 2:
                    gens0 = np.concatenate(
                        [[J.detach().cpu().numpy()[2]], K.detach().cpu().numpy()[:-1]],
                        axis=0
                    )
                    gens1 = gens0
                    gens2 = np.concatenate(
                        irrep_lie_algebra_gens_so31(0, 0),
                        axis=0
                    )[:3]
            elif group is Group.rotations:
                gens0 = J.detach().cpu().numpy()
                gens1 = gens0
                gens2 = irrep_lie_algebra_gens_so31(0, 0)[0]

            print(gens0.shape, gens1.shape, gens2.shape)
            cg_coeffs = cg_lib.clebsch_gordan(
                gens0,
                gens1,
                gens2,
                zero=1e-2
            )

            if type(cg_coeffs) is int and cg_coeffs == 0:
                print('0 norms found, TRY AGAIN')
                # exit(1)
                return None
            if cg_coeffs.shape[0] > 1:
                print('too many norms, TRY AGAIN')
                # exit(1)
                return None

            print('~convergence')
            break

        if loss.item() < 100 and optimizer.param_groups[0]['lr'] / loss.item() < 1e-4:
            print('lr / loss < 1e-4 TRY AGAIN')
            return None

        if optimizer.param_groups[0]['lr'] < lr_cutoff:
            print('TRY AGAIN')
            return None

        t += 1

    gens = np.concatenate((J.detach().cpu().numpy(), K.detach().cpu().numpy()), axis=0)

    if group is Group.lorentz and num_space_dims == 2:
        gens = np.array([gens[2], gens[3], gens[4]])
    elif group is Group.rotations and num_space_dims == 3:
        gens = np.array(gens[:3])
    return {'gens': gens, 'logs': logs}


if __name__ == '__main__':
    convergence_tol = 1e-10

    irrep_dims_so_3_1 = [4]
    irrep_dims_so_2_1 = [3]
    irrep_dims_so_3 = [3]

    irreps = {}
    irreps['so_3_1'] = {
        n: derive_explicit_matrix_reps_retry(
            n=n, group=Group.lorentz,
            convergence_tol=convergence_tol
        )
        for n in irrep_dims_so_3_1
    }
    irreps['so_2_1'] = {
        n: derive_explicit_matrix_reps_retry(
            n=n, group=Group.lorentz,
            convergence_tol=convergence_tol,
            num_space_dims=2
        )
        for n in irrep_dims_so_2_1
    }
    irreps['so_3'] = {
        n: derive_explicit_matrix_reps_retry(
            n=n, group=Group.rotations, num_space_dims=3,
            convergence_tol=convergence_tol
        )
        for n in irrep_dims_so_3
    }

    irreps['so_3_1'][1] = {'gens': np.zeros((6,1,1)), 'logs': None}
    irreps['so_2_1'][1] = {'gens': np.zeros((3,1,1)), 'logs': None}
    irreps['so_3'][1] = {'gens': np.zeros((3,1,1)), 'logs': None}
    np.save('irreps.npy', irreps)
