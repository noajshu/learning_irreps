import torch
from tools.tensor import device, dtype
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from tools import cg_lib
from algebra import poincare
import itertools
from tools.tensor import slugify
from tools import training
from tools import review
import spacetime_nn
from mnist_live.make_data import load_mnist_live


def plot_nn_history(checkpoint_fname):
    print('plotting nnet training history')
    checkpoint = training.load_checkpoint(checkpoint_fname)
    review.render_plot(checkpoint, 'plots/' + checkpoint_fname.replace('.tar', '.pdf'), title='', alpha=0.8)

def print_train_test_acc(checkpoint_fname):
    print('plotting neural network activations')
    checkpoint = training.load_checkpoint(checkpoint_fname)
    model = spacetime_nn.make_model(
        model_kwargs=checkpoint['model_kwargs'],
        group=checkpoint['additional_args'].get('group', 'SO(2,1)'),
        rep_source=checkpoint['additional_args'].get('rep_source', 'tensor_power_gd'),
        gd_reps_fname=checkpoint['additional_args'].get('gd_reps_fname', 'irreps.npy'),
        cg_coeff=checkpoint['cg_coeff']
    )
    # optimizer.load_state_dict(checkpoint[args.resume_from]['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint[args.resume_from]['scheduler_state_dict'])
    # history = checkpoint[args.resume_from]['history']
    model.load_state_dict(checkpoint['current']['model_state_dict'])
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_mnist_live(
        fname=checkpoint['additional_args'].get('data_file', 'mnist_live.npy')
    )
    X_train = torch.tensor(X_train, device=device, dtype=dtype)
    y_train = torch.tensor((y_train==9).astype('int'), device=device, dtype=dtype)
    X_dev = torch.tensor(X_dev, device=device, dtype=dtype)
    y_dev = torch.tensor((y_dev==9).astype('int'), device=device, dtype=dtype)
    X_test = torch.tensor(X_test, device=device, dtype=dtype)
    y_test = torch.tensor((y_test==9).astype('int'), device=device, dtype=dtype)

    X_train = X_train[:4096]
    y_train = y_train[:4096]
    X_dev = X_dev[:124]
    y_dev = y_dev[:124]
    X_test = X_test
    y_test = y_test

    X = torch.tensor(X_test, device=device, dtype=dtype)
    y = torch.tensor(y_test, device=device, dtype=dtype)
    V = spacetime_nn.setup_first_layer(model, 1, X)

    from tools import expm64
    L, b = poincare.random_group_element()
    X_tot = torch.stack((X, torch.zeros(X.shape, device=device, dtype=dtype)), dim=-1)
    y_tot = y
    total_accuracy = 0
    total_num_batches = 0
    for i in range(0, X_tot.shape[0], 10):
        X = X_tot[i:i+10]
        y = y_tot[i:i+10]
        print(X.shape)

        # X[:,:,:4] = torch.einsum('xisv,ts->xitv', X[:,:,:4], torch.tensor(L, device=device, dtype=dtype)) + torch.tensor(b, device=device, dtype=dtype).view(1,1,4)


        V_out = model(X, 1, V.repeat(X.shape[0], *((len(V.shape)-1)*[1])))
        predicted_classes = spacetime_nn.predict(V_out)
        print(predicted_classes)
        true_classes = y[:,0,0].long()
        accuracy = (predicted_classes.argmax(-1) == true_classes).float().mean().item()
        total_accuracy += accuracy
        print(accuracy)
        total_num_batches += 1
    print(f'TOTAL MEAN TEST ACC = {total_accuracy/total_num_batches}')

def plot_2d_nn_activations(checkpoint_fname):
    print('plotting neural network activations')
    checkpoint = training.load_checkpoint(checkpoint_fname)
    model = spacetime_nn.make_model(
        model_kwargs=checkpoint['model_kwargs'],
        group=checkpoint['additional_args'].get('group', 'SO(2,1)'),
        rep_source=checkpoint['additional_args'].get('rep_source', 'tensor_power_gd'),
        gd_reps_fname=checkpoint['additional_args'].get('gd_reps_fname', 'irreps.npy'),
        cg_coeff=checkpoint['cg_coeff']
    )
    # optimizer.load_state_dict(checkpoint[args.resume_from]['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint[args.resume_from]['scheduler_state_dict'])
    # history = checkpoint[args.resume_from]['history']
    model.load_state_dict(checkpoint['current']['model_state_dict'])
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_mnist_live(
        fname=checkpoint['additional_args'].get('data_file', 'mnist_live.npy')
    )
    X = torch.tensor(X_dev, device=device, dtype=dtype)
    V = spacetime_nn.setup_first_layer(model, 1, X)
    # complexify X
    X = torch.stack((X, torch.zeros(X.shape, device=device, dtype=dtype)), dim=-1)
    i = 5

    from tools import expm64
    tf = expm64.expm64(torch.tensor(0*poincare.three_repr_2d[0], device=device, dtype=dtype))
    print('tf', tf)
    print(X.shape)
    X[:,:,:3] = torch.einsum('xisv,ts->xitv', X[:,:,:3], tf)

    V_out = model(X[i:i+1], 1, V[:1])
    # xiqdrz
    three_vec_activations = V_out[0,:,1,0,:3,0].detach().cpu().numpy() + 1j*V_out[0,:,1,0,:3,1].detach().cpu().numpy()
    predicted_classes = spacetime_nn.predict(V_out)
    print(predicted_classes)
    print('three_vec_activations[0]', three_vec_activations[0])
    print(X.shape)
    fig = plt.figure(figsize=(9, 3))
    ax = plt.subplot2grid((1, 3), (0, 0))
    # x, t
    ax.scatter(
        X.detach().cpu().numpy()[i,:,1,0], X.detach().cpu().numpy()[i,:,0,0],
        c='blue'
    )
    rescale = 2e-1
    scale_factor = rescale / three_vec_activations[...,1][0].real
    print(three_vec_activations[...,1].real*scale_factor)
    ax.quiver(
        X.detach().cpu().numpy()[i,:,1,0], X.detach().cpu().numpy()[i,:,0,0],
        three_vec_activations[...,1].real*scale_factor,
        three_vec_activations[...,0].real*scale_factor,
        # color=['r','b','g'],
        color='black',
        scale=10
    )
    ax.set_title('Original')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_ylim((-.75, .75))
    ax.set_xlim((-.75, .75))
    # ax.autoscale(tight=True)


    for k in range(2):
        X = torch.tensor(X_dev, device=device, dtype=dtype)
        V = spacetime_nn.setup_first_layer(model, 1, X)
        # complexify X
        X = torch.stack((X, torch.zeros(X.shape, device=device, dtype=dtype)), dim=-1)
        tf = expm64.expm64(torch.tensor(0.5*(-1)**k*poincare.three_repr_2d[1], device=device, dtype=dtype))
        print('tf', tf)
        print(X.shape)
        X[:,:,:3] = torch.einsum('xisv,ts->xitv', X[:,:,:3], tf)

        V_out = model(X[i:i+1], 1, V[:1])
        # xiqdrz
        three_vec_activations = V_out[0,:,1,0,:3,0].detach().cpu().numpy() + 1j*V_out[0,:,1,0,:3,1].detach().cpu().numpy()
        print('three_vec_activations.shape', three_vec_activations.shape)
        predicted_classes = spacetime_nn.predict(V_out)
        print(predicted_classes)
        print('three_vec_activations[0]', three_vec_activations[0])
        print(X.shape)
        # fig = plt.figure(figsize=(3,3))
        ax = plt.subplot2grid((1, 3), (0, k+1))
        # x, t
        ax.scatter(
            X.detach().cpu().numpy()[i,:,1,0], X.detach().cpu().numpy()[i,:,0,0],
            c='blue'
        )
        print(three_vec_activations[...,1].real*scale_factor)
        ax.quiver(
            X.detach().cpu().numpy()[i,:,1,0],
            X.detach().cpu().numpy()[i,:,0,0],
            three_vec_activations[...,1].real*scale_factor,
            three_vec_activations[...,0].real*scale_factor,
            # color=['r','b','g'],
            color='black',
            scale=10
        )
        ax.set_title(f'Transformed: {"-" if k == 0 else "+"}x Boost')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_ylim((-.75, .75))
        ax.set_xlim((-.75, .75))
        # ax.autoscale(tight=True)

    plt.tight_layout()
    # plt.subplots_adjust(hspace=1)
    # fig.suptitle('Activations of SO(2,1)-Equivariant Neural Network')
    # plt.subplots_adjust(hspace=2)
    plt.savefig('plots/xt_mnist_live_2d_points.pdf', bbox_inches='tight')


def plot_grouprep_learning():

    derived_reps = np.load('irreps.npy', allow_pickle=True)[()]

    for group_key, group_name in [
        ('so_2_1', 'SO(2, 1)'),
        ('so_3_1', 'SO(3, 1)'),
        ('so_3', 'SO(3)'),
    ]:
        for dim in derived_reps.get(group_key, []):
            if dim == 1:
                continue
            result = derived_reps[group_key][dim]

            gens, logs = result['gens'], result['logs']
            assert len(logs['min_norm']) == len(logs['loss'])
            plt.clf()
            plt.close('all')
            fig = plt.figure(figsize=(6, 3.5))
            ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
            ax.set_yscale('log')
            ax.set_ylabel(r'$\mathcal{L}$')
            ax.plot(range(len(logs['loss'])), logs['loss'], label='loss')
            ax.plot(range(len(logs['lr'])), logs['lr'], label='learning rate')
            ax.legend()
            ax.set_title(f'Convergence to Representation of {group_name}')
            ax.set_xticklabels([])

            ax = plt.subplot2grid((4, 1), (3, 0), rowspan=2)
            # ax.set_yscale('log')
            # ax.set_ylabel(r'$1/\min_i{||T_i||_1}$')
            ax.set_ylabel(r'$1/N[T]$')
            ax.plot(range(len(logs['min_norm'])), 1/np.array(logs['min_norm']), label='1 - (min norm)')
            # ax.legend()

            # plt.legend()
            plt.xlabel('iteration')
            ax.tick_params(axis='y', which='major', pad=10)
            plt.tight_layout()
            plt.savefig('plots/' + slugify(f'learning_{dim}D_representations_{group_key}') + '.pdf')

    for title, algebra_reps, algebra_derived_reps in [
        (
            '$SO(3, 1)$',
            [
                ('(0, 0)', np.concatenate(
                    poincare.irrep_lie_algebra_gens_so31(0, 0),
                    axis=0
                )),
                ('(1/2\', 1/2\')', derived_reps['so_3_1'][4]['gens']),
                ('(1/2, 1/2)', np.concatenate(
                    poincare.irrep_lie_algebra_gens_so31(1/2, 1/2),
                    axis=0
                )),
                ('(1, 1)', np.concatenate(
                    poincare.irrep_lie_algebra_gens_so31(1, 1),
                    axis=0
                )),
                ('(3/2, 3/2)', np.concatenate(
                    poincare.irrep_lie_algebra_gens_so31(3/2, 3/2),
                    axis=0
                ))
            ],
            [
                (derived_reps['so_3_1'][4]['gens'], '(1/2\', 1/2\')'),
                (np.concatenate(
                    poincare.irrep_lie_algebra_gens_so31(1/2, 1/2),
                    axis=0
                ), '(1/2, 1/2)')
            ]
        ),
        (
            '$SO(2,1)$',
            [
                ('0', np.concatenate(
                    poincare.irrep_lie_algebra_gens_so31(0, 0),
                    axis=0
                )[:3]),
                ('1\'', derived_reps['so_2_1'][3]['gens'][np.array([1,2,0])]),
            ] + [
                (str(s), np.array([
                    poincare.spin_matrices(s)[0]/1j*-1j,
                    poincare.spin_matrices(s)[1]/1j*-1j,
                    poincare.spin_matrices(s)[2]/1j
                ]))
                for s in [1, 2, 3]
            ],
            [
                (derived_reps['so_2_1'][3]['gens'][np.array([1,2,0])], '1\'')
            ]
        ),
        (
            '$SO(3)$',
            [
                ('0', np.concatenate(
                    poincare.irrep_lie_algebra_gens_so31(0, 0),
                    axis=0
                )[:3]),
                ('1\'', derived_reps['so_3'][3]['gens']),
                ('1', -1j*np.array(list(poincare.spin_matrices((3-1)/2)))),
                ('2', -1j*np.array(list(poincare.spin_matrices((5-1)/2)))),
                ('3', -1j*np.array(list(poincare.spin_matrices((7-1)/2))))
            ],
            [
                (derived_reps['so_3'][3]['gens'], '1\'')
            ]
        )
    ]:
        reps = algebra_reps
        R = {}
        for gens0, title0 in algebra_derived_reps:
            for title1, gens1 in reps:
                R[title1] = {}
                for title2, gens2 in reps:
                    print(f'\n\n{title0} (x) {title1} -> {title2}')
                    cg_coeffs = cg_lib.clebsch_gordan(
                        gens0,
                        gens1,
                        gens2
                    )
                    r = cg_lib.clebsch_sv_ratio(
                        gens0,
                        gens1,
                        gens2
                    )
                    R[title1][title2] = r
                    if type(cg_coeffs) is not int:
                        print(cg_coeffs.shape[0])

            print(R)
            plt.clf()
            plt.close('all')
            xpos = dict(zip([title for title, gens in reps], range(len(reps))))

            w = 0.4
            fig = plt.figure(figsize=(3, 10))
            # fig.suptitle('Learned ' + title + ' Tensor Product Structure')
            for i, title1 in enumerate(sorted(R)):
                ax = plt.subplot(len(R), 1, i+1)
                for title2 in R[title1]:
                    ax.bar(xpos[title2],
                        R[title1][title2],
                        width=w,
                        align='center'
                    )
                ax.set_title(f'${title0} \otimes {title1}$')
                # ax.set_xlabel('product direct sum element')
                ax.set_ylabel('$SV_2 / SV_1$')
                ax.set_yscale('log')
                ax.autoscale(tight=True)
                ax.set_xticks(list(xpos.values()))
                ax.set_xticklabels(list(xpos.keys()), rotation=35)

            plt.subplots_adjust(hspace=1)
            plt.savefig(
                'plots/' + slugify(f'tensor_product_decomposition_SVD_{title0}_{title}') + '.pdf',
                bbox_inches='tight'
            )


if __name__ == '__main__':
    import argparse
    import os.path
    parser = argparse.ArgumentParser(description='generate event points for MNIST Live')
    parser.add_argument('plot_type', type=str, help='grouprep_learning, nn_history, or 2d_nn_activations')
    args = parser.parse_args()
    if args.plot_type == 'grouprep_learning':
        plot_grouprep_learning()
    elif args.plot_type == 'nn_history':
        for fname in ['checkpoint_SO21_xy_plane.tar', 'checkpoint_SO31_xy_plane.tar']:
            if os.path.exists(fname):
                print(f'plotting nn history for {fname}')
                plot_nn_history(fname)
                print(f'printing test accuracy for {fname}')
                print_train_test_acc(fname)
            else:
                print(f'skipping {fname} plot since .tar file not available')
    elif args.plot_type == '2d_nn_activations':
        plot_2d_nn_activations('checkpoint_SO21_xt_plane.tar')
    else:
        raise ValueError(f'unknown plot type: {args.plot_type}')
