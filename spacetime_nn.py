import numpy as np
from algebra import lan
from mnist_live.make_data import load_mnist_live
from tools import training
from tools import tensor
from tools.review import render_plot
import json
from time import time
import torch
from torch import nn
from tools.tensor import dtype, device


def predict(V_out):
    return V_out[:,:,0,:2,0,:].pow(2).sum((1,-1))


def make_model(
    model_kwargs={},
    group='SO(3,1)',
    rep_source='formulas',
    gd_reps_fname='irreps.npy',
    cg_coeff=None
):
    irreps = np.load(gd_reps_fname, allow_pickle=True)[()]
    print(irreps.keys())
    if group == 'SO(2,1)':
        from algebra import poincare
        # space-time 3-vectors (t,x,y)
        three_vector_irrep = lan.LieAlgebraRepresentation(
            poincare.three_repr_2d[np.array([1,2,0])]
        )
        scalar_irrep = lan.LieAlgebraRepresentation(np.zeros((3,1,1)))
        if rep_source == 'formulas':
            representations = lan.LieAlgebraRepresentationDirectSum([
                scalar_irrep,
                three_vector_irrep,
            ] + [
                lan.LieAlgebraRepresentation(np.array([
                    # start from the Wigner D matrices for SO(3)
                    # K_x = -iJ_x
                    poincare.spin_matrices(s)[0]/1j*-1j,
                    # K_y = -iJ_y
                    poincare.spin_matrices(s)[1]/1j*-1j,
                    # J_z = J_z
                    poincare.spin_matrices(s)[2]/1j
                ]))
                for s in [1, 2, 3]
            ])
        elif rep_source == 'tensor_power_base_representation':
            representations = lan.LieAlgebraRepresentationDirectSum([
                scalar_irrep,
                three_vector_irrep, # space-time 3-vectors (t,x,y)
                lan.LieAlgebraTensorProductRepresentation([three_vector_irrep, three_vector_irrep])
            ])
        elif rep_source == 'tensor_power_gd':
            gd_rep = lan.LieAlgebraRepresentation(irreps['so_2_1'][3]['gens'][np.array([1,2,0])])
            representations = lan.LieAlgebraRepresentationDirectSum([
                scalar_irrep,
                three_vector_irrep, # space-time 3-vectors (t,x,y)
                lan.LieAlgebraTensorProductRepresentation([gd_rep, gd_rep])
            ])
        else:
            raise ValueError(f'unknown irrep source {rep_source}')
    elif group == 'SO(3,1)':
        from algebra import poincare
        # space-time 4-vectors (t,x,y,z)
        four_vector_irrep = lan.LieAlgebraRepresentation(
            np.concatenate(poincare.four_repr, axis=0)
        )
        scalar_irrep = lan.LieAlgebraRepresentation(np.zeros((6,1,1)))
        if rep_source == 'formulas':
            representations = lan.LieAlgebraRepresentationDirectSum([
                scalar_irrep,
                four_vector_irrep,
            ] + [
                lan.LieAlgebraRepresentation(poincare.irrep_lie_algebra_gens_so31(s, s))
                for s in [1, 3/2]
            ])
        elif rep_source == 'tensor_power_base_representation':
            representations = lan.LieAlgebraRepresentationDirectSum([
                scalar_irrep,
                four_vector_irrep,
                lan.LieAlgebraTensorProductRepresentation([four_vector_irrep, four_vector_irrep])
            ])
        elif rep_source == 'tensor_power_gd':
            gd_rep = lan.LieAlgebraRepresentation(irreps['so_3_1'][4]['gens'])
            representations = lan.LieAlgebraRepresentationDirectSum([
                scalar_irrep,
                four_vector_irrep, # space-time 3-vectors (t,x,y)
                lan.LieAlgebraTensorProductRepresentation([gd_rep, gd_rep])
            ])
        else:
            raise ValueError(f'unknown irrep source {rep_source}')
    elif group == 'Galilei(2,1)':
        from algebra import galileo

        raise NotImplementedError
        # space-time 3-vectors (t,x,y)
        three_vector_irrep = lan.LieAlgebraRepresentation(galileo.three_repr_2d)
        scalar_irrep = lan.LieAlgebraRepresentation(np.zeros((three_vector_irrep.algebra.dim,1,1)))
        representations = lan.LieAlgebraRepresentationDirectSum([
            scalar_irrep,
            three_vector_irrep,
            lan.LieAlgebraTensorProductRepresentation([three_vector_irrep, three_vector_irrep])
        ])
    elif group == 'Galilei(3,1)':
        from gal_lib import four_repr
        four_repr = np.concatenate(four_repr, axis=0)

        raise NotImplementedError

    else:
        raise ValueError(f'unknown group {group}')

    if cg_coeff is not None:
        representations.cg_coeff = tensor.np_to_cplx(cg_coeff)
    model = lan.LieGroupEquivariantNeuralNetwork(representations, **model_kwargs)
    return model


def setup_first_layer(model, batch_size, X):
    # xjmctu
    V = torch.zeros(
        batch_size, X.shape[1], model.rep_coll.num_reps,
        model.num_channels, model.rep_coll.max_dim, 2,
        device=device, dtype=dtype
    )
    V[:,:,0,0,0,0] = 1
    return V


if __name__ == '__main__':
    args = training.args('MNIST-Live Galilean-Equivariant Neural Network')
    model_kwargs = json.loads(args.model_kwargs_json)
    additional_args = json.loads(args.additional_args_json)
    rep_source = additional_args.get('rep_source', 'formulas')

    group = additional_args.get('group', 'Galilei(3,1)')
    gd_reps_fname = additional_args.get('gd_reps_fname', 'irreps.npy')
    num_layers = model_kwargs.get('num_layers', 2)
    num_channels = model_kwargs.get('num_channels', 10)
    data_fname = additional_args.get('data_file', 'mnist_live.npy')

    print(f'using representations from {rep_source}')
    print(f'gradient-descent-derived representations are in file {gd_reps_fname}')
    print(f'model kwargs: {model_kwargs}')
    print(f'using {group}-equivariant neural model with {num_layers} layers and {num_channels} channels')

    print(f'args.skip_equivariance_test = {args.skip_equivariance_test}')

    model = make_model(
        model_kwargs=model_kwargs,
        group=group,
        rep_source=rep_source,
        gd_reps_fname=gd_reps_fname
    )

    if args.skip_equivariance_test:
        print('skipping equivariance self-test')
    else:
        print('testing model equivariance')
        assert model.test_equivariance(1)

    X_train, y_train, X_dev, y_dev, X_test, y_test = load_mnist_live(
        fname=data_fname
    )
    X_train = torch.tensor(X_train, device=device, dtype=dtype)
    y_train = torch.tensor((y_train==9).astype('int'), device=device, dtype=dtype)
    X_dev = torch.tensor(X_dev, device=device, dtype=dtype)
    y_dev = torch.tensor((y_dev==9).astype('int'), device=device, dtype=dtype)


    train_size = additional_args.get('train_size', None)
    dev_size = additional_args.get('dev_size', None)
    if X_train.shape[0] < train_size or X_dev.shape[0] < dev_size:
        raise ValueError(f'invalid train or dev size: {(train_size, dev_size)} vs {(X_train.shape[0], X_dev.shape[0])}')
    if train_size is not None:
        X_train, y_train = X_train[:train_size], y_train[:train_size]
    else:
        train_size = X_train.shape[0]
    if dev_size is not None:
        X_dev, y_dev = X_dev[:dev_size], y_dev[:dev_size]
    else:
        dev_size = X_dev.shape[0]

    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_dev, y_dev)


    num_epochs = args.epochs
    #####################
    #### boilerplate ####
    #####################
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=additional_args.get('shuffle_train', True), num_workers=0, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size,
        num_workers=0, pin_memory=False
    )

    V = setup_first_layer(model, args.batch_size, X_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, cooldown=100, patience=100)
    velocity_loss_func = torch.nn.MSELoss()
    classification_loss_func = torch.nn.CrossEntropyLoss()


    loss = 1e10
    history = {
        'train': {
            'accs': {
                'accs': [],
                'epoch_accs': [],
                'epoch_times': [],
                'times': [],
                'batch': []
            },
            'losses': {
                'batch': [],
                'times': [],
                'losses': []
            },
            'lr': {
                'epochs': [],
                'lrs': []
            }
        },
        'test': {
            'accs': {
                'accs': [],
                'times': [],
                'batch': []
            },
            'losses': {
                'batch': [],
                'times': [],
                'losses': []
            }
        },
        'dev_acc':  {
            'batch': [],
            'acc': []
        },
        'dev_random_tf_acc':  {
            'batch': [],
            'acc': []
        },
    }
    with torch.autograd.set_detect_anomaly(True):
    # # with torch.autograd.set_detect_anomaly(False):
        t = 0
        epoch = 0
        while epoch < num_epochs:
            print('epoch {}'.format(epoch))
            total_train_acc = 0
            for i, (X, y) in enumerate(train_loader):
                print('\n\nbatch {}'.format(i))
                # complexify X
                X = torch.stack((X, torch.zeros(X.shape, device=device, dtype=dtype)), dim=-1)

                if i % args.checkpoint_on_batch == 0:
                    print('dev eval')
                    total_accuracy = 0
                    total_loss = 0
                    if i > 0:
                        for j, (X, y) in enumerate(test_loader):
                            print('\n\nbatch {}'.format(j))
                            X = torch.stack((X, torch.zeros(X.shape, device=device, dtype=dtype)), dim=-1)

                            model.eval()
                            V_out = model(X, 1, V[:X.shape[0]])
                            # global predictions
                            predicted_classes = predict(V_out)
                            true_classes = y[:,0,0].long()
                            classification_loss = classification_loss_func(
                                predicted_classes,
                                true_classes
                            ) / args.batch_size
                            accuracy = (predicted_classes.argmax(-1) == true_classes).float().mean().item()
                            total_accuracy += accuracy
                            total_loss += classification_loss.item()

                        mean_test_acc = total_accuracy / float(j+1)
                        mean_loss = total_loss / float(j+1)
                        print(f'mean test acc = {mean_test_acc}')
                        history['test']['losses']['batch'].append(history['train']['losses']['batch'][-1])
                        history['test']['accs']['batch'].append(history['train']['losses']['batch'][-1])
                        history['test']['losses']['times'].append(time())
                        history['test']['accs']['times'].append(time())
                        history['test']['accs']['accs'].append(mean_test_acc)
                        history['test']['losses']['losses'].append(mean_loss)

                    print('checkpointing to {}'.format(args.checkpoint))
                    training.save_to_checkpoint(
                        model, optimizer, scheduler,
                        loss if type(loss) is float else loss.item(),
                        history, model_kwargs, tensor.cplx_to_np(model.rep_coll.cg()),
                        args.checkpoint,
                        additional_args=additional_args
                    )

                model.train()
                optimizer.zero_grad()
                V_out = model(X, 1, V[:X.shape[0]])
                predicted_classes = predict(V_out)

                true_classes = y[:,0,0].long()
                classification_loss = classification_loss_func(
                    predicted_classes,
                    true_classes
                ) / args.batch_size

                print('classification loss: {}'.format(classification_loss.item()))

                loss = classification_loss

                print(f'total loss: {loss.item()}')
                print(f'current lr = {optimizer.param_groups[0]["lr"]}')
                history['train']['losses']['times'].append(time())
                if not history['train']['losses']['batch']:
                    history['train']['losses']['batch'] = [i]
                else:
                    history['train']['losses']['batch'].append(history['train']['losses']['batch'][-1] + 1)
                history['train']['losses']['losses'].append(loss.item())

                accuracy = (predicted_classes.argmax(-1) == true_classes).float().mean().item()
                print('class accuracy: {}'.format(accuracy))
                total_train_acc += accuracy

                history['train']['accs']['accs'].append(accuracy)
                history['train']['accs']['times'].append(time())

                print('performing weights update')
                loss.backward()
                optimizer.step()
                # scheduler.step(loss)

            history['train']['lr']['epochs'].append(epoch)
            history['train']['lr']['lrs'].append(optimizer.param_groups[0]["lr"])


            epoch += 1
            print(f'total train acc: {total_train_acc / float(i+1)}')
            scheduler.step(total_train_acc / float(i+1))
            history['train']['accs']['epoch_accs'].append(total_train_acc / float(i+1))
            history['train']['accs']['epoch_times'].append(time())

            if (epoch % 20 == 0 or train_size >= 100) and dev_size > 0:
                print('dev eval')
                total_accuracy = 0
                total_loss = 0
                for j, (X, y) in enumerate(test_loader):
                    print('\n\nbatch {}'.format(j))
                    X = torch.stack((X, torch.zeros(X.shape, device=device, dtype=dtype)), dim=-1)

                    model.eval()
                    V_out = model(X, 1, V[:X.shape[0]])
                    # global predictions
                    predicted_classes = predict(V_out)
                    true_classes = y[:,0,0].long()
                    classification_loss = classification_loss_func(
                        predicted_classes,
                        true_classes
                    ) / args.batch_size
                    accuracy = (predicted_classes.argmax(-1) == true_classes).float().mean().item()
                    total_accuracy += accuracy
                    total_loss += classification_loss.item()

                mean_test_acc = total_accuracy / float(j+1)
                mean_loss = total_loss / float(j+1)
                print(f'mean test acc = {mean_test_acc}')
                history['test']['losses']['batch'].append(history['train']['losses']['batch'][-1])
                history['test']['accs']['batch'].append(history['train']['losses']['batch'][-1])
                history['test']['losses']['times'].append(time())
                history['test']['accs']['times'].append(time())
                history['test']['accs']['accs'].append(mean_test_acc)
                history['test']['losses']['losses'].append(mean_loss)

            if epoch % 50 == 0 or train_size >= 100:
                print('checkpointing to {}'.format(args.checkpoint))
                training.save_to_checkpoint(
                    model, optimizer, scheduler,
                    loss if type(loss) is float else loss.item(),
                    history, model_kwargs,
                    tensor.cplx_to_np(model.rep_coll.cg()),
                    args.checkpoint,
                    additional_args=additional_args
                )
                render_plot(torch.load(args.checkpoint), args.plot_to, title='', alpha=0.8)
