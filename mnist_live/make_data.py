import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from mnist import MNIST
from tqdm import tqdm
import numpy as np
import json

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab as plt
from colorsys import hls_to_rgb


# %matplotlib osx

def uniform_unit(d=3):
    n = np.random.randn(d)
    n /= np.sqrt(np.sum(n**2))
    return n


def mnist_live(
    scale=1.0,
    ndim=3,
    num_events=100,
    tmin=-1,
    tmax=1,
    included_classes=range(10),
    plane='xy',
    noise_scale=0
):
    os.system('./get_data.sh')
    mndata = MNIST('./data/')

    images, labels = mndata.load_training()
    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape((images.shape[0], 28, 28))

    xy_coords = np.stack(np.meshgrid(
        np.linspace(-scale/2., scale/2., 28),
        np.linspace(-scale/2., scale/2., 28)
    ), axis=-1)
    X = []
    y = []

    for image, l in tqdm(zip(images, labels), total=images.shape[0]):
        if l not in included_classes:
            continue

        # turn into a point cloud
        t = np.random.uniform(tmin, tmax, num_events)
        # t = np.zeros(num_events)
        xy = xy_coords.reshape(-1, 2)[
            np.random.choice(
                range(xy_coords.reshape(-1, 2).shape[0]),
                size=num_events,
                p=image.reshape(-1) / np.sum(image),
                replace=True
            )
        ]
        xy += np.random.randn(*xy.shape) * noise_scale
        if ndim == 3:
            z = np.zeros(num_events)
            txyz = np.concatenate((t.reshape(-1, 1), xy, z.reshape(-1, 1)), axis=-1)
            X.append(txyz)
        elif ndim == 2:
            txy = np.concatenate((t.reshape(-1, 1), xy), axis=-1)
            X.append(txy)
        else:
            raise ValueError(f'only supports ndim 2 or 3, not {ndim}')

        y.append([
            # (0, 0) irrep
            [l] + ndim*[0],
            # (1/2, 1/2) irreps
            # [0, 1, 0, 0], # a 4-position,
            # [0, 0, 1, 0], # a 4-position,
            # [0, 0, 0, -1], # a 4-position,
            # [1, 0, 0, 0] # a 4-velocity
        ])

    X = np.array(X)
    y = np.array(y)

    print(f'embedding digits flat in the {plane} plane')
    if plane == 'xy':
        pass
    elif plane == 'xt':
        # swap y and t
        X[...,0], X[...,2] = X[...,2], X[...,0]
        # invert t
        X[...,0] *= -1
    else:
        raise ValueError(f'unknown plane:  {plane}')

    train_size = int(0.4*X.shape[0])
    dev_size = int(0.3*X.shape[0])
    test_size = X.shape[0] - train_size - dev_size
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_dev = X[train_size:][:dev_size]
    y_dev = y[train_size:][:dev_size]
    X_test = X[train_size:][dev_size:]
    y_test = y[train_size:][dev_size:]
    return (
        X_train, y_train,
        X_dev, y_dev,
        X_test, y_test
        # [], []
    )


def load_mnist_live(fname='mnist_live.npy'):
    data = np.load(
        fname,
        allow_pickle=True
    )[()]
    return (
        data['X_train'], data['y_train'],
        data['X_dev'], data['y_dev'],
        data['X_test'], data['y_test']
    )


def plot_mnist_live_example(events, outputs, cmap=mpl.cm.cool, ax=None):
    fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    norm = mpl.colors.Normalize(vmin=min(events[:,0]), vmax=max(events[:,0]))
    if events.shape[1] == 4:
        ax.scatter(
            events[:,1], events[:,2], events[:,3],
            c=events[:,0], cmap=cmap, norm=norm
        )
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
    elif events.shape[1] == 3:
        ax.scatter(
            events[:,0], events[:,1], events[:,2],
            c=events[:,0], cmap=cmap, norm=norm
        )
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$y$')
        ax.auto_scale_xyz(
            [min(events[:,0]), max(events[:,0])],
            *(2*[[
                min(min(events[:,1]), min(events[:,2])),
                max(max(events[:,1]), max(events[:,2]))
            ]])
        )
    else:
        raise ValueError(f'only supports ndim 2 or 3, not {events.shape[1]-1}')
    # final_events = np.array([np.mean(events, axis=0)])# events[np.argsort(events[:,0])[-10:]]
    # ax.scatter(
    #     final_events[:,1], final_events[:,2], final_events[:,3],
    #     c='black'
    # )
    if outputs is not None:
        ax.quiver(
            final_events[:,1], final_events[:,2], final_events[:,3],
            outputs[-2,1],
            outputs[-2,2],
            outputs[-2,3],
            length=0.1, normalize=True,
            label=r'$\hat{n}$',
            color='blue'
        )
        ax.quiver(
            final_events[:,1], final_events[:,2], final_events[:,3],
            outputs[-1,1],
            outputs[-1,2],
            outputs[-1,3],
            length=0.1, normalize=True,
            label=r'$v$',
            color='red'
        )
    # cbar = matplotlib.colorbar.ColorbarBase()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cmap=cmap, norm=norm)#, ticks=np.linspace(min(events[:,0]), max(events[:,0]), 10),)
    cbar.set_label(r'$t$')
             # boundaries=np.arange(-0.05,2.1,.1))
    #     ax, pos='bottom' #cmap=cmap, norm=norm
    # )
    plt.legend()
    # ax.auto_scale_xyz([-.5, .5], [-.5, .5], [-.5, .5])
    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(-1, 1)
    # plt.show()
    return plt

def plot(fname='mnist_live.npy'):
    X_train, y_train, X_dev, y_dev, X_test, y_test = load_mnist_live(fname=fname)
    plt = plot_mnist_live_example(X_train[5], None)
    # plt.show()
    plt.savefig('mnist_live.pdf')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='generate event points for MNIST Live')
    parser.add_argument('--num-events', type=int, default=64, help='number of events to register')
    parser.add_argument('--tmin', type=float, default=-1, help='minimum t value to draw the events from')
    parser.add_argument('--tmax', type=float, default=1, help='maximum t value to draw the events from')
    parser.add_argument('--ndim', type=int, default=3, help='number of spatial dimensions')
    parser.add_argument('--plane', type=str, default='xy', help='plane to embed digit in (xy or xt)')
    parser.add_argument('--fname', type=str, default='mnist_live.npy')
    parser.add_argument('--noise-scale', type=float, default=0)
    parser.add_argument('--included-classes', type=str, default='[0,1,2,3,4,5,6,7,8,9]', help='maximum t value to draw the events from')
    args = parser.parse_args()
    print('generating MNIST Live dataset')
    print(f'using {args.num_events} events per image with tmin {args.tmin} tmax {args.tmax}')
    print('using {} spatial dimensions'.format(args.ndim))
    print(f'saving to {args.fname}')
    X_train, y_train, X_dev, y_dev, X_test, y_test = mnist_live(
        num_events=args.num_events,
        ndim=args.ndim,
        tmin=args.tmin,
        tmax=args.tmax,
        included_classes=json.loads(args.included_classes),
        plane=args.plane,
        noise_scale=args.noise_scale
    )
    np.save(
        args.fname,
        {
            'X_train': X_train,
            'y_train': y_train,
            'X_dev': X_dev,
            'y_dev': y_dev,
            'X_test': X_test,
            'y_test': y_test,
        }
    )
    plot(fname=args.fname)
