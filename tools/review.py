import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

def moving_avg(arr, n=10):
    return (
        [np.mean(arr[i:i+n]) for i in range(len(arr))],
        [np.std(arr[i:i+n]) for i in range(len(arr))]
    )

def render_plot(checkpoint, fname, title='', alpha=0.8):
    test_accs_times = np.array(checkpoint['current']['history']['test']['accs']['times']) / 3600.
    test_losses_times = np.array(checkpoint['current']['history']['test']['losses']['times']) / 3600.
    test_accs = np.array(checkpoint['current']['history']['test']['accs']['accs'])
    test_losses = np.array(checkpoint['current']['history']['test']['losses']['losses'])
    train_losses = np.array(checkpoint['current']['history']['train']['losses']['losses'])

    train_accs_times = np.array(checkpoint['current']['history']['train']['accs']['times']) / 3600.
    train_losses_times = np.array(checkpoint['current']['history']['train']['losses']['times']) / 3600.
    train_losses = np.array(checkpoint['current']['history']['train']['losses']['losses'])
    train_accs = np.array(checkpoint['current']['history']['train']['accs']['accs'])

    train_epoch = np.array(checkpoint['current']['history']['train']['lr']['epochs'])
    train_lr = np.array(checkpoint['current']['history']['train']['lr']['lrs'])

    plt.close('all')
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(4, 5))

    ax = plt.subplot(211)
    # ax.set_title('Accuracy')
    plt.plot(
        test_accs_times - train_accs_times[0],
        test_accs,
        label='test accuracy',
        alpha=alpha
    )
    # plt.plot(
    #     test_accs_times - train_accs_times[0],
    #     1 - test_accs,
    #     label='test error',
    #     alpha=alpha
    # )
    means, stds = moving_avg(train_accs, n=15)
    # plt.plot(
    #     train_accs_times - train_accs_times[0],
    #     means,
    #     label='train accuracy',
    #     alpha=alpha
    # )

    plt.errorbar(
        (train_accs_times - train_accs_times[0])[::10],
        means[::10],
        stds[::10],
        label=r'train accuracy $(\pm\sigma)$',
        alpha=alpha
    )
    plt.ylim((0, 1))
    # plt.plot(
    #     train_accs_times - train_accs_times[0],
    #     1 - train_accs,
    #     label='train error',
    #     alpha=alpha
    # )
    # plt.yscale('log')
    plt.xlabel('time (hr)')
    plt.legend()
    # plt.title(title)

    ax = plt.subplot(413)
    # ax.set_title('Loss')
    plt.plot(
        test_losses_times - train_losses_times[0],
        test_losses,
        label='test loss',
        alpha=alpha
    )
    # plt.yscale('log')
    plt.xlabel('time (hr)')
    plt.legend()
    # plt.title(title)

    plt.subplot(414)
    # plt.plot(
    #     train_losses_times - train_losses_times[0],
    #     train_losses,
    #     label='train loss',
    #     alpha=alpha
    # )
    means, stds = moving_avg(train_losses, n=15)
    plt.errorbar(
        (train_losses_times - train_losses_times[0])[::10],
        means[::10],
        stds[::10],
        label=r'train loss $(\pm\sigma)$',
        alpha=alpha
    )
    # plt.yscale('log')
    plt.xlabel('time (hr)')
    plt.legend()

    # plt.subplot(414)
    # plt.plot(
    #     train_epoch,
    #     train_lr
    # )
    # plt.yscale('log')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')

    # plt.title(title)
    plt.tight_layout()
    plt.savefig(fname)
    plt.clf()

def dict_get(d, path):
    for key in path[1:].split('.'):
        d = d[key]
    return d


if __name__ == '__main__':
    parser = argparse.ArgumentParser('review some training logs')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.tar',
        help='where to load/store checkpoints')
    parser.add_argument(
        '--print', type=str, default=None,
        help='(default: None) If present, print the array '
        '(row-delimited) at the specified JSON path. '
        'Ex: ".current.history.test.accs"')
    parser.add_argument('--plot-to', type=str, default=None,
        help='(default: None) If present, will plot '
        'loss and accuracy and save to this file')
    parser.add_argument('--plot-title', type=str, default='')

    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if args.print is not None:
        for val in dict_get(checkpoint, args.print):
            print(val)

    if args.plot_to is not None:
        render_plot(checkpoint, args.plot_to, title=args.plot_title)
