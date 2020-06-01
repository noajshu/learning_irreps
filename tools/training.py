import torch
import argparse
from time import time
import json
import inspect
import os
if os.environ.get('DEVICE'):
    device = torch.device(os.environ.get('DEVICE'))
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def args(title):
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--checkpoint', type=str, default='checkpoint.tar',
        help='where to load/store checkpoints')
    parser.add_argument('--resume-from', type=str, default='current',
        help='determines which checkpoint to resume from ("current" (default) / "best")')
    parser.add_argument('--data-dir', type=str, default='./')
    parser.add_argument('--skip-equivariance-test', default=False, action='store_true')
    parser.add_argument('--batch-size', type=int, default=32, metavar='n',
                        help='input batch size for training (default: 128)')
    # parser.add_argument('--epoch-size', type=int, default=None, metavar='N',
    #                 help='epoch size (leave blank to use entire dataset)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01) note this is ignored if'
                        'restoring optimizer from a checkpoint')
    parser.add_argument('--model-kwargs-json', default='{}',
        help='kwargs for instantiating the model. '
        'These are backed up in checkpoints, and used by default. '
        'If not provided, will use checkpoint kwargs (if available) or the defaults.')

    parser.add_argument('--additional-args-json', default='{}',
        help='catch-all args to be processed by user code')

    parser.add_argument('--plot-to', default='training_plot.pdf', type=str,
        help='where to save training plots (loss etc.)')

    parser.add_argument('--dev-eval-on-batch', default=100, type=int,
        help='number of batches to run before each dev eval')
    parser.add_argument('--checkpoint-on-batch', default=100, type=int,
        help='number of batches to run before each save of the checkpoint')
    args = parser.parse_args()
    return args


def save_to_checkpoint(
    model, optimizer, scheduler, loss,
    history, model_kwargs, cg_coeff, fname,
    additional_args={}
):
    current = {
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'history': history,
        'source': inspect.getsource(inspect.getmodule(inspect.currentframe()))
    }
    if os.path.exists(fname):
        checkpoint = torch.load(fname)
        best = checkpoint['best']
        if best['loss'] > current['loss']:
            print('new best loss')
            best = current
    else:
        best = current
    torch.save(
        {
            'model_kwargs': model_kwargs,
            'cg_coeff': cg_coeff,
            'additional_args': additional_args,
            'current': current,
            'best': best
        },
        fname
    )


def load_checkpoint(fname, device=device):
    if os.path.exists(fname):
        checkpoint = torch.load(fname, map_location=device)
    else:
        checkpoint = None
    return checkpoint
