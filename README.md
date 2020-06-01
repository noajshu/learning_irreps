

# learning_irreps
Learning irreducible representations of noncommutative Lie groups, applied to constructing group equivariant neural networks.


![4D "Lorentz harmonics", analogues of the "spherical harmonic" functions](figs/poincare_lorentz_harmonics.gif | width=256)
The tensor product of two 4D irreps of $\text{SO}(3,1)$ decomposes as the direct sum of a 1D representation (known in physics as the "spacetime interval"), 9D irrep, and 2 isomorphic 3D irreps. I.e., $4 \otimes 4 = 1 \oplus 3 \oplus 3 \oplus 9$.

Let $u = (u_1, ..., u_9)$ denote the 9D irrep decomposed from the tensor product $v \otimes v$ where $v = (t,x,y,z)$ is a spacetime 4-vector. The animated figure above depicts the 9 components of $u$ as a function of the components of $v$. Since $v$ is 4D, we choose to make it animated, with each frame plotting v within fixed-time slices of $\mathbb{R}^4$.

<!-- ![Alt Text](figs/boosted_activations.gif)
The figure above shows: (left) an MNIST live digit in the xt plane, under the action of a Lorentz boost in the x direction; (right) all activations of a layer of an SO(1,1)-equivariant LAN (Lie Algebraic Network) as the boost is applied. Though the activations are comprised of fields of representations (more formally, [a section of a vector bundle associated to a principal bundle](https://papers.nips.cc/paper/9114-a-general-theory-of-equivariant-cnns-on-homogeneous-spaces.pdf)), they are plotted here as vectors with tails at the origin to make it easier to see the covariance. -->



## Introduction
This document is a guide for reproducing the results presented in our submission by using the source code included in the file `learning_irreps.zip`.

Please see the main submission for all theoretical background and definitions.

We intend to publish the included source code on GitHub after the review process is complete.

## Dependencies
Please ensure `wget` is installed and available.

Please create a Python 3.7 environment. We suggest using `pip` to manage dependencies.

Run the script `install_deps.sh`. This will use `pip` to install all needed dependencies. The `requirements.txt` file is incomplete due to our use of one package that is hosted as a github repository. Please see `install_deps.sh` for details.

We use the PyTorch deep learning library \cite{pytorch}.


## Shortcut Script
The script
```
reproduce_paper.sh
```
will automatically run through the tasks outlined in this document. These tasks are described individually in the following sections.

## Learning Irreducible Representations (GroupReps)
This experiment may be reproduced by running:
```
python learn_spacetime_reps.py
```
This takes about 10 minutes on a 1.4 GHz Dual-Core Intel Core i7 CPU\footnote{We recommend using CPU for learning the GroupReps as 64 bit floating point arithmetic is used. Once the GroupReps are learned to high precision they may be used to build equivariant networks of lower (e.g. 32 bit) precision.}.

This uses random initialization points so the total time required may fluctuate, but in practice it rarely takes longer than 15 minutes.

After the GroupReps are learned they are stored in the `numpy` data file `irreps.npy`. You are of course free to inspect the contents manually verify that the matrices satisfy the appropriate commutation relations. However we suggest instead using our utilities to plot the tensor product structure and loss vs. iteration:
```
python make_plots.py grouprep_learning
```
This will produce seven files total inside the `plots/` directory. The first three files have names `learning_$Nd_representations_$ALGEBRA_NAME.pdf` in which `$N` is the dimension of the GroupRep and `$ALGEBRA_NAME` is one of $\text{SO}(3), \text{SO}(2,1),$ or $\text{SO}(3,1)$. These correspond to the plots in Figure 3 of the main submission.

The remaining four plots are at paths
`tensor_product_decomposition_svd_$REP_$ALGEBRA_NAME.pdf`
in which `$ALGEBRA_NAME` is as above and `$REP` is the identity of the GroupRep. Primed GroupReps are those learned by gradient descent, while unprimed GroupReps come from formulas, as explained in the submission.

## Generating MNIST-Live Datasets
The program `mnist_live/make_data.py` makes datasets. The command line arguments are somewhat self-explanatory. Please run the following commands to generate the datasets we used to train Poincar\'e-equivariant neural networks:
```
python mnist_live/make_data.py \
    --included-classes='[0,9]' \
    --ndim 2 \
    --plane xy \
    --fname mnist_live__xy_plane_2D.npy

python mnist_live/make_data.py \
    --included-classes='[0,9]' \
    --ndim 3 \
    --plane xy \
    --fname mnist_live__xy_plane_3D.npy
```


## Training Poincar\'e-equivariant Neural Networks
The program `spacetime_nn.py` can train neural networks which are equivariant to the groups $\text{SO}(2,1)$ and $\text{SO}(3,1)$ using GroupReps obtained from formulas or learned through gradient descent (the latter are referred to in the code as "gd" reps).
Please run the following command
```
python spacetime_nn.py \
    --additional-args-json='{"group": "SO(2,1)", "data_file": "mnist_live__xy_plane_2D.npy", "train_size": 4096, "dev_size": 124, "rep_source": "tensor_power_gd"}' \
    --model-kwargs-json='{"num_channels":3,"num_layers":3}' \
    --skip-equivariance-test \
    --checkpoint='checkpoint_SO21_xy_plane.tar' \
    --epochs 2 --batch-size 16 \
    --checkpoint-on-batch=20 \
    --plot-to='training_plot_SO21_xy_plane.pdf'


python spacetime_nn.py \
    --additional-args-json='{"group": "SO(3,1)", "data_file": "mnist_live__xy_plane_3D.npy", "train_size": 4096, "dev_size": 124, "rep_source": "tensor_power_gd"}' \
    --model-kwargs-json='{"num_channels":3,"num_layers":3}' \
    --skip-equivariance-test \
    --checkpoint='checkpoint_SO31_xy_plane.tar' \
    --epochs 2 --batch-size 16 \
    --checkpoint-on-batch=20 \
    --plot-to='training_plot_SO31_xy_plane.pdf'
```

As a first step in setting up the equivariant networks, this program will solve for the Clebsch-Gordan coefficients as described in our paper. This may take some time due to our use of a randomized algorithm to compute the coefficients. Error messages of the form "Encountered error with CG coeffs..." may safely be ignored, as the algorithm will automatically retry until succeeding. Once the coefficients are obtained they will be saved with the model checkpoints for future use.

The models will checkpoint to `checkpoint_SO21_xy_plane.tar` and `checkpoint_SO31_xy_plane.tar` as set in the command line arguments.


To plot the training performance etc., please run
```
python make_plots.py nn_history
```
The plotted neural network training history is in `plots/checkpoint_$NETWORK.pdf` where `$NETWORK` indicates the model type. This is how we produce the plots for Figure 5 of the submission.
This will also print the total accuracy on the held-out test set. We obtain accuracies of $0.81195$ and $0.827171$ for the $\text{SO}(2,1)$- and $\text{SO}(3,1)$- equivariant networks, respectively.
