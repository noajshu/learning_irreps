set -x -e
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
trap 'echo "\"${last_command}\" command finished with exit code $?."' EXIT


./install_deps.sh


python learn_spacetime_reps.py


python make_plots.py grouprep_learning


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


python make_plots.py nn_history
