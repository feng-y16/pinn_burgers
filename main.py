import pdb
import time
import lib.tf_silent
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.gridspec import GridSpec
from lib.pinn import PINN
from lib.gfnn import GFNN
from lib.network import Network
from lib.optimizer import Optimizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--maxiter', type=int, default=1000)
    parser.add_argument('-ntr', '--num-train-samples', type=int, default=1000)
    parser.add_argument('-nte', '--num-test-samples', type=int, default=1000)
    parser.add_argument('-n', '--network', type=str, default='pinn')
    parser.add_argument('-l', '--loss', type=str, default='l2')
    parser.add_argument('-gi', '--gradient-interval', type=int, default=100)
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    """
    Test the physics informed neural network (PINN) model for Burgers' equation
    """

    args = parse_args()
    # number of training samples
    num_train_samples = args.num_train_samples
    # number of test samples
    num_test_samples = args.num_test_samples
    # kinematic viscosity
    nu = 0.01 / np.pi

    # build a core network model
    if args.network == 'pinn':
        network = Network.build()
        network.summary()
        # build a PINN model
        model = PINN(network, nu).build()
    else:
        network = Network.build(num_outputs=2)
        network.summary()
        # build a PINN model
        model = GFNN(network, nu).build()

    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)          # t_eqn =  0 ~ +1
    tx_eqn[..., 1] = 2 * tx_eqn[..., 1] - 1                # x_eqn = -1 ~ +1
    tx_ini = 2 * np.random.rand(num_train_samples, 2) - 1  # x_ini = -1 ~ +1
    tx_ini[..., 0] = 0                                     # t_ini =  0
    tx_bnd = np.random.rand(num_train_samples, 2)          # t_bnd =  0 ~ +1
    tx_bnd[..., 1] = 2 * np.round(tx_bnd[..., 1]) - 1      # x_bnd = -1 or +1
    # create training output
    u_eqn = np.zeros((num_train_samples, 1))               # u_eqn = 0
    if args.network == 'pinn':
        u_ini = np.sin(-np.pi * tx_ini[..., 1, np.newaxis])    # u_ini = -sin(pi*x_ini)
    else:
        u_ini = -np.pi * np.cos(-np.pi * tx_ini[..., 1, np.newaxis])  # u_ini = -sin(pi*x_ini)
    u_bnd = np.zeros((num_train_samples, 1))               # u_bnd = 0

    # train the model using adam algorithm
    x_train = [tx_eqn, tx_ini, tx_bnd]
    y_train = [u_eqn,  u_ini,  u_bnd]
    optimizer = Optimizer(model=model, x_train=x_train, y_train=y_train, dict_params=args.__dict__)
    optimizer.fit()

    # predict u(t,x) distribution
    t_flat = np.linspace(0, 1, num_test_samples)
    x_flat = np.linspace(-1, 1, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = network.predict(tx, batch_size=num_test_samples)[..., -1]
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7, 4))
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u, cmap='rainbow')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0.25, 0.5, 0.75]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = network.predict(tx, batch_size=num_test_samples)[..., -1]
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    plt.savefig(os.path.join('figures', args.__dict__.__str__().replace(': ', '-') + str(time.time()) + '.png'))
    plt.close()
