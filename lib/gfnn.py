import copy
import pdb
import numpy as np
import tensorflow as tf
from .layer import GradientLayerGFNN


class GFNN:
    """
    Build a physics informed neural network (PINN) model for Burgers' equation.

    Attributes:
        network: keras network model with input (t, x) and output the gradient of u(t, x).
        nu: kinematic viscosity.
        grads: gradient layer.
    """

    def __init__(self, network, nu):
        """
        Args:
            network: keras network model with input (t, x) and output u(t, x).
            nu: kinematic viscosity.
        """

        self.network = network
        self.nu = nu
        self.grads = GradientLayerGFNN(self.network)

    @staticmethod
    def euler_update(h_list, dh_list, dt):
        return zip_map(zip(h_list, dh_list), lambda h, dh: h + tf.cast(dt, h.dtype) * dh)

    @staticmethod
    def rk4_step(func, dt, state):
        k1 = func(state)
        k2 = func(euler_update(state, k1, dt / 2))
        k3 = func(euler_update(state, k2, dt / 2))
        k4 = func(euler_update(state, k3, dt))

        return zip_map(
            zip(state, k1, k2, k3, k4),
            lambda h, dk1, dk2, dk3, dk4:
            h + dt * (dk1 + 2 * dk2 + 2 * dk3 + dk4) / 6,
        )

    @staticmethod
    def integral(t1, t2, func):
        n = 10
        fractions = np.linspace(0, 1, n)
        df = None
        for fraction in fractions:
            t = t1 + fraction * (t2 - t1)
            temp = tf.reduce_sum(func(t) * (t2 - t1) / n, axis=-1, keepdims=True)
            if df is None:
                df = temp
            else:
                df += temp
        return df

    def build(self):
        """
        Build a PINN model for Burgers' equation.

        Returns:
            PINN model for the projectile motion with
                input: [ (t, x) relative to equation,
                         (t=0, x) relative to initial condition,
                         (t, x=bounds) relative to boundary condition ],
                output: [ u(t,x) relative to equation (must be zero),
                          u(t=0, x) relative to initial condition,
                          u(t, x=bounds) relative to boundary condition ]
        """

        # equation input: (t, x)
        tx_eqn = tf.keras.layers.Input(shape=(2,))
        # initial condition input: (t=0, x)
        tx_ini = tf.keras.layers.Input(shape=(2,))
        # boundary condition input: (t, x=-1) or (t, x=+1)
        tx_bnd = tf.keras.layers.Input(shape=(2,))

        # compute gradients
        du_dt, du_dx, d2u_dxdt, d2u_dtdx, d2u_dx2 = self.grads.__call__(tx_eqn)

        # equation output being zero
        # anchor_index = 0 if tf.random.uniform([1], maxval=1, minval=0)[0] > 0.5 else -1
        anchor_index = int(tf.random.uniform([1], maxval=tx_eqn.shape[0], minval=0)[0])
        tx_eqn_anchor = tx_eqn[anchor_index]
        du_dx_anchor = du_dx[anchor_index]
        du_dt_anchor = du_dt[anchor_index]
        d2u_dx2_anchor = d2u_dx2[anchor_index]
        du = self.integral(tx_eqn_anchor, tx_eqn, self.network)

        u_eqn = du_dt * du_dx_anchor - du_dt_anchor * du_dx + du * du_dx * du_dx_anchor \
                - self.nu * d2u_dx2 * du_dx_anchor + self.nu * d2u_dx2_anchor * du_dx
        # initial condition output
        u_ini = self.network(tx_ini)[..., 1: 2]
        # boundary condition output
        u_bnd = self.network(tx_bnd)[..., 0: 1]
        # curl
        u_curl = d2u_dxdt - d2u_dtdx

        # build the PINN model for Burgers' equation
        return tf.keras.models.Model(
            inputs=[tx_eqn, tx_ini, tx_bnd], outputs=[u_eqn, u_ini, u_bnd, u_curl])
