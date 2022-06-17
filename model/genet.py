import borg
import tensorflow as tf
from jax import vjp
import jax.numpy as jnp
import numpy as np

from utils.utils import *


class Genet(borg.forward.BaseForwardModel):
    # Constructor
    def __init__(self, box, prev_chain, NN):
        super().__init__(box, box)
        self.prev_chain = prev_chain
        self.NN = NN
        self.box = box

    # IO "preferences"
    def getPreferredInput(self):
        return borg.forward.PREFERRED_REAL

    def getPreferredOutput(self):
        return borg.forward.PREFERRED_REAL

    # Forward part
    def forwardModel_v2_impl(self, input_array):
        # Extract particle positions
        abs_pos = np.zeros((self.prev_chain.getNumberOfParticles(), 3))
        self.prev_chain.getParticlePositions(abs_pos)

        # Compute disp
        initial, disp = compute_displacement(abs_pos, self.box.L[0], self.box.N[0], order='F')
        initial = np.reshape(initial, (self.box.N[0], self.box.N[0], self.box.N[0], 3), order='F')
        DPF = np.reshape(disp, (self.box.N[0], self.box.N[0], self.box.N[0], 3), order='F')

        # Run through NN
        # Fix shape of input
        DPF_inp = tf.expand_dims(
            tf.Variable(DPF, dtype=tf.float32),
            axis=0)

        # Prep Automatic Diff.
        with tf.GradientTape() as tape:
            # x is not a tf.Variable and needs to be watched
            tape.watch(DPF_inp)
            DPF_out = self.NN(DPF_inp)

        self.NNgrad = tape.gradient(DPF_out, DPF_inp)

        self.save, self.ag_dpf = vjp(lambda d, i: self.get_abs_pos_from_dpf(d, i), DPF_out.numpy(), initial)
        # TODO: Ask if we also need to implement 'GetParticlePosition' / 'GetParticleVelocity'

    def getDensityFinal_impl(self, output_array):
        output_array[:], self.cic_grad = vjp(lambda x, y, z: jax_cic(x, y, z, *self.box.N + self.box.L),
                                             self.save[:, 0],
                                             self.save[:, 1], self.save[:, 2])
        output_array[:] = compute_cic(self.save[:, 0], self.save[:, 1], self.save[:, 2], self.box.L[0], self.box.N[0])

    # Adjoint part

    def adjointModel_v2_impl(self, input_ag):
        # input_ag is the ag of a over-density field

        # FIXME: Avoid intermediate tree map by merging self.cic_grad and self.ag_dpf
        self.ag_pos = np.asarray(self.cic_grad(input_ag)).T
        dpf_ag = np.asarray(self.ag_dpf(self.ag_pos)[0])

        # FIXME: Correct mismatch of dimensions
        # TOASK: is matmul/tensordot the correct way to propagating the adjoint gradient?
        # np.tensordot(self.NNgrad.numpy().squeeze().T, dpf_ag.squeeze())
        self.prev_chain.adjointModelParticles(ag_pos=self.ag_pos, vel=np.zeros)

    def getAdjointModel(self, output_ag):
        output_ag[:] = 0

    def get_abs_pos_from_dpf(self, dpf, initial_pos):
        DPF = jnp.reshape(dpf, (self.box.N[0], self.box.N[0], self.box.N[0], 3), order='F')
        # Convert back to abs pos
        # FIXME: clean up this mess
        abs_pos_GENET = jnp.reshape(DPF + initial_pos, (self.box.N[0] ** 3, 3))
        abs_pos_GENET = abs_pos_GENET.at[abs_pos_GENET > self.box.L[0]].set(
            abs_pos_GENET[abs_pos_GENET > self.box.L[0]] - self.box.L[0])
        abs_pos_GENET = abs_pos_GENET.at[abs_pos_GENET < 0].set(abs_pos_GENET[abs_pos_GENET < 0] + self.box.L[0])

        return jnp.array(abs_pos_GENET)
