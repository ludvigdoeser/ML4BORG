import borg
import numpy as np
import tensorflow as tf

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

        # Pipe through NN
        pred = self.NN.predict(tf.expand_dims(DPF, axis=0)) # TODO: Get adjoint gradient for predict
        DPF = np.reshape(disp, (self.box.N[0], self.box.N[0], self.box.N[0], 3), order='F')

        # Convert back to abs pos
        abs_pos_GENET = np.reshape(DPF + initial, (self.box.N[0] ** 3, 3))
        abs_pos_GENET[abs_pos_GENET > self.box.L[0]] -= self.box.L[0]
        abs_pos_GENET[abs_pos_GENET < 0] += self.box.L[0]

        self.save = abs_pos_GENET
        # TODO: Ask if we also need to implement 'GetParticlePosition' / 'GetParticleVelocity'

    def getDensityFinal_impl(self, output_array):
        output_array[:] = compute_cic(self.save[:, 0], self.save[:, 1], self.save[:, 2], self.box.L[0], self.box.N[0])

    # Adjoint part

    def adjointModel_v2(self, input_ag):
        # TODO: adjoint gradient of cic + adjoint gradient of NN
        # self.ag = ag_nn(ag_cic(input_ag))
        # self.prev_chain.adjointModelParticles(pos, vel=np.zeros)

        self.ag = input_ag

    def getAdjointModel(self, output_ag):
        output_ag[:] = 2 * self.ag * self.save
