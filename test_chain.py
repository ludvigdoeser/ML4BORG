import borg
import numpy as np
import pytest
import matplotlib.pyplot as plt

from jax import grad, vmap, vjp

from model.genet import Genet
from model.model_unet2 import U_net_3d_2

from utils.utils import *


@pytest.fixture
def box():
    bb = borg.forward.BoxModel()
    L = 125
    Nt = 64
    bb.L = L, L, L
    bb.N = Nt, Nt, Nt

    return bb


@pytest.fixture
def cosmo_param():
    return borg.cosmo.CosmologicalParameters()


@pytest.fixture
def a_initial():
    return 0.001


@pytest.fixture
def linear_density(box, a_initial, cosmo_param):
    np.random.seed(42)
    s_field = np.random.normal(0, 1, box.N)
    ic = np.fft.rfftn(s_field / np.sqrt(box.N[0] ** 3))
    assert np.std(ic) == pytest.approx(1, rel=1e-4)

    chain = borg.forward.ChainForwardModel(box)
    chain.addModel(borg.forward.models.Primordial(box, a_initial))
    chain.addModel(borg.forward.models.EisensteinHu(box))
    chain.setCosmoParams(cosmo_param)

    chain.forwardModel_v2(ic)

    density = np.zeros(box.N)
    chain.getDensityFinal(density)

    return density


def test_forward_pass(box, linear_density, cosmo_param):
    width, height, depth = 64, 64, 64
    unet3d = U_net_3d_2(width, height, depth, lr=0.01, input_ch=3)
    unet3d.load_weights('data/model_UNET4BORG.h5')

    # Fiducial scale factor to express initial conditions
    z_start = 69
    nsteps = 1

    lpt = borg.forward.models.BorgPM(box, box, ai=0.001, af=1.0, z_start=z_start, particle_factor=1, force_factor=2,
                                     supersampling=1, nsteps=nsteps, tCOLA=1)
    pos_array = np.zeros((nsteps, box.N[0] ** 3, 3))
    step_id = 0

    def notif(t, Np, ids, poss, vels):
        global step_id
        print(f"Step {t} / {step_id}  (Np={Np})")
        pos_array[step_id, :, :] = poss
        step_id += 1

    lpt.setStepNotifier(notif, with_particles=True)
    genet = Genet(box, lpt, unet3d)

    delta_m = np.zeros(box.N)
    corrected_delta_m = np.zeros(box.N)

    lpt.setCosmoParams(cosmo_param)
    lpt.forwardModel_v2(linear_density)
    lpt.getDensityFinal(delta_m)

    genet.forwardModel_v2(delta_m)
    genet.getDensityFinal(corrected_delta_m)

    abs_pos_COLA = np.zeros((lpt.getNumberOfParticles(), 3))
    lpt.getParticlePositions(abs_pos_COLA)

    # Convert abs_pos_COlA to jax and pass the slices
    standard_density = compute_cic(abs_pos_COLA[:, 0], abs_pos_COLA[:, 1], abs_pos_COLA[:, 2], box.L[0], box.N[0])
    # TOASK: why use the reverse-mode for the auto-diff
    jax_density, cic_grad = vjp(lambda x, y, z: jax_cic(x, y, z, *box.N + box.L), abs_pos_COLA[:, 0],
                                abs_pos_COLA[:, 1], abs_pos_COLA[:, 2])
    # FIXME: correct flipping internally and use jax
    jax_density = np.swapaxes(jax_density, 0, 2)

    # ix = np.linspace(0, box.L[0], box.N[0])
    # iy = np.linspace(0, box.L[0], box.N[0])
    # iz = np.linspace(0, box.L[0], box.N[0])
    #
    # mx, my, mz = np.meshgrid(ix, iy, iz, indexing='ij')
    # plt.pcolormesh(mx[:, 4, :], mz[:, 4, :], jax_density[:, 3, :])
    # plt.show()

    ag = np.random.uniform(size=box.N) / 10.0
    genet.adjointModel_v2(ag)
    ag_genet = np.zeros(box.N)
    genet.getAdjointModel(ag_genet)

    # TOASK: what relative tolerance is acceptable currently 1e-1 seems very large
    # FIXME: Check if voxels may be miss-aligned
    assert np.allclose(standard_density - 1, jax_density, rtol=1e-1)
