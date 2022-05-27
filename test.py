import borg
import numpy as np

from model.genet import Genet
from model.model_unet2 import U_net_3d_2


def test_foward_pass():
    global L, bb
    width, height, depth = 64, 64, 64
    unet3d = U_net_3d_2(width, height, depth, lr=0.01, input_ch=3)
    unet3d.load_weights('data/model_UNET4BORG.h5')
    global step_id
    chain = 0
    # setup the box
    bb = borg.forward.BoxModel()
    L = 125
    Nt = 64
    bb.L = L, L, L
    bb.N = Nt, Nt, Nt
    # Initialize some default cosmology
    cosmo = borg.cosmo.CosmologicalParameters()
    # Fiducial scale factor to express initial conditions
    z_start = 69
    a0 = 0.001
    nsteps = 1
    chain = borg.forward.ChainForwardModel(bb)
    # Add primordial fluctuations
    chain.addModel(borg.forward.models.Primordial(bb, a0))
    # Add E&Hu transfer function
    chain.addModel(borg.forward.models.EisensteinHu(bb))
    # Run an LPT model from a=0.0 to af. The ai=a0 is the scale factor at which the IC are expressed
    # lpt = borg.forward.models.BorgLpt(bb, bb, ai=a0, af=1.0)
    lpt = borg.forward.models.BorgPM(bb, bb, ai=a0, af=1.0, z_start=z_start, particle_factor=1, force_factor=2,
                                     supersampling=1, nsteps=nsteps, tCOLA=1)
    chain.addModel(lpt)
    genet = Genet(bb, lpt, unet3d)
    chain.addModel(genet)
    pos_array = np.zeros((nsteps, Nt ** 3, 3))
    step_id = 0

    def notif(t, Np, ids, poss, vels):
        global step_id
        print(f"Step {t} / {step_id}  (Np={Np})")
        pos_array[step_id, :, :] = poss
        step_id += 1

    lpt.setStepNotifier(notif, with_particles=True)
    # Set the cosmology
    chain.setCosmoParams(cosmo)
    # Generate white noise: it has to be scaled by 1/N**(3./2) to be one in Fourier
    np.random.seed(42)
    s_field = np.random.normal(0, 1, (Nt, Nt, Nt))
    ic = np.fft.rfftn(s_field / np.sqrt(Nt ** 3))
    print('np.std(ic) = ', np.std(ic))
    # ic = np.fft.rfftn(np.random.randn(Ng, Ng, Ng)/np.sqrt(Ng**3))
    delta_m = np.zeros((Nt, Nt, Nt))
    # RUN!
    chain.forwardModel_v2(ic)
    chain.getDensityFinal(delta_m)
    # Get pos
    abs_pos_COLA = np.zeros((lpt.getNumberOfParticles(), 3))
    lpt.getParticlePositions(abs_pos_COLA)
    print("Done.")
    print('delta_m = ',delta_m[0][0][0])


#test_foward_pass()
