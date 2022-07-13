# +
# +

# This script demonstrates how to do a full gradient test in python for some
# BORG model
# It also produces an output file that can be reread with the check_gradient script.
#

import sys
import os
import h5py as h5
import itertools
from scipy.special import hyp2f1
import torch
from map2map_emu.map2map.models.d2d import *
#import sys
#sys.path.append("/cfs/home/ludo4644/ML4BORG/map2map_emu/map2map/models/d2d")
#from d2d.styled_vnet import *

import jax.numpy as jnp
from jax import vjp
from jax.config import config
config.update("jax_enable_x64", True)

from utils.utils import *

try:
    import rich
    from tqdm.rich import tqdm
except:
    print("No rich support")
    from tqdm import tqdm

os.environ["PYBORG_QUIET"] = "1"
import borg
import numpy as np
import argparse

# Get the object for the console
cons = borg.console()

cons.outputToFile('log.txt')

# Make a quick printer adaptor
myprint = lambda x: cons.print_std(x) if type(x) == str else cons.print_std(repr(x))

sane_opts = {
    "LPT_CIC": {
        "a_initial": 1.0,
        "a_final": 1.0,
        "do_rsd": True,
        "supersampling": 1,
        "lightcone": False,
        "part_factor": 1.1,
    },
    "2LPT_CIC": {
        "a_initial": 1.0,
        "a_final": 1.0,
        "do_rsd": True,
        "supersampling": 1,
        "lightcone": False,
        "part_factor": 1.1,
    },
    "PM_CIC": {
        "forcesampling": 1,
        "pm_start_z": 50,
        "pm_nsteps": 5,
        "tcola": True,
        "a_initial": 1.0,
        "a_final": 1.0,
        "do_rsd": True,
        "supersampling": 1,
        "lightcone": False,
        "part_factor": 1.1,
    },
}


class emulator(borg.forward.BaseForwardModel): 
    
    # Constructor
    def __init__(self, box, prev_module, up):
        super().__init__(box, box)
        self.prev_module = prev_module
        # Since we won't return an adjoint for the density, but for positions, we need to do:
        self.prev_module.accumulateAdjoint(True)
        self.box = box
        self.use_pad = up
        
    # IO "preferences"
    def getPreferredInput(self):
        return borg.forward.PREFERRED_REAL

    def getPreferredOutput(self):
        return borg.forward.PREFERRED_REAL

    # Forward part
    def forwardModel_v2_impl(self, input_array):
        # Extract particle positions
        pos = np.zeros((self.prev_module.getNumberOfParticles(), 3)) #output shape: (N^3, 3)
        self.prev_module.getParticlePositions(pos)
        self.pos_temp = np.copy(pos)
        self.pos_out = pos
        print('pos = ',pos.dtype)
        print('self.pos_temp = ',self.pos_temp[0])
        
        if self.use_pad: 
            pos_reshaped = np.reshape(self.pos_out.T, (3,self.box.N[0],self.box.N[0],self.box.N[0]), order='C') #output shape: (3, N, N, N)
            print('in_pad type = ',pos_reshaped.dtype)
            print('in_pad shape = ',pos_reshaped.shape)
            self.pos_pad, self.ag_pad = vjp(padding, pos_reshaped) #output shape: (3, N+96, N+96, N+96)
            print('out_pad shape = ',np.shape(np.asarray(self.pos_pad)))
            print('out_pad type = ',self.pos_pad.dtype)
            
            self.pos_out = self.pos_pad
        
    def getDensityFinal_impl(self, output_array):
        # Perhaps we should move this to the function 
        output_array[:] = 0
        
    # Adjoint part
    def adjointModel_v2_impl(self, input_ag):
        # input_ag is the ag of a over-density field
        print('input_ag.shape = ',input_ag.shape)
        print('input_ag.dtype = ',input_ag.dtype)
        
        print('Computing the loss')
        self.Loss, self.agL = vjp(loss,self.pos_out)
        
        if self.use_pad:
            print('Use pad')
            ag = np.asarray(self.agL(1.0))[0]
            print('ag.shape = ',ag.shape)
            ag = np.asarray(self.ag_pad(ag))[0]
            print('ag.shape = ',ag.shape)
            
            #ag = (ag+np.zeros(np.shape(ag)))/1 
            
            self.ag_pos = ag.reshape(3,self.box.N[0]**3,order='C').T
            
            self.ag_pos = (self.ag_pos+np.zeros(np.shape(self.ag_pos)))/1 

    def getAdjointModel_impl(self, output_ag):
        output_ag[:] = 0
        
        print('hejhej')
       
        if not self.use_pad:
            input_ag = np.asarray(self.agL(1.0))[0]
            print('input_ag = ',input_ag[0])
            print('pos_out = ',np.array(self.pos_out, dtype=np.float64)[0])
            
            #self.prev_module.adjointModelParticles(ag_pos=input_ag, ag_vel=np.zeros_like(self.pos_out, dtype=np.float64))
            
            # Is the same as setting the following when using a Gaussian loss with zero mean
            self.prev_module.adjointModelParticles(ag_pos=np.array(self.pos_out, dtype=np.float64), ag_vel=np.zeros_like(self.pos_out, dtype=np.float64))
        else:
            print('Get Adjoint for pad')

            print('self.ag_pos = ',self.ag_pos[:5])
            
            #self.ag_pos = self.ag_pos % self.box.L[0]
            
            self.prev_module.adjointModelParticles(ag_pos=self.ag_pos, ag_vel=np.zeros_like(self.ag_pos, dtype=np.float64))
            
        
def loss(out):
    global truth
    #print('truth = ',truth[0])
    #return jnp.sum(0.5*(out-truth)**2)
    #return jnp.sum(0.5*(out[64,64,64])**2)
    """
    mask = jnp.zeros_like(out)    
    mask = mask.at[64,64,64].set(1)
    print('np.sum(out*mask) = ',np.sum(out*mask))
    """
    return jnp.sum(0.5*(out)**2)

def padding(x):
    return jnp.pad(x,((0,0),(48,48),(48,48),(48,48)),'wrap')

def pick_out(pos):
    mask = jnp.zeros_like(pos)    
    mask = mask.at[1056832,0].set(1)
    print('pos[1056832,0] = ',pos[1056832,0])
    print('np.sum(pos*mask) = ',np.sum(pos*mask))
    return pos*mask
        
    
def build_gravity_model_test(box, cpar, name, user_opts, use_emu, up, v3=False, requires_grad=True):
    
    chain = 0 
    bb = box

    # Initialize some default cosmology
    cosmo = borg.cosmo.CosmologicalParameters()

    # Fiducial scale factor to express initial conditions
    z_start = 69
    a0 = 1

    chain = borg.forward.ChainForwardModel(bb)

    # Add fluctuations and transfer
    chain.addModel(borg.forward.models.Primordial(bb, a0)) # Add primordial fluctuations    
    chain.addModel(borg.forward.models.EisensteinHu(bb)) # Add E&Hu transfer function

    # Run LPT model from a=0.0 to af. The ai=a0 is the scale factor at which the IC are expressed
    lpt = borg.forward.models.BorgLpt(bb, bb, ai=a0, af=1.0)    
    chain.addModel(lpt)
   
    print('Adding emulator to the chain')

    # Create module in BORG chain
    emu = emulator(bb, lpt, up)
    chain.addModel(emu)
    
    # Set cosmology
    chain.setCosmoParams(cosmo)
    
    return chain, emu

# Check that we are not running inside hades_python and that
# this is running as a script and not through an import.
if not borg.EMBEDDED and __name__ == "__main__":
    global truth 
    
    parser = argparse.ArgumentParser(prog="run_gradient_test")
    parser.add_argument("model", type=str, nargs="?", default=None)
    parser.add_argument("--Ngrid", type=int, default=8)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--emu", action="store_true", default=False)
    parser.add_argument("--fd", action="store_false", default=True)
    parser.add_argument("--use_pad", action="store_true", default=False)
    parser.add_argument("--test_points", type=int, default=4)
    parser.add_argument("--list", action="store_true", default=False)
    parser.add_argument("--opt", action="append", default=[], type=str)
    parser.add_argument("--v3", action="store_true", default=False)

    args = parser.parse_args()

    if args.list:
        print(" ".join(sane_opts.keys()))
        sys.exit(0)

    model_opts = {}
    for o in args.opt:
        key, val = o.split("=", maxsplit=1)
        model_opts[key] = val

    if args.model is None:
        parser.print_usage()
        sys.exit(1)

    # Run FD too?
    fd = args.fd
    print('Run finite diff: ',fd)
    
    up = args.use_pad
    if up:
        print('Use pad')
        
    # Grid size
    Ng = args.Ngrid
    print('Ng = ',Ng)
    # Box size
    L = args.L
    print('L = ',L)
    tP = args.test_points
    print('Test points = ',tP)
    # Use emulator
    use_emu = args.emu
    print('use_emu =',use_emu)
    
    ## ------
    
    # Create the auxiliary objects
    # Cosmology and box
    cpar = borg.cosmo.CosmologicalParameters()
    box = borg.forward.BoxModel(L, Ng)

    # Get the forward model
    fwd, emu = build_gravity_model_test(box, cpar, args.model, model_opts, use_emu, up, v3=args.v3)
   
    # Generate some fiducial point where to compute the gradient
    np.random.seed(42)
    s = np.random.randn(Ng, Ng, Ng)
    s_hat = np.fft.rfftn(s / Ng ** (1.5))
    print('s_hat.dtype = ',s_hat.dtype)
    
    """
    s_true = np.copy(s) 
    perturb = 0.0001
    s_true = np.cos(perturb)*s_true + np.sin(perturb)*np.random.normal(0., 1., (Ng,Ng,Ng))
    s_hat_true = np.fft.rfftn(s_true / Ng ** (1.5))
    
    fwd.forwardModel_v2(s_hat_true)
    #truth = emu.pos_out
    truth = 0
    """
    
    def run_forward(s_hat):
        # RUN!
        fwd.forwardModel_v2(s_hat)
        return emu.pos_out
    
    # Compute a log-likelihood
    def dfield_like(s_hat):
        #print('truth = ',truth[0])
        #return 0.5 * (np.array(run_forward(s_hat), copy=False) ** 2).sum()
        out = run_forward(s_hat)
        
        lo = loss(out)
        print('L = ',lo)
        return lo

    # Compute the adjoint-gradient of the log-likelihood
    def dfield_ag(s_hat):
        fwd.forwardModel_v2(s_hat)
        
        dlogL_drho = np.zeros(fwd.getOutputBoxModel().N)
        # Here fill up dlogL_drho from the gradient of the likelihood
        fwd.adjointModel_v2(dlogL_drho)
        
        print('Then getAdjointModel')
        
        ag = np.zeros((Ng, Ng, Ng // 2 + 1), dtype=np.complex128)
        fwd.getAdjointModel(ag)
        
        return np.array(ag)

    myprint("Running adjoint")

    # Prepare and compute the adjoint-gradient
    num_gradient = np.zeros((Ng, Ng, Ng // 2 + 1), dtype=np.complex128)
    s_hat_epsilon = s_hat.copy()
    analytic_gradient = dfield_ag(s_hat)

    # By how much to perturb the initial conditions
    epsilon = args.eps
    print('epsilon = ',epsilon)
    
    if fd:

        # Loop over one mode
        for i, j, k in tqdm(
            itertools.product(*map(range, [tP,1,1 // 2 + 1])),
            total=tP * 1 * (1 // 2 + 1),
        ):
            print('i,j,k = ',i,j,k)
            # Perturb
            s_hat_epsilon[i, j, k] = s_hat[i, j, k] + epsilon
            
            ## ------------------------------------------------------------
            ## TEST HOW MUCH DIFF BETWEEN POSITIONS...
            fwd.forwardModel_v2(np.copy(s_hat))
            test1 = emu.pos_temp 
            fwd.forwardModel_v2(np.copy(s_hat_epsilon))
            test1 -= emu.pos_temp 
            print('np.shape(test1) = ',np.shape(test1))
            print(test1)
            np.save(f'test1_{i}',test1)
            print('Diff in pos = ',np.sum(test1,axis=0))
            print('Where non-zero diff in pos = ',test1[np.where(test1>1e-6)])
            try:
                print('np.amax(pos_diff) = ',np.amax(test1[np.where(test1>1e-6)]))
            except ValueError:
                pass
            print('Where non-zero diff in pos = ',test1[np.nonzero(test1)])
            print('len(non-zero) = ',len(test1[np.where(test1>1e-6)]))
            print('len(non-zero) = ',len(test1[np.nonzero(test1)]))
            ## ------------------------------------------------------------
            
            # Compute log-likelihood
            L_old = L = dfield_like(s_hat_epsilon)
            # Re-perturb
            s_hat_epsilon[i, j, k] = s_hat[i, j, k] - epsilon
            # Recompute
            L -= dfield_like(s_hat_epsilon)
            # Centered numerical gradient
            QQ = L / (2.0 * epsilon)

            # ... and the same but the imaginary part
            s_hat_epsilon[i, j, k] = s_hat[i, j, k] + 1j * epsilon
            L = dfield_like(s_hat_epsilon)
            s_hat_epsilon[i, j, k] = s_hat[i, j, k] - 1j * epsilon
            L -= dfield_like(s_hat_epsilon)
            QQ = QQ + L * 1j / (2.0 * epsilon)

            s_hat_epsilon[i, j, k] = s_hat[i, j, k]

            num_gradient[i, j, k] = QQ
    
    with h5.File("dump.h5", mode="w") as ff:
        ff["scalars/gradient_array_lh"] = analytic_gradient
        ff["scalars/gradient_array_lh_ref"] = num_gradient
        ff["scalars/gradient_array_prior"] = np.zeros_like(analytic_gradient)
        ff["scalars/gradient_array_prior_ref"] = np.zeros_like(analytic_gradient)
