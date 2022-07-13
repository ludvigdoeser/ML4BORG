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
import time

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

def check(disp,L,moved_over_bound,max_disp_1d,i,axis):
    idxsup900 = disp[:,i]>moved_over_bound
    idx100 = np.abs(disp[:,i])<max_disp_1d
    idxsubminus900 = disp[:,i]<-moved_over_bound

    sup900 = len(disp[:,i][idxsup900])
    within100 = len(disp[:,i][idx100])
    subminus900 = len(disp[:,i][idxsubminus900])

    """
    print(f'Disp in {axis[i]} direction under -{moved_over_bound} Mpc/h is = ', subminus900)
    print(f'|Disp| in {axis[i]} direction under {max_disp_1d} Mpc/h is = ', within100)
    print(f'Disp in {axis[i]} direction over {moved_over_bound} Mpc/h is = ', sup900)
    print('These add up to: ', subminus900+within100+sup900)
    print('\n')
    """
    
    assert subminus900+within100+sup900 == len(disp[:,i])
    
    return idxsup900, idxsubminus900
    
def correct_displacement_over_periodic_boundaries(disp,L,max_disp_1d=100):
    # Need to correct for positions moving over the periodic boundary

    axis = ['x','y','z']
    moved_over_bound = L - max_disp_1d
    
    for i in [0,1,2]:

        #print(f'Before correcting {axis[i]} direction: ')
        #print('len(disp[:,i]) = ',len(disp[:,i]))

        idxsup900, idxsubminus900 = check(disp,L,moved_over_bound,max_disp_1d,i,axis)

        # Correct positions
        disp[:,i][idxsup900] -= L
        disp[:,i][idxsubminus900] += L

        #print(f'After correcting {axis[i]} direction: ')
        _, _ = check(disp,L,moved_over_bound,max_disp_1d,i,axis)

        assert np.amin(disp[:,i]) >= -max_disp_1d and np.amax(disp[:,i]) <= max_disp_1d
        
    return disp
    
def test_correct_pos(pos,pos_reshaped):
    # The first particle starts at 0,0,0
    # The second particle starts at 0,0,dx
    # The thrid part starts at 0,0,2*dx etc
    tol = 1e-15
    for i in range(0,10):
        assert np.sum(pos[i,:] - pos_reshaped[:,0,0,i]) < tol
        

def dis(x, undo=False, z=0.0, dis_std=6.0, **kwargs):
    dis_norm = dis_std * D(z)  # [Mpc/h]

    if not undo:
        dis_norm = 1 / dis_norm

    x *= dis_norm
    
def vel(x, undo=False, z=0.0, dis_std=6.0, **kwargs):
    vel_norm = dis_std * D(z) * H(z) * f(z) / (1 + z)  # [km/s]

    if not undo:
        vel_norm = 1 / vel_norm

    x *= vel_norm


def D(z, Om=0.31):
    """linear growth function for flat LambdaCDM, normalized to 1 at redshift zero
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return a * hyp2f1(1, 1/3, 11/6, - OL * a**3 / Om) \
             / hyp2f1(1, 1/3, 11/6, - OL / Om)

def f(z, Om=0.31):
    """linear growth rate for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    aa3 = OL * a**3 / Om
    return 1 - 6/11*aa3 * hyp2f1(2, 4/3, 17/6, -aa3) \
                        / hyp2f1(1, 1/3, 11/6, -aa3)

def H(z, Om=0.31):
    """Hubble in [h km/s/Mpc] for flat LambdaCDM
    """
    OL = 1 - Om
    a = 1 / (1+z)
    return 100 * np.sqrt(Om / a**3 + OL)

class emulator(borg.forward.BaseForwardModel): 
    # Constructor
    def __init__(self, box, prev_module, NN, Om, requires_grad, upan, use_float64):
        super().__init__(box, box)
        self.prev_module = prev_module
        # Since we won't return an adjoint for the density, but for positions, we need to do:
        self.prev_module.accumulateAdjoint(True)
        self.NN = NN
        self.box = box
        self.Om = Om
        self.requires_grad = requires_grad
        self.use_pad_and_NN = upan
        self.use_float64 = use_float64

    # IO "preferences"
    def getPreferredInput(self):
        return borg.forward.PREFERRED_REAL

    def getPreferredOutput(self):
        return borg.forward.PREFERRED_REAL

    # Forward part
    def forwardModel_v2_impl(self, input_array):
        start_time = time.time()
        
        # Step 0 - Extract particle positions
        pos = np.zeros((self.prev_module.getNumberOfParticles(), 3)) #output shape: (N^3, 3)
        self.prev_module.getParticlePositions(pos)
        #print('pos = ',pos.dtype)
        temp_time = time.time()
        print("Step 0 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # Step 1 - find displacements
        #q, disp = compute_displacement(pos,self.box.L[0], self.box.N[0], order='F') #output shapes: (N^3, 3)
        q = np.load('data/q_initial_L128N250.npy')
        disp = pos - q
        #print('disp type = ',disp.dtype)
        temp_time = time.time()
        print("Step 1 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # Step 2 - correct for particles that moved over the periodic boundary
        disp_temp = correct_displacement_over_periodic_boundaries(disp,L=self.box.L[0],max_disp_1d=100)
        #print('disp_temp type = ',disp_temp.dtype)
        temp_time = time.time()
        print("Step 2 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # Step 3 - reshaping initial pos and displacement
        # not sure why order='C' is working here... not sure if it matters... could change it below
        q_reshaped = np.reshape(q.T, (3,self.box.N[0],self.box.N[0],self.box.N[0]), order='C') #output shape: (3, N, N, N)
        dis_in = np.reshape(disp_temp.T, (3,self.box.N[0],self.box.N[0],self.box.N[0]), order='C') #output shape: (3, N, N, N)
        temp_time = time.time()
        print("Step 3 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # Step 4 - normalize
        #print('dis_in[:,0,0,0] = ',dis_in[:,0,0,0])
        dis(dis_in)
        #print('dis_in[:,0,0,0] = ',dis_in[:,0,0,0])
        temp_time = time.time()
        print("Step 4 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        if self.use_pad_and_NN:
            if not self.use_float64:
                dis_in = dis_in.astype(np.float32)
                temp_time = time.time()
                print("Step 4.5 of forward pass took %s seconds" % (temp_time - start_time))
                start_time = temp_time
                
            # Step 5 - padding to (3,N+48*2,N+48*2,N+48*2)
            #print('in_pad type = ',dis_in.dtype)
            #print('in_pad shape = ',dis_in.shape)
            dis_in_padded, self.ag_pad = vjp(self.padding, dis_in) #output shape: (3, N+96, N+96, N+96)
            #print('out_pad shape = ',np.shape(np.asarray(dis_in_padded)))
            #print('out_pad type = ',dis_in_padded.dtype)
            temp_time = time.time()
            print("Step 5 of forward pass took %s seconds" % (temp_time - start_time))
            start_time = temp_time
            
            # Step 6 - turn into a pytorch tensor (unsquueze because batch = 1)
            if self.use_float64:
                self.x = torch.unsqueeze(torch.tensor(np.asarray(dis_in_padded),dtype=torch.float64, requires_grad=self.requires_grad),dim=0) #output shape: (1, 3, N+96, N+96, N+96)
            else:
                self.x = torch.unsqueeze(torch.tensor(np.asarray(dis_in_padded),dtype=torch.float32, requires_grad=self.requires_grad),dim=0) #output shape: (1, 3, N+96, N+96, N+96)
            #print('Om.dtype = ',self.Om.dtype)
            #print('x.dtype = ',self.x.dtype)
            temp_time = time.time()
            print("Step 6 of forward pass took %s seconds" % (temp_time - start_time))
            start_time = temp_time
            
            # Step 7 - Pipe through emulator  
            if self.requires_grad:
                self.y = self.NN(self.x,self.Om) #output shape: (1, 3, N, N, N)
            else:
                with torch.no_grad():
                    self.y = self.NN(self.x,self.Om) #output shape: (1, 3, N, N, N)
            #print('self.y.dtype = ',self.y.dtype)
            temp_time = time.time()
            print("Step 7 of forward pass took %s seconds" % (temp_time - start_time))
            start_time = temp_time
            
            # Step 8 - N-body sim displacement 
            dis_out = torch.squeeze(self.y).detach().numpy() #output shape: (3, N, N, N)
            #print('dis_out.dtype =',dis_out.dtype)
            temp_time = time.time()
            print("Step 8 of forward pass took %s seconds" % (temp_time - start_time))
            start_time = temp_time
        else:
            dis_out = dis_in
        
        # Step 9 - undo the normalization
        #print('dis_out[:,0,0,0] = ',dis_out[:,0,0,0])
        dis(dis_out,undo=True)
        temp_time = time.time()
        print("Step 9 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        #print('dis_out[:,0,0,0] = ',dis_out[:,0,0,0])
        
        # Step 10 - convert displacement into positions
        pos = dis_out + q_reshaped
        temp_time = time.time()
        print("Step 10 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # Step 11 - make sure everything within the box
        pos[pos>self.box.L[0]] -= self.box.L[0]
        pos[pos<0] += self.box.L[0]
        temp_time = time.time()
        print("Step 11 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # Step 12 - reshape positions 
        #print('step12, inshape = ',pos.shape)
        self.pos_out = pos.reshape(3,self.box.N[0]**3,order='C').T #output shape: (N^3, 3)
        #print('step12, outshape = ',self.pos_out.shape)
        #print('step12, dtype = ',self.pos_out.dtype)
        temp_time = time.time()
        print("Step 12 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # Step 13 - CIC
        self.dens_out, self.cic_grad = vjp(lambda x, y, z: jax_cic(x, y, z, *self.box.N + self.box.L),
                                             self.pos_out[:, 0],
                                             self.pos_out[:, 1], 
                                             self.pos_out[:, 2])
        temp_time = time.time()
        print("Step 13 of forward pass took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        #self.pos_out = pos
        
    def getDensityFinal_impl(self, output_array):
        # Perhaps we should move this to the function 
        
        output_array[:] =  self.dens_out
        
        
    # Adjoint part
    def adjointModel_v2_impl(self, input_ag):
        # input_ag is the ag of a over-density field
        #print('input_ag.shape = ',input_ag.shape)
        #print('input_ag.dtype = ',input_ag.dtype)
        
        #input_ag = input_ag.T
        #print('Transposed the input adjoint')
        start_time = time.time()
        
        ag = input_ag
        #print('input_ag = ',input_ag)
       
        #print('Do CIC grad')
        # reverse step 13 (CIC)
        ag = np.asarray(cic_analytical_grad(ag,self.pos_out[:,0],self.pos_out[:,1],self.pos_out[:,2],128,128,128,250,250,250)) 
        #ag = np.asarray(self.cic_grad(ag)) #test: might have to flip?
        #print('ag_pos_out.shape = ',ag.shape)
        #print(ag)
        temp_time = time.time()
        print("Reverse step 13 took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        #print('Need to transpose!')
        # reverse step 11
        ag = np.reshape(ag.T, (3,self.box.N[0],self.box.N[0],self.box.N[0]), order='C')
        #print('ag_pos_out.shape = ',ag.shape)
        #print(ag)
        temp_time = time.time()
        print("Reverse step 11 took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # reverse step 9
        #print('ag_pos_out[:,0,0,0] = ',ag[:,0,0,0])
        dis(ag,undo=False)
        #print('ag_pos_out[:,0,0,0] = ',ag[:,0,0,0])
        temp_time = time.time()
        print("Reverse step 9 took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        if self.use_pad_and_NN:
        
            # reverse step 8
            if self.use_float64:
                ag = torch.unsqueeze(torch.tensor(ag,dtype=torch.float64),dim=0)
            else:
                ag = torch.unsqueeze(torch.tensor(ag,dtype=torch.float32),dim=0)
            #print('ag_dis_out.shape = ',ag.shape)
            #print('ag.dtype = ',ag.dtype)
            temp_time = time.time()
            print("Reverse step 8 took %s seconds" % (temp_time - start_time))
            start_time = temp_time
            
            # reverse step 7
            ag = torch.autograd.grad(self.y, self.x, grad_outputs=ag, retain_graph=False)[0] #TODO: get rid off retain? , retain_graph=True
            #print('ag from NN shape = ',ag.shape)
            #print('ag.dtype = ',ag.dtype)
            temp_time = time.time()
            print("Reverse step 7 took %s seconds" % (temp_time - start_time))
            start_time = temp_time
        
             # reverse step 6
            ag = torch.squeeze(ag).detach().numpy()
            #print('ag from NN shape = ',ag.shape)
            #print('ag.dtype = ',ag.dtype)
            temp_time = time.time()
            print("Reverse step 6 took %s seconds" % (temp_time - start_time))
            start_time = temp_time
        
            # reverse step 5
            ag = np.asarray(self.ag_pad(ag))[0] #not sure why adjoint outputs shape (1,3,128,128,128)
            #print('ag_padded = ',ag.shape)
            #print('ag.dtype = ',ag.dtype)
            temp_time = time.time()
            print("Reverse step 5 took %s seconds" % (temp_time - start_time))
            start_time = temp_time
            
            # magic... 
            #ag = (ag+np.zeros(np.shape(ag)))/1 
        
        # reverse step 4
        #print('ag_padded[:,0,0,0] = ',ag[:,0,0,0])
        dis(ag,undo=True)
        #print('ag_padded[:,0,0,0] = ',ag[:,0,0,0])
        temp_time = time.time()
        print("Reverse step 4 took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        # reverse step 3
        self.ag_pos = ag.reshape(3,self.box.N[0]**3,order='C').T
        #print('self.ag_pos shape = ',self.ag_pos.shape)
        #print('self.ag_pos type = ',self.ag_pos.dtype)
        temp_time = time.time()
        print("Reverse step 3 took %s seconds" % (temp_time - start_time))
        start_time = temp_time
        
        #self.ag_pos = ag
        #print('self.ag_pos.shape = ',self.ag_pos.shape)
        
        # magic... 
        self.ag_pos = (self.ag_pos+np.zeros(np.shape(self.ag_pos)))/1
        
        #print('final ag = ',self.ag_pos[:4,0])
        
    def getAdjointModel_impl(self, output_ag):
        # TOASK: do we want this?? will only be the ag for positions
        #print('output_ag.shape = ',output_ag.shape)
        #print('self.ag_pos.shape = ',self.ag_pos.shape)
        #output_ag[:] = np.array(self.ag_pos, dtype=np.float64)
        
        #output_ag[:] = self.input_ag
        
        #print('output_ag.shape = ',output_ag.shape)
        output_ag[:] = 0
        #print('output_ag.shape = ',output_ag.shape)
        
        #print('Set self.prev_module.adjointModelParticles(ag_pos=np.array(self.ag_pos, dtype=np.float64), ag_vel=np.zeros_like(self.ag_pos, dtype=np.float64)) with self.ag_pos.shape = ',self.ag_pos.shape)
        #print('np.array(self.ag_pos, dtype=np.float64) = ',np.array(self.ag_pos, dtype=np.float64))
        if self.use_float64:
            self.prev_module.adjointModelParticles(ag_pos=np.array(self.ag_pos, dtype=np.float64), ag_vel=np.zeros_like(self.ag_pos, dtype=np.float64))
        else:
            self.prev_module.adjointModelParticles(ag_pos=np.array(self.ag_pos, dtype=np.float32), ag_vel=np.zeros_like(self.ag_pos, dtype=np.float32))
        

    def padding(self,x):
        return jnp.pad(x,((0,0),(48,48),(48,48),(48,48)),'wrap')

# Gravity model builder. This example may be expanded later
# to show case other easy to build model.
def build_gravity_model(box, cpar, name, user_opts, v3=False):
    opts = sane_opts.get(name, {})
    opts.update(user_opts)
    if not v3:
        lpt = borg.forward.models.newModel(name, box, opts)
    else:
        lpt = borg.forward.models.newModel_v3(name, opts=opts)
    if name == "SphereProjection":
        lpt.setModelParams({"lensing_sources": np.array([[0.1, 0.1]])})
    chain = borg.buildDefaultChain(box, cpar, 1.0, lpt)
    return chain, lpt

def build_gravity_model_test(box, cpar, name, user_opts, use_emu, upan, use_float64, v3=False, requires_grad=True):
    
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
    
    if use_emu:
        print('Adding emulator to the chain')
        
        # Load weights
        f = '/cfs/home/ludo4644/ML4BORG/map2map_emu/map2map/weights/d2d_weights.pt'
        emu_weights = torch.load(f,map_location=torch.device('cpu'))

        # Initialize model
        model = StyledVNet(1,3,3)
        model.load_state_dict(emu_weights['model'])
        if use_float64:
            model.double()

        # Extract omega as style param
        if use_float64:
            Om = torch.tensor([cosmo.omega_m],dtype=torch.float64) # style parameter
        else:
            Om = torch.tensor([cosmo.omega_m],dtype=torch.float32) # style parameter
        # from emulator hacking:
        Om -= torch.tensor([0.3])
        Om *= torch.tensor([5.0])
        
        # Create module in BORG chain
        emu = emulator(bb, lpt, model, Om, requires_grad, upan, use_float64)
        chain.addModel(emu)
    
    # Set cosmology
    chain.setCosmoParams(cosmo)
    
    return chain, lpt

# Check that we are not running inside hades_python and that
# this is running as a script and not through an import.
if not borg.EMBEDDED and __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run_gradient_test")
    parser.add_argument("model", type=str, nargs="?", default=None)
    parser.add_argument("--write_to_file", type=str, nargs="?", default=None)
    parser.add_argument("--Ngrid", type=int, default=8)
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--emu", action="store_true", default=False)
    parser.add_argument("--fd", action="store_false", default=True)
    parser.add_argument("--use_float64", action="store_true", default=False)
    parser.add_argument("--use_pad_and_NN", action="store_true", default=False)
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
    upan = args.use_pad_and_NN
    print('Use pad and NN: ',upan)
    use_float64 = args.use_float64
    print('Use float64: ',use_float64)
    
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
    
    
    # Create the auxiliary objects
    # Cosmology and box
    cpar = borg.cosmo.CosmologicalParameters()
    box = borg.forward.BoxModel(L, Ng)

    # Get the forward model
    fwd, lpt = build_gravity_model_test(box, cpar, args.model, model_opts, use_emu, upan, use_float64, v3=args.v3)
    # For finite diff we don't need to store the gradients
    if fd:
        fwd_FD, _ = build_gravity_model_test(box, cpar, args.model, model_opts, use_emu, upan, use_float64, v3=args.v3, requires_grad=False)

    # Generate some fiducial point where to compute the gradient
    np.random.seed(42)
    s_hat = np.fft.rfftn(np.random.randn(Ng, Ng, Ng) / Ng ** (1.5))
    print('s_hat.dtype = ',s_hat.dtype)
    
    def run_forward(s_hat):
        start_time = time.time()
        fwd.forwardModel_v3(
            borg.modelio.GInput(borg.modelio.newInputModelIO(box, s_hat))
        )
        rho_desc = fwd.getOutputDescription().makeTemporaryForward()
        rho_out = fwd.getResultForward_v3(borg.modelio.GOutput(rho_desc))
        rho_desc = rho_out.disown()
        print("--- Full forward pass took %s seconds ---" % (time.time() - start_time))
        return rho_desc
    
    def run_forward_FD(s_hat):
        fwd_FD.forwardModel_v3(
            borg.modelio.GInput(borg.modelio.newInputModelIO(box, s_hat))
        )
        rho_desc = fwd_FD.getOutputDescription().makeTemporaryForward()
        rho_out = fwd_FD.getResultForward_v3(borg.modelio.GOutput(rho_desc))
        rho_desc = rho_out.disown()
        return rho_desc

    # Compute a log-likelihood
    def dfield_like(s_hat):
        return 0.5 * (np.array(run_forward_FD(s_hat), copy=False) ** 2).sum()

    # Compute the adjoint-gradient of the log-likelihood
    def dfield_ag(s_hat):
        start_time = time.time()
        # The derivative of square is 2 * vector
        dgrid = run_forward(s_hat)
        if hasattr(dgrid, "morph"):
            dgrid = dgrid.morph(borg.modelio.ModelIOType.INPUT_ADJOINT)

        print("Calling adjoint")
        # We have to trigger the adjoint computation in any case
        fwd.adjointModel_v3(borg.modelio.GInputAdjoint(dgrid))
        analytic_gradient = borg.modelio.makeModelIODescriptor(
            box, borg.modelio.ModelIOType.OUTPUT_ADJOINT, fourier=True
        ).makeTemporaryAdjointGradient()
        analytic_gradient = fwd.getResultAdjointGradient_v3(
            borg.modelio.GOutputAdjoint(analytic_gradient)
        ).disown()
        fwd.clearAdjointGradient()
        print("--- Full backward pass took %s seconds ---" % (time.time() - start_time))
        return np.array(analytic_gradient)

    myprint("Running adjoint")

    # Prepare and compute the adjoint-gradient
    num_gradient = np.zeros((Ng, Ng, Ng // 2 + 1), dtype=np.complex128)
    s_hat_epsilon = s_hat.copy()
    analytic_gradient = dfield_ag(s_hat)

    # By how much to perturb the initial conditions
    epsilon = args.eps
    print('epsilon = ',epsilon)
    
    if fd:
        # Loop over all modes
        """
        for i, j, k in tqdm(
            itertools.product(*map(range, [Ng, Ng, Ng // 2 + 1])),
            total=Ng * Ng * (Ng // 2 + 1),
        ):
        """
        
        # Loop over one mode
        for i, j, k in tqdm(
            itertools.product(*map(range, [tP,1,1 // 2 + 1])),
            total=tP * 1 * (1 // 2 + 1),
        ):
            print('i,j,k = ',i,j,k)
            # Perturb
            s_hat_epsilon[i, j, k] = s_hat[i, j, k] + epsilon
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
    
    if args.write_to_file == None:
        write_to_file = "dump.h5"
    else:
        write_to_file = args.write_to_file+'.h5'
        
    print('Write to file = ', write_to_file)

    with h5.File(write_to_file, mode="w") as ff:
        ff["scalars/gradient_array_lh"] = analytic_gradient
        ff["scalars/gradient_array_lh_ref"] = num_gradient
        ff["scalars/gradient_array_prior"] = np.zeros_like(analytic_gradient)
        ff["scalars/gradient_array_prior_ref"] = np.zeros_like(analytic_gradient)
    