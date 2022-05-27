import numpy as np
import yt

## Define param
L = 1000 # Box length
N = 512 # Number of particles**(1/3)
date = '19052022'
snap_nr = 25
np.random.seed(42) #have to be the same as the one used for generating the ICs for P-Gadget!
outfile = 'test'

## Function

def pre_process(abs_pos,L,Ng,order='F'):
    
    # Memory layout of multi-dimensional arrays (row vs column major)
    psi = np.zeros(np.shape(abs_pos))
    x_q_v = np.zeros(np.shape(abs_pos))
    dx = L/Ng

    for i in range(0,Ng):
        for j in range(0,Ng):
            for k in range(0,Ng):
                if order=='F':
                    n = k + Ng*(j + Ng*i) #column
                elif order=='C':
                    n = i + Ng*(j + Ng*k) #row

                qx = i*dx
                qy = j*dx
                qz = k*dx

                x_q_v[n] = [qx,qy,qz]

                psi[n] = abs_pos[n] - [qx,qy,qz]
    
    return x_q_v, psi

## ----------------------------------------------------------
## ------------------------ P-Gadget ------------------------
## ----------------------------------------------------------

fname = '/cfs/home/ludo4644/ML4BORG/ICs/N128L250_11052022_May/resim/snap_borg_025.hdf5'
#fname = '/cfs/home/ludo4644/ML4BORG/ICs/N128L250_11052022_May/resim/snap_borg_000.hdf5'
#fname = '/cfs/home/ludo4644/ML4BORG/ICs/borg_resim_at_sol-login.fysik.su.se_11052022_May/resim/snap_borg_000.hdf5'
fname = '/cfs/home/ludo4644/ML4BORG/ICs/N{}L{}_'.format(N,L)+date+'/resim/snap_borg_{}.hdf5'.format(snap_nr)

unit_base = {'UnitLength_in_cm'         : 3.085678e24,
             'UnitMass_in_g'            :   1.989e+43,
             'UnitVelocity_in_cm_per_s' :      1e5}

ds = yt.load(fname,unit_base=unit_base)
ds.index
ad = ds.all_data()

# Sort particles according to id
coordinates_unsorted = ad["nbody","Coordinates"] 
arg_ind = np.argsort(np.array(ad["nbody","ParticleIDs"]))
coordinates = coordinates_unsorted[arg_ind]

# Change order from F to C
coordinates_neworder = np.zeros(np.shape(coordinates))
for i in range(0,N):
        for j in range(0,N):
            for k in range(0,N):
                n_new = k + N*(j + N*i) #column
                n_old = i + N*(j + N*k) #row
            
                coordinates_neworder[n_old] = coordinates[n_new]

# Also, swap x and z axes
abs_pos_GADGET = np.array([coordinates_neworder[:,2],coordinates_neworder[:,1],coordinates_neworder[:,0]]).T

## ----------------------------------------------------------
## -------------------------- COLA --------------------------
## ----------------------------------------------------------

import borg
import h5py as h5

global step_id
chain = 0 

# setup the box
bb = borg.forward.BoxModel()
L = L 
Nt = N

bb.L = L,L,L
bb.N = Nt,Nt,Nt

# Initialize some default cosmology
cosmo = borg.cosmo.CosmologicalParameters()

# Fiducial scale factor to express initial conditions
z_start = 69
a0 = 0.001 #change to be consistent with z_start?
nsteps = 20

chain = borg.forward.ChainForwardModel(bb)
# Add primordial fluctuations
chain.addModel(borg.forward.models.Primordial(bb, a0))
# Add E&Hu transfer function
chain.addModel(borg.forward.models.EisensteinHu(bb))
# Run an LPT model from a=0.0 to af. The ai=a0 is the scale factor at which the IC are expressed
#lpt = borg.forward.models.BorgLpt(bb, bb, ai=a0, af=1.0)
lpt = borg.forward.models.BorgPM(bb, bb, ai=a0, af=1.0, z_start=z_start, particle_factor=1, force_factor=2, supersampling=1, nsteps=nsteps, tCOLA = 1)
chain.addModel(lpt)

pos_array = np.zeros((nsteps,Nt**3,3))
step_id=0
def notif(t, Np, ids, poss, vels):
    global step_id
    print(f"Step {t} / {step_id}  (Np={Np})")
    pos_array[step_id,:,:] = poss
    step_id+=1

lpt.setStepNotifier(notif, with_particles=True)

# Set the cosmology
chain.setCosmoParams(cosmo)

# Generate white noise: it has to be scaled by 1/N**(3./2) to be one in Fourier
s_field = np.random.normal(0,1,(Nt,Nt,Nt))
ic = np.fft.rfftn(s_field/np.sqrt(Nt**3))
print('np.std(ic) = ',np.std(ic))
#ic = np.fft.rfftn(np.random.randn(Ng, Ng, Ng)/np.sqrt(Ng**3))
delta_m = np.zeros((Nt,Nt,Nt))

# RUN!
chain.forwardModel_v2(ic)
chain.getDensityFinal(delta_m)

# Get pos
abs_pos_COLA = np.zeros((lpt.getNumberOfParticles(),3))
lpt.getParticlePositions(abs_pos_COLA)

## ------------------------ SAVE ------------------------

x_q_v_gad, disp_gad = pre_process(abs_pos_GADGET,L,N,order='F')
DPF_GADGET = np.reshape(disp_gad,(N,N,N,3),order='F')

x_q_v_col, disp_col = pre_process(abs_pos_COLA,L,Nt,order='F')
DPF_COLA = np.reshape(disp_col,(Nt,Nt,Nt,3),order='F')

np.savez(outfile, 
         abs_pos_GADGET=abs_pos_GADGET,  #list sorted after id
         abs_pos_COLA=abs_pos_COLA, 
         DPF_GADGET=DPF_GADGET,
         DPF_COLA=DPF_COLA,
        )








