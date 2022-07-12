#+
#   ARES/HADES/BORG Package -- ./scripts/check_gradients.py
#   Copyright (C) 2014-2022 Guilhem Lavaux <guilhem.lavaux@iap.fr>
#   Copyright (C) 2009-2022 Jens Jasche <jens.jasche@fysik.su.se>
#
#   Additional contributions from:
#      Fabian Schmidt <fabians@mpa-garching.mpg.de> (2021)
#      Guilhem Lavaux <guilhem.lavaux@iap.fr> (2016-2021)
#      Jens Jasche <j.jasche@tum.de> (2016-2017)
#      elsner <f.elsner@mpa-garching.mpg.de> (2017)
#
#+
import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append('/cfs/home/ludo4644/WienerFilter/Aquila/ares/scripts/')
from ares_tools import read_all_h5
import pylab as plt
from sys import argv
import argparse

file='dump.h5'

parser = argparse.ArgumentParser()
parser.add_argument("--filename", default="dump.h5", type=str)
parser.add_argument("--only-numerical", action="store_true")
parser.add_argument("--only-analytical", action="store_true")
parser.add_argument("--test_points", type=int, default=4)

args = parser.parse_args()
file = args.filename
only_analytical=True
only_numerical=True
if args.only_numerical:
    only_analytical=False
if args.only_analytical:
    only_numerical=False

print('Reading from %s' % file)
g=read_all_h5(file)

#ss=16
#step=10

ss=1#8*8
step=1
maxs = args.test_points

#'F' order so can compare using itertools.product(*map(range, [4,1,1 // 2 + 1])) in run_gradient_test.py
prior = g.scalars.gradient_array_prior[::ss,:,:].flatten('F') 
prior_ref = g.scalars.gradient_array_prior[::ss,:,:].flatten('F')

dpr_adj_re= prior.real
dpr_ref_re= prior_ref.real
dpr_adj_im= prior.imag
dpr_ref_im= prior_ref.imag

lh = g.scalars.gradient_array_lh[::ss,:,:].flatten('F')
lh_ref = g.scalars.gradient_array_lh_ref[::ss,:,:].flatten('F')

dlh_adj_re= lh.real
dlh_ref_re= 1*lh_ref.real
dlh_adj_im= lh.imag
dlh_ref_im= 1*lh_ref.imag

print('dlh_adj_re = ',dlh_adj_re)
print('dlh_adj_re[0] = ',dlh_adj_re[0])
print('dlh_ref_re = ',dlh_ref_re)
print('dlh_adj_im = ',dlh_adj_im)
print('dlh_ref_im = ',dlh_ref_im)

fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.95, wspace=0.25, hspace=0.16)

ax1=plt.subplot(2,2,1)             # left subplot in top row
plt.axhline(0.0, color='black', linestyle=':')
if only_analytical:
    plt.plot(dpr_adj_re[:maxs][::step],'ro',markersize=5.)
if only_numerical:
    plt.plot(dpr_ref_re[:maxs][::step],color='blue')
ax1.yaxis.get_major_formatter().set_powerlimits((-2, 2))
ax1.xaxis.set_ticklabels('')
plt.ylabel('dPSI_prior_real')

labels=[]
if only_analytical:
    labels.append('gradient')
if only_numerical:
    labels.append('finite diff')

ax2=plt.subplot(2,2,2)             # right subplot in top row
plt.axhline(0.0, color='black', linestyle=':')
arts=[]
if only_analytical:
    arts.append(plt.plot(dpr_adj_im[:maxs][::step],'ro',markersize=5.)[0])

if only_numerical:
    arts.append(plt.plot(dpr_ref_im[:maxs][::step],color='blue')[0])
ax2.legend(arts,labels)
ax2.yaxis.get_major_formatter().set_powerlimits((-2, 2))
ax2.xaxis.set_ticklabels('')
plt.ylabel('dPSI_prior_imag')

ax3=plt.subplot(2,2,3)             # left subplot in bottom row
plt.axhline(0.0, color='black', linestyle=':')
if only_analytical:
    plt.plot(dlh_adj_re[:maxs][::step],'ro',markersize=5.)
if only_numerical:
    plt.plot(dlh_ref_re[:maxs][::step],color='blue')
ax3.yaxis.get_major_formatter().set_powerlimits((-2, 2))
plt.xlabel('voxel ID')
plt.ylabel('dPSI_likelihood_real')

ax4=plt.subplot(2,2,4)             # right subplot in bottom row
plt.axhline(0.0, color='black', linestyle=':')
if only_analytical:
    plt.plot(dlh_adj_im[:maxs][::step],'ro',markersize=5.)
if only_numerical:
    plt.plot(dlh_ref_im[:maxs][::step],color='blue')
ax4.yaxis.get_major_formatter().set_powerlimits((-2, 2))
plt.xlabel('voxel ID')
plt.ylabel('dPSI_likelihood_imag')

plt.savefig('check_gradient.png')

#plt.scatter(dpr_adj_re,dpr_ref_re)
#plt.show()
