import numpy as np
from ttv2fast2furious.kepler import KOISystemObservations
from sys import stdout
import h5py
h5data = h5py.File("/Users/shadden/Projects/00_Codes_And_Data/HL17_Posteriors/NBody_MCMC_Posteriors.hdf5")
import sys
sys.path.append("../01_code/")
from HL17_posterior_to_rb_sim import *
KOI_num,kepler_name = 157, "Kepler-11"
observations = KOISystemObservations(KOI_num)
mcmc_data = h5data[kepler_name]['HighMassPriors']['PosteriorSample']


# periods of planets in observed data
obs_periods = [o.linear_best_fit()[1] for o in observations.values()]

# Periods from MCMC data
data_periods = mcmc_data[0,1::5]

# Number of planets in ttv fit
Nplanets = len(data_periods)

# put the observational data in to correct order
observations_list = [list(observations.values())[np.argmin(np.abs(obs_periods - data_p))] for data_p in data_periods]
# add un-modeled last planet to list
obs_g = observations['KOI-157.05']
observations_list+= [obs_g]

# minimum and maximum observed transit times
tmin,tmax = np.min([np.min(obs.times) for obs in observations_list]),np.max([np.max(obs.times) for obs in observations_list])

# get random posterior sample 
mcmc_sample = mcmc_data[np.random.randint(0,len(mcmc_data))]

# get transit time residuals for planet g
def planet_g_resids(per,Tc,m=0,ecos=0,esin=0):
    mcmc_pars = np.append(mcmc_sample,[m,per,ecos,esin,Tc])
    ttvfast_pars,epoch = hdf5row_to_ttvfast_params_and_epoch(mcmc_pars)
    transit_times = ParamsToTransitTimes(6,ttvfast_pars,tmax,epoch=epoch,Mstar=1)
    times_g = transit_times[-1]
    return (times_g[obs_g.transit_numbers] - obs_g.times) / obs_g.uncertainties

# minimize planet g residuals for given value of mass and ecc
from scipy.optimize import least_squares

# set mass and eccentricity of planet g
m,ecos,esin = 10*3e-6,np.random.normal(scale=0.03),np.random.normal(scale=0.03)

# get best-fit period and time of conjunction
lsq_result = least_squares(lambda x: planet_g_resids(*x,m,ecos,esin),obs_g.linear_best_fit())

# generate transit times
per,Tc = lsq_result.x
bestfit_mcmc_pars = np.append(mcmc_sample,[m,per,ecos,esin,Tc])

params_full,epoch = hdf5row_to_ttvfast_params_and_epoch(bestfit_mcmc_pars)

sim = ttv_fast_params_to_rebound_simulation(params_full,epoch)
print("Initial simulation generated...")
stdout.flush()
# apply transformation from osculating to mean elements
from celmech import Poincare
from celmech.lie_transformations import FirstOrderGeneratingFunction
rd1,rd2 = dict(),dict()
pvars = Poincare.from_Simulation(sim)
chi = FirstOrderGeneratingFunction(pvars)
for i in range(1,pvars.N):
    pi = pvars.particles[i]
    for j in range(i+1,pvars.N):
        chi.add_zeroth_order_term(indexIn=i,indexOut=j)
        pj = pvars.particles[j]
        Pratio = pj.P/pi.P
        J = int(np.floor(1 + 1/(Pratio - 1)))
        rd1[(i,j)] = J
        rd2[(i,j)] = J + 1
        chi.add_MMR_terms(J,1,indexIn=i,indexOut=j)
        chi.add_MMR_terms(J+1,1,indexIn=i,indexOut=j)
print("Genearting function created...")
stdout.flush()

chi.osculating_to_mean()
print("Genearting function correction applied...")
stdout.flush()

# Get Laplace-Lagrange secular matrices
from celmech.secular import LaplaceLagrangeSystem
llsys = LaplaceLagrangeSystem.from_Poincare(pvars)
llsys.add_first_order_resonance_terms(rd1)
llsys.add_first_order_resonance_terms(rd2)
Te,De = llsys.diagonalize_eccentricity()
Ti,Di = llsys.diagonalize_inclination()

print("Secular system solved...")
stdout.flush()

# Spread AMD uniformly among ecc/inc modes
from SampleAMD import get_critical_AMD, get_samples
AMDcrit = get_critical_AMD(pvars)
f = 1
mode_amps = f * AMDcrit * get_samples(2*(pvars.N-1) - 1 )
uvdata = np.sqrt(mode_amps) * np.exp(1j * np.random.uniform(-np.pi,np.pi,2*(pvars.N-1) - 1))
u = uvdata[:(pvars.N-1)]
v = np.concatenate(([0],uvdata[(pvars.N-1):])) # set 0-freq mode amp to 0
x = Te.T @ u
y = Ti.T @ v
kappa = np.real(x)
eta = np.imag(x)
sigma = np.real(y)
rho = np.imag(y)
for i in range(1,pvars.N):
    pvars.qp[pvars.qp_vars[3*(i-1)+1]] = eta[i-1]
    pvars.qp[pvars.qp_vars[3*(i-1)+2]] = rho[i-1]
    pvars.qp[pvars.qp_vars[3*(i-1)+1 + 3*(pvars.N-1)]] = kappa[i-1]   
    pvars.qp[pvars.qp_vars[3*(i-1)+2 + 3*(pvars.N-1)]] = sigma[i-1]

print("System set up...")
stdout.flush()

# transform back to osculating variables
chi.mean_to_osculating()
print("Mean to osculating correction applied...")
stdout.flush()

sim_to_save = pvars.to_Simulation()
sim_to_save.save_to_file("kepler_11_sim.sa")

