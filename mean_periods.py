import numpy as np
import rebound as rb
import h5py
h5data = h5py.File("/Users/shadden/Projects/00_Codes_And_Data/HL17_Posteriors/NBody_MCMC_Posteriors.hdf5")
from HL17_posterior_to_rb_sim import *
KOI_num,kepler_name = 157, "Kepler-11"
mcmc_data = h5data[kepler_name]['HighMassPriors']['PosteriorSample']
masses_and_mean_periods = np.zeros((100,10))
for indx in range(100):
    print(indx)
    # get random posterior sample 
    mcmc_sample = mcmc_data[np.random.randint(0,len(mcmc_data))]

    # generate TTVFast input parameters
    params,epoch = hdf5row_to_ttvfast_params_and_epoch(mcmc_sample)

    # get rebound simulation
    sim = ttv_fast_params_to_rebound_simulation(params,epoch)

    # set integration times
    tmax = sim.particles[1].P * 500
    times = np.linspace(0,tmax,2048)
    exp_iM = np.zeros((5,len(times)),dtype = np.complex128) # store e^{iM} for each planet

    for i,t in enumerate(times):
        sim.integrate(t,exact_finish_time=1)
        orbits = sim.orbits(primary=sim.particles[0])
        for j,o in enumerate(orbits):
            exp_iM[j,i] = np.exp(1j * o.M)

    # get orbital periods
    from celmech.miscellaneous import frequency_modified_fourier_transform as fmft
    periods = np.zeros(5)
    for j in range(5):
        fmft_result = fmft(times,exp_iM[j],4)
        periods[j] = 2*np.pi / list(fmft_result.keys())[0]
    masses = np.array([p.m for p in sim.particles[1:]])
    masses_and_mean_periods[indx] = np.concatenate((masses,periods))

# construct median simulation
median_masses = np.median(masses_and_mean_periods[:,:5],axis=0)
median_period_ratios = np.median(np.array([x/x[0] for x in masses_and_mean_periods[:,5:]]),axis=0)
sim = rb.Simulation()
sim.add(m=1)
star= sim.particles[0]
for j in range(5):
    sim.add(m=median_masses[j],P = median_period_ratios[j], primary=star)

# add planet g
Pg = 118.378 / 10.303
mg = 0.5 * (median_masses[3] + median_masses[4])
sim.add(m=mg,P = Pg, primary=star)

sim.save_to_file("median_kepler-11-config.bin")
