import numpy as np
import ttvfast
import rebound as rb

def convert_results_to_times_and_rvs(results,Npl):
    """
    Convenience function to convert TTVFast 'results' object into
    planet transit times and radial velocity measurements.
    """
    planet_mask = np.array(results['positions'][0])
    times = np.array(results['positions'][2])
    planet_times =[times[np.logical_and(times>0,planet_mask==N)] for N in range(Npl)]
    rvs = results['rv']
    return planet_times,rvs


def ParamsToTransitTimes(Nplanets,params,TMax,epoch=0,Mstar=1):
    """
    A wrapper for ttvfast. 
    Convert parameters to times and radial velocities.
    
    Arguments
    ---------
    params : (Nplanets,7)
        An array containing the parameters of each planet in the TTV model.
        The parameters of the ith planet are given as:
            params[i] = (mp_i,period_i,e_i,inc_i,Omega_i,omega_i,mean_anom_i)
    Tmax : float
        Integrate from epoch to Tmax
    epoch : float
        Reference time at which orbital elements are defined.
    Mstar : float, optional
        Star mass in solar masses. Default is 1.
        
    Returns
    -------
    list of ndarrays : 
        List of each planet's transit times.
    """
    assert params.shape == (Nplanets,7), "Improper shape for params array!"
    min_Tperi = np.infty
    planets = []
    for i in range(Nplanets):
        pars = params[i]
        planet_i = ttvfast.models.Planet(*pars)
        period = pars[1]
        e = pars[2]
        min_Tperi = np.min((min_Tperi ,period * (1-e)**(1.5) / np.sqrt(1+e)))
        planets.append(planet_i)
    # Set times-step to 1/30th of shortest peri timescale
    dt = min_Tperi / 30
    
    result = ttvfast.ttvfast(planets,Mstar,epoch,dt,TMax,input_flag=1) # input_flag=1 --> astrocentric coordinates
    planet_times,rv_values = convert_results_to_times_and_rvs(result,Nplanets)
    
    return planet_times

def hdf5row_to_ttvfast_params_and_epoch(row):
    """
    Convert a row of the HDF5 file containing the posteriors of a TTV system
    into TTVFast parameters.

    Parameters
    ----------
    row : ndarray
        A numpy array of size N*5 where N is the number of planets. Entries are
        m_1,P_1,ecos_1,esin_1,Tc_1,...,m_N,P_N,ecos_N,esin_N,Tc_N

    Returns
    -------
    params : ndarray
        An (N,7) array of input parameters for the TTVFast code.
    epoch : float
        The time at which the system configuration is specified. In the original
        posterior data, the epoch is set to min(Tc_i) - 0.1 * min(P_i) where
        min(Tc_i) and min(P_i) are the minimum transit time and appearing in the
        input parmeters.
    """
    Nplanets = len(row)//5
    params = np.zeros((Nplanets,7))
    RAD2DEG = 180 / np.pi
    row_reshaped = row.reshape((Nplanets,5))
    Pmin = np.min(row_reshaped[:,1])
    TCmin = np.min(row_reshaped[:,4])
    epoch = TCmin - 0.1 * Pmin
    for i,data in enumerate(row.reshape((Nplanets,5))):
        m,per,ecos,esin,Tc = data
        e = np.sqrt(ecos*ecos + esin*esin)
        l0 = np.mod(2 * np.pi * (epoch - Tc) / per,2*np.pi) + 0.5 * np.pi
        pmg = np.arctan2(esin,ecos)
        params[i] = np.array([m,per,e,90.,0.,np.mod(RAD2DEG * pmg,360.),np.mod(RAD2DEG * (l0-pmg),360.)])

    return params,epoch

def ttv_fast_params_to_rebound_simulation(params, epoch, Mstar = 1.):
    """
    Convert a set of TTVFast parameters to a rebound simulation.

    Parameters
    ----------
    params : ndarray
        ((N,7)) set of masses and orbital elements for each planet.
    epoch : float, optional
        Time of initial conditions, by default 0
    Mstar : float, optional
        Mass of central star in solar masses, by default 1

    Returns
    -------
    rebound.simulation.Simulation
        Simulation of the planetary system with initial conditions specified by
        'params'.
    """
    sim = rb.Simulation()
    sim.units = ('days','AU','Msun')
    sim.add(m=Mstar)
    star = sim.particles[0]
    deg2rad = lambda x: np.mod(x*np.pi/180,2*np.pi)
    for row in params:
        mu,per,e,inc,Omega,omega,M = row
        sim.add(
            m=mu*Mstar,P=per,e=e,
            inc=deg2rad(inc),Omega=deg2rad(Omega),omega=deg2rad(omega),
            M = deg2rad(M),
            primary=star,
            jacobi_masses=False
        )
    sim.t = epoch
    sim.move_to_com()
    return sim