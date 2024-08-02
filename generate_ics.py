import numpy as np
import rebound as rb
import celmech as cm
from sys import stdout

# generate initial Poincare object
sim0 = rb.Simulation('median_kepler-11-config.bin')
pvars = cm.Poincare.from_Simulation(sim0)

# get generating function
from celmech.lie_transformations import FirstOrderGeneratingFunction
chi = FirstOrderGeneratingFunction(pvars)

# store nearest resonant interactions between planets
rd1,rd2 = dict(),dict()
for i in range(1,pvars.N):
    pi = pvars.particles[i]
    for j in range(i+1,pvars.N):
        if j-i < 3:
            chi.add_zeroth_order_term(indexIn=i,indexOut=j)
        pj = pvars.particles[j]
        Pratio = pj.P/pi.P
        J = int(np.floor(1 + 1/(Pratio - 1)))
        rd1[(i,j)] = J
        rd2[(i,j)] = J + 1
        chi.add_MMR_terms(J,1,indexIn=i,indexOut=j)
        chi.add_MMR_terms(J+1,1,indexIn=i,indexOut=j)

# second order MMRs for c/d and d/e
chi.add_MMR_terms(5,2,indexIn=2,indexOut=3)
chi.add_MMR_terms(7,2,indexIn=3,indexOut=4)

print("Genearting function created...")
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

print("Updating generating function...")
stdout.flush()
chi._update()

print("Creating simulations...")
stdout.flush()

# Spread AMD uniformly among ecc/inc modes
from SampleAMD import get_critical_AMD, get_samples
Nic = 100
f_AMDs = np.geomspace(1/3,10,10)
AMDcrit = get_critical_AMD(cm.Poincare.from_Simulation(sim0))


lmbda_indx = [chi.qp_vars.index(s) for s in sp.symbols("lambda(1:7)")]
eta_indx = [chi.qp_vars.index(s) for s in sp.symbols("eta(1:7)")]
kappa_indx = [chi.qp_vars.index(s) for s in sp.symbols("kappa(1:7)")]
rho_indx = [chi.qp_vars.index(s) for s in sp.symbols("rho(1:7)")]
sigma_indx = [chi.qp_vars.index(s) for s in sp.symbols("sigma(1:7)")]

for f_AMD in f_AMDs:
    
    # take intial masses and periods from sim0
    y_mean = np.array([cm.Poincare.from_Simulation(sim0).values for _ in range(Nic)])
    
    # randomize lambda values
    y_mean[:,lmbda_indx] = np.random.uniform(-np.pi,np.pi,(Nic,6))

    # distribute AMD uniformly among modes
    for i in range(Nic):
        # random modes
        mode_amps = f_AMD * AMDcrit * get_samples(2*(pvars.N-1) - 1 )
        uvdata = np.sqrt(mode_amps) * np.exp(1j * np.random.uniform(-np.pi,np.pi,2*(pvars.N-1) - 1))
        u = uvdata[:(pvars.N-1)]
        v = np.concatenate(([0],uvdata[(pvars.N-1):])) # set 0-freq mode amp to 0

        # to x/y
        x = Te.T @ u
        y = Ti.T @ v

        # to real variables 
        kappa = np.real(x)
        eta = np.imag(x)
        sigma = np.real(y)
        rho = np.imag(y)

        # save
        y_mean[i,eta_indx] = eta
        y_mean[i,kappa_indx] = kappa
        y_mean[i,rho_indx] = rho
        y_mean[i,sigma_indx] = sigma
    # convert to osculating variables
    y_osc = np.array([chi.mean_to_osculating_state_vector(v) for v in y_mean])
    for i,ic in enumerate(y_osc):
        pvars.values = ic
        sim = pvars.to_Simulation()
        sim.move_to_com()
        cm.nbody_simulation_utilities.align_simulation(sim)
        sim.save_to_file(f"kepler-11-ics/kep-11_fAMD_{f_AMD:.2f}_id{i:03d}.bin")