# approxposterior

### Scope
- generalized to any function optimization or mcmc problem
    - 'Basic usage' template with default settings 
        - as simple to use as `scipy.optimize` or `emcee`
        - helper functions for constructing common priors and likelihoods
    - 'Advanced usage' tutorials 
        - users can experiment with different GP and MCMC settings
        - flexible for user defined functions, priors, likelihoods 
- should NOT contain any functionality specific to vplanet (this will go in the `vplanet_inference` package)

### Current workflow:
```python
from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np

# Define algorithm parameters
m0 = 50                           # Initial size of training set
m = 20                            # Number of new points to find each iteration
nmax = 2                          # Maximum number of iterations
bounds = [(-5,5), (-5,5)]         # Prior bounds
algorithm = "bape"                # Use the Kandasamy et al. (2017) formalism
seed = 57                         # RNG seed
np.random.seed(seed)

# emcee MCMC parameters
samplerKwargs = {"nwalkers" : 20}        # emcee.EnsembleSampler parameters
mcmcKwargs = {"iterations" : int(2.0e4)} # emcee.EnsembleSampler.run_mcmc parameters

# Sample design points from prior
theta = lh.rosenbrockSample(m0)

# Evaluate forward model log likelihood + lnprior for each theta
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.rosenbrockLnlike(theta[ii]) + lh.rosenbrockLnprior(theta[ii])

# Default GP with an ExpSquaredKernel
gp = gpUtils.defaultGP(theta, y, white_noise=-12)

# Initialize object using the Wang & Li (2018) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=lh.rosenbrockLnprior,
                            lnlike=lh.rosenbrockLnlike,
                            priorSample=lh.rosenbrockSample,
                            bounds=bounds,
                            algorithm=algorithm)

# Run!
ap.run(m=m, nmax=nmax, estBurnin=True, nGPRestarts=3, mcmcKwargs=mcmcKwargs,
    cache=False, samplerKwargs=samplerKwargs, verbose=True, thinChains=False,
    onlyLastMCMC=True)

# Check out the final posterior distribution!
import corner

# Load in chain from last iteration
samples = ap.sampler.get_chain(discard=ap.iburns[-1], flat=True, thin=ap.ithins[-1])

# Corner plot!
fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], show_titles=True,
                    scale_hist=True, plot_contours=True)

# Plot where forward model was evaluated - uncomment to plot!
fig.axes[2].scatter(ap.theta[m0:,0], ap.theta[m0:,1], s=10, color="red", zorder=20)

# Save figure
fig.savefig("finalPosterior.png", bbox_inches="tight")
```

### Proposed workflow:
```python
def fn(x):
    return 1 - x**2

bounds = [(-1,1)]
```
```python
import alabi

ap = alabi.SurrogateModel(fn=fn, bounds=bounds)  # opt: gp=gp

ap.initial_train()  # opt: ninit=100, sampling='uniform'

ap.active_train()   # opt: niter=1000

ap.run_mcmc()       # opt: mcmc_sampler='emcee', mcmc_kwargs={}

ap.bayes_opt()      # minObjMethod='nelder-mead'
```
```
ap.plot(plots=['corner', 'training'])
```

### Tasks:
- [ ] which parts of approxposterior do we keep? which are depricated/should be cut?
- [ ] rename package
    - gaussian process accelerated bayesian inference (GABI)
    - kernel regression accelerated bayesian inference (KRABI)
    - active learning kernel regression accelerated bayesian inference (ALKABI)
    - active learning accelerated bayesian inference (ALABI)
- [ ] test and implement new convergence criteria
    - convergence of GP hyperparameters?
    - convergence of GP mean and covariance? 
- [ ] create default aprroxposterior settings for:
    - `ninit` - initial training sample size (`m0`)
    - `niter` - number of GP training iterations (`m`)
    - `mcmc_sampler` - mcmc sampling algorithm (e.g. `emcee`, `dynesty`)
- [ ] add initial training sample function 
    - compute samples of target function, parallelized 
    - implement different hypercube sampling options (e.g. grid, uniform, sobol)
- [ ] add prior and likelihood helper functions
    - add ability to use a prior from a previous run
- [ ] parallelize GP optimization (install python 3.9)
- [ ] add diagnostic plotting functions
    - [ ] training sample corner plot, colored by function (lnP) value
    - [ ] iteration vs. lnP 
    - [ ] density corner plot of mcmc samples
- [ ] implement other mcmc sampling options
    - [ ] dynesty (ask Jake for prior transform functions)
- [ ] good 'benchmark' distributions? 
- [ ] plan for user plugins for specific packages like vplanet


# vplanet_inference

### Directory structure:
```
src/
    model.py
```
### Tasks:
- [ ] user specified units? 
- [ ] automatically handle vplanet minus signs in inputs?
    - look in v2.0 bigplanet?
- [ ] integrate with bigplanet hdf5 compression

# science repo (tidalq)

### Directory structure:
```
infile/
    ctl/
        primary.in
        secondary.in
        vpl.in
    cpl/
        primary.in
        secondary.in
        vpl.in
src/
    __init__.py
    model_ctl.py
    model_cpl.py
    run.py
scripts/
    rup147/
        ctl/
            test0/
                config.py
                output/
                    ...
                plots/
                    ...
                results/
                    ...
```

### src/model_ctl.py
```python
import vplanet_inference as vpi

__all__ = ['vpm']


inparams = ['primary.dMass', 
            'secondary.dMass', 
            'primary.dRotPeriod', 
            'secondary.dRotPeriod', 
            'primary.dTidalTau', 
            'secondary.dTidalTau', 
            'primary.dObliquity', 
            'secondary.dObliquity', 
            'secondary.dEcc', 
            'vpl.dStopTime',
            'secondary.dSemi']

outparams = ['final.primary.Radius', 
             'final.secondary.Radius',
             'final.primary.Luminosity', 
             'final.secondary.Luminosity',
             'final.primary.RotPer', 
             'final.secondary.RotPer',
             'final.secondary.OrbPeriod',
             'final.secondary.Eccentricity',
             'final.secondary.SemiMajorAxis']

infile_list = ['vpl.in', 'primary.in', 'secondary.in']

inpath  = '../infile/ctl/'

def tau_conversion(tau):
    return (10 ** tau) / YEARSEC

factor = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1e9, -1])
conversions = {4:tau_conversion, 5:tau_conversion}

vpm = vpi.VplanetModel(inparams, inpath=inpath, infile_list=infile_list, factor=factor, conversions=conversions)
```

### src/run.py
```python
import approxposterior as approx
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--foo', help='foo help')
args = parser.parse_args()

model, config  <-  parser

vpm = model.vpm
vpm.InitializeBayes(data=config.data, bounds=config.bounds)
fn = vpm.LnPosterior()

if init_training_samples == True:
    theta0, y0 = approx.initialSample(fn, bounds=config.bounds, method=config.sample_method)
else:
    theta0, y0 = np.load()

ap = approx.ApproxPosterior(theta=theta0, y=y0, fn=fn, bounds=config.bounds)

if train_gp == True:
    ap.train(niter=config.niter)

if run_mcmc == True:
    ap.run_mcmc(mcmc_sampler=config.mcmc_sampler)
```

### scripts/ctl/test0/config.py

```python
__all__ = ['bounds', 'data', 
           'sample_method', 
           'mcmc_sampler'
           'algorithm']

gp = gpUtils.defaultGP(theta, y, white_noise=-12)
```


### run commands:

```
>>> python src/run.py <model file> <config file> --init_training_samples
```
```
>>> python src/run.py <model file> <config file> --train_gp --load_cache
```
```
>>> python src/run.py <model file> <config file> --run_mcmc --sampler=emcee
```

### Tasks:
- [ ] define configuration script setup (`config.py`) 