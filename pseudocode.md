# approxposterior

```
src/
    __init__.py
    approx.py
    gpUtils.py
    mcmcUtils.py
    utility.py
```
### Tasks:
- [ ] aprroxposterior default settings:
    - `ninit` - initial training sample size (`m0`)
    - `niter` - number of GP training iterations (`m`)
    - `mcmc_sampler` - mcmc sampling algorithm (e.g. `emcee`, `dynesty`)
- [ ] add initial training sample function - compute samples of target function, parallelized 
- [ ] add diagnostic plotting functions
    - [ ] training sample corner plot, colored by function (lnP) value
    - [ ] iteration vs. lnP 
    - [ ] density corner plot of mcmc samples
- [ ] implement other mcmc sampling options
    - [ ] dynesty

### Basic Usage
```
def fn(x):
    return 1 - x**2

bounds = [(-1,1)]
```
```
import approxposterior as approx

theta0, y0 = approx.initialSample(fn, bounds=bounds)  # opt: ninit=100

ap = approx.ApproxPosterior(theta=theta0, y=y0, fn=fn, bounds=bounds)  # opt: gp=gp
ap.train()       # opt: niter=1000

ap.run_mcmc()    # opt: mcmc_sampler='emcee'

ap.bayesOpt()    # minObjMethod='nelder-mead'
```
```
ap.plot(plots=['corner', 'training'])
```


# vplanet_inference

```
src/
    model.py
```
### Tasks:
- [ ] 

# science repo (tidalq)

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

### Tasks:
- [ ] 

### src/model.py
```
import vplanet_inference as vpi

inparams = ['primary.dMass', 'secondary.dMass', 
            'primary.dRotPeriod', 'secondary.dRotPeriod', 
            'primary.dTidalTau', 'secondary.dTidalTau', 
            'primary.dObliquity', 'secondary.dObliquity', 
            'secondary.dEcc', 
            'vpl.dStopTime',
            'secondary.dSemi']

outparams = ['final.primary.Radius', 'final.secondary.Radius',
             'final.primary.Luminosity', 'final.secondary.Luminosity',
             'final.primary.RotPer', 'final.secondary.RotPer',
             'final.secondary.OrbPeriod',
             'final.secondary.Eccentricity',
             'final.secondary.SemiMajorAxis']

infile_list = ['vpl.in', 'primary.in', 'secondary.in']

inpath  = '../infile_stellar_eqtide/'
outpath = 'output/'

def tau_conversion(tau):
    return (10 ** tau) / YEARSEC

factor = np.array([1, 1, -1, -1, 1, 1, -1, -1, 1, 1e9, -1])
conversions = {4:tau_conversion, 5:tau_conversion}

# input parameters
theta_true = [1.0, 0.9, 5, 5, -1, -1, 10, 10, .2, 2.5, .09]

vpm = vpi.VplanetModel(inparams, inpath=inpath, infile_list=infile_list, factor=factor, conversions=conversions)
```

### src/run.py
```
import approxposterior as approx
import vplanet_inference as vpi

import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('files', metavar='N', type=str, nargs='+', help='')

model, config  <-  parser

if init_training_samples == True:
    theta0, y0 = approx.initialSample(fn, bounds=config.bounds, method=config.sample_method)
else:
    theta0, y0 = np.load()

ap = approx.ApproxPosterior(theta=theta0,
                            y=y0,
                            gp=gp,
                            fn=config.fn,
                            bounds=config.bounds,
                            algorithm=config.algorithm)

if train_gp == True:
    ap.train(niter=config.m)

if run_mcmc == True:
    ap.run_mcmc(mcmc_sampler=config.mcmc_sampler)
```

### scripts/ctl/test0/config.py

```
__all__ = ['bounds', 'data', 
           'sample_method', 
           'mcmc_sampler'
           'algorithm']

gp = gpUtils.defaultGP(theta, y, white_noise=-12)
```


### run command

```
>>> python run.py <model> <config> --init_training_samples
```
```
>>> python run.py <model> <config> --train_gp --load_cache
```
```
>>> python run.py <model> <config> --run_mcmc --sampler=emcee
```