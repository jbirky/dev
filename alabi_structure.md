# approxposterior

### Directory structure:
![](img/approxposterior.svg)


```
[K] = keep module
[C] = change module
[M] = move functionality to different module or rename
[R] = remove module
[A] = add new module

(O) = to be completed
(S) = started
(X) = completed
```
```
__init__.py             [C] (O) remove deprecated imports 

approx.py
    ApproxPosterior     
        __init__        [C] (S) default options for: priorSample
                            (S) get rid of inputs: lnlike, lnprior 
                                add input: fn (replaces lnlike)
                            (S) option to load theta0, y0
        initialSample   [A] (O) compute initial training samples (theta0, y0)
        _gpll           [K]     
        optGP           [K]     calls gpUtils.optimizeGP; might need to tweak inputs
        run             [M] (O) rename -> train
                        [C] (O) remove MCMC, only call findNextPoint
                            (O) remove old convergence criteria; add new cc from gpUtils
                            (O) option to restart training from where you left off
        findNextPoint   [C] (O) think this is mostly what I want; change some variable names
        runMCMC         [C] (O) add option for choosing MCMC package
        findMAP         [K]
        bayesOpt        [C] (O) remove old convergence criteria; add new cc from gpUtils
        plot            [A] (O) call functions from visualization.py

gpUtils.py
    defaultHyperPrior   [K] 
    defaultGP           [K]
    optimizeGP          [C] (O) parallelize
    convergenceCheck    [A] (O) implement convergence criteria (tbd)
    hyperCubeSample     [A] (O) sampling methods (uniform, grid, sobol)

mcmcUtils.py            [M] (O) these functions are specific to emcee -> emceeUtils?
                                maybe want generalized mcmcUtils wrapper?
    validateMCMCKwargs  [K]
    batchMeansMCSE      [K]
    estimateBurnin      [K]

utility.py  
    logsubexp           [K]
    AGPUtility          [K]
    BAPEUtility         [K]
    JonesUtility        [K]
    minimizeObjective   [K]
    klNumerical         [R]

likelihood.py           [M] (O) rename to benchmarks.py; add more benchmark functions
    rosenbrockLnlike
    rosenbrockLnprior
    rosenbrockSample
    rosenbrockLnprob
    ... 

```
```
bayes.py                [A] (O) utilities for constructing common likelihood/prior fns
                                include prior transform function

defaults.py             [A] (O) define default settings and import to other files?

visualization.py        [A] (O) import into approx.py
    cornerLnP           [A] (O)
    cornerDensity       [A] (O)
    iterationLnP        [A] (O)
```
```
priors.py               [R]     not implemented functions

gmmUtils.py             [R]     used for Wang & Li convergence method (deprecated)
    fitGMM              [R]
```


# alabi

<!-- ### Directory structure:
![](img/alabi.svg) -->

