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
(X) = completed
```
```
__init__.py             [C] (O) remove deprecated imports 

approx.py
    ApproxPosterior     
        __init__        [C] (O) option to load cached sims
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

gmmUtils.py             [R]     used for Wang & Li convergence method (deprecated)
    fitGMM              [R]

gpUtils.py
    defaultHyperPrior   [K] 
    defaultGP           [K]
    optimizeGP          [C] (O) parallelize
    convergenceCheck    [A] (O) implement convergence criteria (tbd)

likelihood.py           [M] (O) rename to benchmarks.py; add more benchmark functions
    rosenbrockLnlike
    rosenbrockLnprior
    rosenbrockSample
    rosenbrockLnprob
    ... 

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

visualization.py        [A] (O) import into approx.py
    cornerLnP           [A] (O)
    cornerDensity       [A] (O)
    iterationLnP        [A] (O)
```


# alabi

<!-- ### Directory structure:
![](img/alabi.svg) -->

