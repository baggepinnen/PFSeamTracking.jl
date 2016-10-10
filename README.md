# PFSeamTracking

[![Build Status](https://travis-ci.org/baggepinnen/PFSeamTracking.jl.svg?branch=master)](https://travis-ci.org/baggepinnen/PFSeamTracking.jl)

Repository implementing the framework detailed in  
  **Particle Filter Framework for 6D Seam Tracking Under Large External Forces Using 2D Laser Sensors**  
  F Bagge Carlson, M Karlsson, A Robertsson, R Johansson  
  *2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*

Available online: http://portal.research.lu.se/portal/files/12204979/seamTrackingPaper.pdf

```bibtex
@inproceedings{baggecarlson2016particle,
  title={Particle Filter Framework for 6D Seam Tracking Under Large External Forces Using 2D Laser Sensors},
  author={Bagge Carlson, Fredrik and Karlsson, Martin and Robertsson, Anders and Johansson, Rolf},
  booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2016},
  organization={IEEE--Institute of Electrical and Electronics Engineers Inc.}
}
```

# Installation
This package is not registered in `METADATA` and must thus be installed with the command  
`Pkg.clone("git@github.com:baggepinnen/PFSeamTracking.jl.git")`  
To test the functionality of the package, execute  
`Pkg.test("PFSeamTracking")`  
To plot results etc., install the package `Plots.jl` and a compatible backend. To perform the statistical analysis, install `ExperimentalAnalysis.jl`

# Usage
The file `simulate_tracking.jl` contains an example that executes a numer of simulations in parallel (start julia with `julia -p x` where `x` is your desired number of workers). The script then performs statistical analysis of the results using linear modeling with parameters as factors, as described in the paper.
