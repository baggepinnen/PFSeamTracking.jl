


using Robotlib
using Plots
include("../src/pf_state_estimator.jl")
include("../src/simfuns.jl")
import PFstateEstimator
import PFsimulator
pyplot()
default(size=(1000,700))
import DSP

srand(1)

# Particle filter parameters
h           = 0.01
sigmaW      = 2*[1.5,  0.02*pi/180, 1e-6] # t, R, f
sigmaV      = [0.1, 0.1, 1*pi/180] # x, y, theta
dFK         = 1. # plateau width
kd          = 0*1e-6 # extension to plateau width due to force (essentialy a super simple compliance model)
sigmaVFK    = [0.1, 0.05*pi/180, dFK, kd] # xyz, theta, plateau width, norm(f)
Npart       = 200
force_filter= 0.1 # in [0,1], lower value = more filtering

# Simulation parameters
T           = 80
σt          = 0
σR          = 0
σq          = 0
σm          = 0
σf          = 0
force_ref   = zeros(6) # Jag satte ner kraften för att debugga
Kreact      = zeros(6,6)
Kcomp       = 0*eye(6) # Compliance matrix
R_TF_M      = [
-1 0 0;
0  0 1;
0  1 0]
T_TF_M      = Rt2T(R_TF_M, [0,0.,0]) # Tool flange to meta


# Construct nominal trajectories
triwave     = cumsum(sign(sin(0.05*(1:T))))
triwave   ./= maximum(triwave)
triwave     = DSP.filtfilt(ones(15),[15.], triwave)
traj_nom    = PFsimulator.traj2frames3D([20*(triwave) linspace(0,200,T) zeros(T)]')

# Setup parameter objects
simparams                   = PFsimulator.SimParams(h, T, σt, σR, σq, σm, σf, force_ref, Kreact, Kcomp, T_TF_M)

pfparams                    = PFstateEstimator.PFparams(sigmaW, sigmaV, sigmaVFK, Npart, h, force_filter)

simdata    = PFsimulator.generate_simdata(traj_nom, simparams, optimize = false)
simresult  = PFsimulator.run_tracking("Single sensor", simdata, simparams, pfparams, 1, savetrace = true)

traj_diff = simresult.simdata.traj_nom-simresult.simdata.traj_act

# PFsimulator.plot(simresult, :tracking, :resample, :meas)
# plot_traj_sub(traj_diff, title="Noise in traj_act")

# Tests
@assert var(simresult.simdata.meas) < 1e-10 "Laser Measurement incorrect"
@assert var(traj_diff[1:3,1:4,:]) < 1e-10 "traj_act is incorrect"
@assert sum([simresult.trace[i].resample for i = 1:T]) < T/2 "Lots of resampling required!"
# @assert norm(simresult.et) < 2 "Tracking error is very high"


# Modify traj_act and run new Tests
simdata2 = deepcopy(simdata)
add = linspace(0,1,T)
for (i,a) in enumerate(add)
    simdata2.traj_act[1,4,i] += a
end
simdata2.meas = PFsimulator.simulate_sensor(simdata2.traj_nom, simdata2.traj_act, simdata2.search_traj, simparams)
simresult2  = PFsimulator.run_tracking("Error in x", simdata2, simparams, pfparams, 1, savetrace = true)

@assert norm(simresult2.et) < 0.8 "norm(simresult2.et) < 0.8"



# Dual sensor tests
T_TF_M      = cat(3,T_TF_M, Rt2T(R_TF_M, [0.,20.,0]))
simparams3  = PFsimulator.SimParams(h, T, σt, σR, σq, σm, σf, force_ref, Kreact, Kcomp, T_TF_M)
simdata3    = PFsimulator.generate_simdata(traj_nom, simparams3, optimize = false)
simdata3.traj_act = simdata2.traj_act
simdata3.meas = PFsimulator.simulate_sensor(simdata3.traj_nom, simdata3.traj_act, simdata3.search_traj, simparams3)
simresult3  = PFsimulator.run_tracking("Dual sensor, error in x", simdata3, simparams3, pfparams, 1, savetrace = true)
PFsimulator.plot_trace(simresult3)

@assert norm(simresult3.et) < 0.7 "norm(simresult3.et) < 0.7 "
