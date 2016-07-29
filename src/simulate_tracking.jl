# @everywhere ENV["JULIA_PKGDIR"] = "/work/fredrikb/.julia/"
@everywhere using Robotlib
@everywhere using Plots
@everywhere include("pf_state_estimator.jl")
@everywhere import PFstateEstimator
@everywhere include("simfuns.jl")
@everywhere import PFsimulator
@everywhere pyplot()
@everywhere default(size=(1000,700))
import DSP
@everywhere using Robotlib: Rt2T, ad, adi, T2R, T2t

MCruns          = 100
@assert MCruns >= 2
# @everywhere srand(1)

# Particle filter parameters
Npart       = round(Int,logspace(3,2,MCruns))#round(Int,logspace(3,2.3,MCruns))
h           = 0.01
sigmaW      = [0.3,  0.02π/180, 1e-6] # t, R, f
sigmaV      = [0.1, 0.1, 0.1π/180] # x, y, theta
dFK         = 1.2 # plateau width
kd          = 1e-3 # extension to plateau width due to force (essentialy a super simple compliance model)
drFK        = 1π/180
sigmaVFK    = [0.15, 1π/180, dFK, kd, drFK] # xyz, theta, plateau width, norm(f)
NpartInit   = 100
force_filter= 0.1 # in [0,1], lower value = more filtering

# Simulation parameters
T           = 500
σt          = 0.4
σR          = 0.02π/180
σq          = 0*0.0001π/180 # TODO: Verkar vara något vajsing på denna, även detta sjukt lilla värdet är högst significant, kanske ska man inte köra J *= 1000 utan /1000 på orienteringselementen
σm          = 0.1sigmaV
σf          = 1
force_ref   = [1,100,500,1,1,1] #
Kreact      = zeros(6,6)
Kcomp       = diagm([1e-3,1e-3,1e-3, 0.1π/180000, 0.1π/180000, 0.1π/180000]) # Compliance matrix
Kcomp[4,2]  = 0.5*π/180000 # Force in negative y dir creates deflection around x axis
R_TF_M      = [
-1 0 0;
0  0 1;
0  1 0]
# Kcomp      += 1e-6randn(size(Kcomp))
T_TF_M      = Rt2T(R_TF_M, [0,20.,0]) # Tool flange to meta
T_TF_M      = cat(3,T_TF_M, Rt2T(R_TF_M, [0.,40.,0]))


# Construct nominal trajectories
triwave     = cumsum(sign(sin(0.05*(1:T))))
triwave   ./= maximum(triwave)
triwave     = DSP.filtfilt(ones(15),[15.], triwave)
sinwave     = sin(0.005*(1:T))
traj_nom    = PFsimulator.traj2frames3D([5triwave linspace(0,200,T) zeros(T)]')
traj_nomyz  = PFsimulator.traj2frames3D([zeros(T) linspace(0,200,T) 100sinwave]', "x")


# Setup parameter objects
simparams   = PFsimulator.SimParams(h, T, σt, σR, σq, σm, σf, force_ref, Kreact, Kcomp, T_TF_M)



# Setup main loop
seeds               = rand(1:1000,MCruns)
simresult           = Array(Any,MCruns)
simresultNoMeas     = Array(Any,MCruns)
simresult1Sensor    = Array(Any,MCruns)

simresultyz         = Array(Any,MCruns)
simresultyzNoMeas   = Array(Any,MCruns)
simresultyz1Sensor  = Array(Any,MCruns)
savetrace           = MCruns <= 20
factor              = 1


tic()
# Main loop, perform all simulations asynchronously
@sync for mc = 1:MCruns

    # Generate simulation data
    simparamsi           = PFsimulator.perturb_simparams(simparams, factor)
    simdata              = PFsimulator.generate_simdata(traj_nom, simparamsi, optimize = false)
    simdataNoMeas        = deepcopy(simdata)
    simdataNoMeas.meas  *= 0
    simparams1Sensor     = deepcopy(simparamsi)
    simparams1Sensor.T_TF_M = T_TF_M[:,:,1]
    pfparams             = PFstateEstimator.PFparams(sigmaW, sigmaV, sigmaVFK, Npart[mc], NpartInit, h, force_filter)
    PFsimulator.perturb_pfparams!(pfparams, factor)

    # Generate simulation data
    simdatayz              = PFsimulator.generate_simdata(traj_nomyz, simparamsi, optimize = false)
    simdatayzNoMeas        = deepcopy(simdatayz)
    simdatayzNoMeas.meas  *= 0

    # Run simulations
    simresult[mc]        = @spawn PFsimulator.run_tracking("xy 2 sens", simdata,       simparamsi,
    pfparams,   seeds[mc], savetrace = savetrace)
    simresultNoMeas[mc]  = @spawn PFsimulator.run_tracking("xy 0 sens",     simdataNoMeas, simparamsi,
    pfparams,   seeds[mc], savetrace = savetrace)
    simresult1Sensor[mc] = @spawn PFsimulator.run_tracking("xy 1 sens", simdata, simparams1Sensor,
    pfparams,   seeds[mc], savetrace = savetrace)

    simresultyz[mc]        = @spawn PFsimulator.run_tracking("yz 2 sens", simdatayz,       simparamsi,
    pfparams,   seeds[mc], savetrace = savetrace)
    simresultyzNoMeas[mc]  = @spawn PFsimulator.run_tracking("yz 0 sens",     simdatayzNoMeas, simparamsi,
    pfparams,   seeds[mc], savetrace = savetrace)
    simresultyz1Sensor[mc] = @spawn PFsimulator.run_tracking("yz 1 sens", simdatayz, simparams1Sensor,
    pfparams,   seeds[mc], savetrace = savetrace)

    println("Spawned MCrun ", mc)
end


# Fetch result of simulations
for mc = 1:MCruns
    simresult[mc]        = fetch(simresult[mc])
    simresultNoMeas[mc]  = fetch(simresultNoMeas[mc])
    simresult1Sensor[mc] = fetch(simresult1Sensor[mc])

    simresultyz[mc]        = fetch(simresultyz[mc])
    simresultyzNoMeas[mc]  = fetch(simresultyzNoMeas[mc])
    simresultyz1Sensor[mc] = fetch(simresultyz1Sensor[mc])
end
time = toq()
println("Done! It took $time seconds")

# Plot results

PFsimulator.plot_errors(simresultNoMeas, simresult1Sensor, simresult, simresultyzNoMeas, simresultyz1Sensor, simresultyz)
# PFsimulator.savetikz("/local/home/fredrikb/papers/PFseamTracking/errorbox.tex", PyPlot.gcf(), ["x tick label style={rotate=45,anchor=east}"])


# PFsimulator.plot(simresult[1], :tracking, :resample)
# PFsimulator.plot(simresult1Sensor[1], :tracking, :resample, :angle)
# PFsimulator.plot(simresultNoMeas[1], :tracking, :resample, :angle)


# plot_traj_sub(simresult[1].simdata.traj_nom-simresult[1].simdata.traj_act, title="Noise in traj_act")
# plot_traj_sub(simresult[1].simdata.traj_nom-simresult[1].simdata.traj_meas, title="Noise in traj_meas")





# Linear modeling
using GLM, DataFrames
import ExperimentalAnalysis
df      = PFsimulator.get_dataframe(simresultNoMeas, simresult1Sensor, simresult, simresultyzNoMeas, simresultyz1Sensor, simresultyz);
# ExperimentalAnalysis.scattermatrix(df[(df[:sens] .>= 1) & (df[:traj_type] .== "yz"),:],ex +ey+ez+er +sens ~ Npart + σW1 + σW2 + σW3 + σV1+ σV2+ σV3)
# ExperimentalAnalysis.scattermatrix(df[df[:sens] .>= 1,:],ex +ey+ez+er ~ σVFK1+ σVFK2+ σVFK3+ σVFK4+ σVFK5 )
# ExperimentalAnalysis.scattermatrix(df[df[:sens] .>= 1,:],ex +ey+ez+er ~ force_r1 + force_r2 + force_r3 + force_r4 + force_r5 + force_r6 )
# ExperimentalAnalysis.scattermatrix(df[df[:sens] .>= 1,:],ex +ey+ez+er+sens ~  σm1 + σm2 + σm3 + σf + sens)

modelx  = lm(ex~Npart + sens + et0 + σW1 + σW2 + σW3+ σm1+ σm2+ σm3 + σV1+ σV2+ σV3+ σVFK1+ σVFK2+ σVFK3+ σVFK4+ σVFK5 + force_r1 + force_r2 + force_r3 + force_r4 + force_r5 + force_r6 + σf, df)
modely  = lm(ey~Npart + sens + et0 + σW1 + σW2 + σW3+ σm1+ σm2+ σm3 + σV1+ σV2+ σV3+ σVFK1+ σVFK2+ σVFK3+ σVFK4+ σVFK5 + force_r1 + force_r2 + force_r3 + force_r4 + force_r5 + force_r6 + σf, df)
modelz  = lm(ez~Npart + sens + et0 + σW1 + σW2 + σW3+ σm1+ σm2+ σm3 + σV1+ σV2+ σV3+ σVFK1+ σVFK2+ σVFK3+ σVFK4+ σVFK5 + force_r1 + force_r2 + force_r3 + force_r4 + force_r5 + force_r6 + σf, df)
modelr  = lm(er~Npart + sens + et0 + σW1 + σW2 + σW3+ σm1+ σm2+ σm3 + σV1+ σV2+ σV3+ σVFK1+ σVFK2+ σVFK3+ σVFK4+ σVFK5 + force_r1 + force_r2 + force_r3 + force_r4 + force_r5 + force_r6 + σf, df)


# modelx  = lm(ex~Npart + sens + et0 + traj_type, df)
# modely  = lm(ey~Npart + sens + et0 + traj_type, df)
# modelz  = lm(ez~Npart + sens + et0 + traj_type, df)
# modelr  = lm(er~Npart + sens + et0 + traj_type, df)

modeld = []
for sym in [:ex, :ey, :ez, :er]
    f = $sym + (sens .== 1) - (sens .== 2) ~ Npart + et0 + traj_type + σW1*σV1 + σW2 + σW3 + σm1+ σm2+ σm3 + σV2+ σV3+ σVFK1+ σVFK2+ σVFK3+ σVFK4+ σVFK5 + σf
    push!(modeld, PFsimulator.dglm(f,df, Normal()))
    println("=== Linear modeling of differences for $sym ===")
    display(modeld[end])
end



fig = ExperimentalAnalysis.modelheatmap(["x", "y", "z", "R"], modelx, modely, modelz, modelr)
# PFsimulator.savetikz("/local/home/fredrikb/papers/PFseamTracking/heatmap.tex", PyPlot.gcf(), ["x tick label style={rotate=45,anchor=east}"])

ExperimentalAnalysis.modelheatmap(["ex", "dy", "dz", "dR"], modeld...)
# PFsimulator.savetikz("/local/home/fredrikb/papers/PFseamTracking/heatmapdiff.tex", PyPlot.gcf(), ["x tick label style={rotate=45,anchor=east}"])


scatter(df[:Npart][(df[:sens2].==true) & (df[:traj_type].=="xy")],df[:ey][(df[:sens2].==true) & (df[:traj_type].=="xy")])



# file = open("/work/fredrikb/savefile_varyN","w")
# serialize(file,(simresultNoMeas, simresult1Sensor, simresult, simresultyzNoMeas, simresultyz1Sensor, simresultyz))
# close(file)

# file = open("/work/fredrikb/savefile_varyN","r")
# (simresultNoMeas, simresult1Sensor, simresult, simresultyzNoMeas, simresultyz1Sensor, simresultyz) = deserialize(file);
# close(file)
#

savetrace && PFsimulator.plot_trace(simresult[1], xIndices = 1:3, yIndices = 1:3)
pyplot()
