module PFsimulator
using Robotlib
using DataFrames
# using Debug
using DSP
using Plots
import Plots.plot
using PFstateEstimator
import PyPlot
import Convex
import SCS
using GLM, DataFrames

include("particlePlot.jl")
export SimParams, SimData, generate_simdata, randse3, rand_traj, perturb_traj, traj2frames1D, traj2frames3D, ctraj_intrap, simulate_force, simulate_sensor, plot_tracking, run_tracking, get_search_traj, plot_errors, plot_meas, plot_trace, get_dataframe

sq(T) = squeeze(T[1:3,4,:],2)'
rms(x::Matrix) = sqrt(mean(x.^2,1))[:]
rms(x::Vector) = sqrt(mean(x.^2))

type SimParams
    h       # Sample time
    T       # Time steps
    σt      # std for translation error in FK
    σR      # std for rotation error in FK
    σq      # std for resolver/encoder error in joints
    σm      # Sensor noise
    σf      # Force noise
    force_ref
    Kreact  # Reactive force, f_react = Kreact * f
    Kcomp   # Compliance matrix
    T_TF_M  # Tool flange to sensor
end

type SimData{T}
    force::AbstractMatrix{T}
    meas::AbstractArray{T,3}
    traj_act::AbstractArray{T,3}
    traj_nom::AbstractArray{T,3}
    traj_meas::AbstractArray{T,3}
    search_traj::Vector{Matrix{T}}
    traj_type::Symbol
end

type SimResult{T}
    name::AbstractString
    x̂::AbstractArray{T,3}
    se::AbstractMatrix{T}
    et::AbstractVector{T}
    er::T
    simdata::SimData
    simparams::SimParams
    pfparams::PFparams
    trace::AbstractVector{PFtrace}
end
SimResult(name, x̂, se, et, er, simdata, simparams, pfparams) = SimResult(name, x̂, se, et, er, simdata, simparams, pfparams,PFtrace[])

# """
# `generate_simdata(traj_nom, p::SimParams)`
# """


function generate_simdata(traj_nom, p::SimParams; optimize = false)
    force       = simulate_force(traj_nom, p)
    traj_act, traj_meas = perturb_traj(traj_nom, force, p)
    search_traj = get_search_traj(traj_nom, p.T_TF_M, optimize)
    meas        = simulate_sensor(traj_nom, traj_act, search_traj, p)
    singular_axis = svd(squeeze(traj_nom[1:3,4,:],2))[1][:,3]
    traj_type   = singular_axis ⋅ [0,0,1] ≈ 1 ? :xy : :yz

    return SimData(force, meas, traj_act, traj_nom, traj_meas, search_traj,traj_type)
end

function randse3(σt,σR)
    R = expω(skew(σR*randn(3)))
    t = σt*randn(3)
    return [R t; 0 0 0 1]
end


function ctraj_intrap(N,T_start, T_end)
    T_out = zeros(4,4,N) # Initialize output trajectory
    xi_start = twistcoords(logT(T_start)) # Convert to start wrench
    xi_end = twistcoords(logT(T_end)) # Convert to end wrench
    xi_traj = zeros(6,N) # Initialize wrench trajectory
    for i = 1:6
        xi_traj[i,:] = linspace(xi_start[i], xi_end[i], N)
    end
    for n = 1:N
        T_out[:,:,n] = expm(skew4(xi_traj[:,n]))
    end
    return T_out
end


"""
`traj2frames1D(traj_in, increment)`
Returns frames along traj_in, with y along seam, z pointing down towards
the work object, and x is chosen so that the frames are orthonormal,
right oriented. T_out orientations are expressed in base frame.
"""
function traj2frames1D(traj_in, increment)
    N = length(traj_in)
    T_out = zeros(4,4,N) # Initialize output frames
    # differentiate:
    dx = centralDiff(traj_in)
    dy = convert(Float64, increment)
    for n = 1:N
        ŷ = [dx[n]; dy; 0.0] / norm([dx[n]; dy; 0.0]) # Points along seam
        ẑ = [0, 0, -1.0] # Always pointing down
        x̂ = ŷ × ẑ # Guarantees orthonormal, right oriented frames
        T_out[1:3, 1:3, n] = [x̂ ŷ ẑ]
        T_out[1:3, 4, n] = [traj_in[n], dy*(n-1), 0]
    end
    T_out[4,4,:] = 1
    return T_out
end


# TODO: test more thoroughly:
"""
´traj2frames3D(traj_in, fix = "z")´
Returns frames along traj_in (3*N matrix), with y along seam, z pointing down towards
the work object, and x is chosen so that the frames are orthonormal,
right oriented. T_out orientations are expressed in base frame. Fix is the direction in
which the seam does not vary.
"""
function traj2frames3D(traj_in, fix = "z")

    if size(traj_in,1) != 3
        traj_in = traj_in'
        if size(traj_in,1) != 3
            println("Input argument traj_in should be a 3*N matrix")
        end
    end

    N = size(traj_in,2)
    T_out = zeros(4,4,N) # Initialize output frames
    # differentiate to get seam direction:
    traj_dir = centralDiff(traj_in')'
    # Normalize to get unit direction vectors
    for n = 1:N
        traj_dir[:,n] /= norm(traj_dir[:,n])
    end
    for n = 1:N
        ŷ = vec(traj_dir[:,n]) # Points along seam
        if fix == "z"
            ẑ = [0, 0, -1.0]
            x̂ = ŷ × ẑ
        else
            x̂ = [-1., 0, 0]
            ẑ = x̂ × ŷ
        end
        T_out[1:3, 1:3, n] = [x̂ ŷ ẑ]
        T_out[1:3, 4, n] = traj_in[:,n]
    end
    T_out[4,4,:] = 1
    return T_out
end

function perturb_traj(traj_nom, force, p::SimParams)
    # Simulate low-frequency noise:
    fkin_noise          = similar(traj_nom)
    fkin_noise_twist    = zeros(6,size(traj_nom,3))
    for i = 1:size(traj_nom,3)
        fkin_noise[:,:,i]       = randse3(p.σt,p.σR)
        # fkin_noise[:,:,i] = randse3(1,1)
        fkin_noise_twist[:,i]   = twistcoords(logm(fkin_noise[:,:,i]))
    end

    #low-pass filter noise:
    L = min(500,p.T)
    fkin_noise_twist_filt = DSP.filtfilt(ones(L),[(L)], fkin_noise_twist')'
    fkin_noise_twist_filt = min(fkin_noise_twist_filt,1) # TODO: link this to dFK or set a max_error in simparams
    fkin_noise_twist_filt = max(fkin_noise_twist_filt,-1)


    # Convert fkin noise back to Ti:
    for i = 1:1:size(fkin_noise_twist_filt,2)
        fkin_noise[:,:,i] = expm(skew4(fkin_noise_twist_filt[:,i]))
    end

    # Addressing noise affecting resolvers/encoders:
    eq                  = p.σq*randn(6,size(traj_nom,3))
    xi                  = DH2twistsPOE(DH7600())
    q_center_of_traj    = ikinePOE(xi, traj_nom[:,:,round(Int,size(traj_nom,3)/2)], ones(6,1))[1] #ones seems to work as initial guess, but not zeros
    J                   = jacobianPOE(q_center_of_traj,xi)[1]
    J[1:3,:]           *=1000
    J[4:6,:]           /=1000
    e_C_twist           = J*eq # The covariance matrix of e_C is then J*σq*J'

    # TODO: perturb due to force, simulate deflections due to forces DONE!!
    force_perturb_twist =  p.Kcomp*force
    traj_meas           = similar(traj_nom)
    traj_act            = similar(traj_nom)
    force_perturb       = similar(traj_nom)
    e_C                 = similar(traj_nom)
    # Perturb traj_act and traj_meas:
    for i = 1:size(traj_nom,3)
        e_C[:,:,i]            = expm(skew4(e_C_twist[:,i]))
        force_perturb[:,:,i]  = expm(skew4(force_perturb_twist[:,i]))
        traj_meas[:,:,i]      = traj_nom[:,:,i]*e_C[:,:,i] # Noise affecting traj_meas
        traj_act[:,:,i]       = traj_nom[:,:,i]*fkin_noise[:,:,i]
        traj_act[1:3,1:3,i]   = traj_act[1:3,1:3,i]*force_perturb[1:3,1:3,i]
        traj_act[1:3,4,i]    += force_perturb[1:3,4,i]
        #resolver/encoder noise, and force perturbations, affecting traj_act
    end
    return traj_act, traj_meas
end

function perturb_simparams(si::SimParams, f)
    # force_ref
    s    = deepcopy(si)
    s.σt = rand(logspace(log10(s.σt/f),log10(s.σt*f),1000000))
    s.σR = rand(logspace(log10(s.σR/f),log10(s.σR*f),1000000))
    s.σq = rand(logspace(log10(s.σq/f),log10(s.σq*f),1000000))
    s.σm = Float64[rand(logspace(log10(s.σm[i]/f),log10(s.σm[i]*f),1000000)) for i = eachindex(s.σm)]
    s.σf = rand(logspace(log10(s.σf/f),log10(s.σf*f),1000000))
    s.force_ref = Float64[rand(logspace(log10(s.force_ref[i]/f),log10(s.force_ref[i]*f),1000000)) for i = eachindex(s.force_ref)]
    # s.T_TF_M[2,3,:] = Float64[rand(logspace(log10(s.T_TF_M[2,3,i]/f),log10(s.T_TF_M[2,3,i]*f),1000000)) for i = eachindex(s.T_TF_M[2,3,:])]
    return s
end

function perturb_pfparams!(p::PFparams, f)
    p.σW    = Float64[rand(logspace(log10(p.σW[i]/f),log10(p.σW[i]*f),1000000)) for i = eachindex(p.σW)]
    p.σV    = Float64[rand(logspace(log10(p.σV[i]/f),log10(p.σV[i]*f),1000000)) for i = eachindex(p.σV)]
    p.σVFK  = Float64[rand(logspace(log10(p.σVFK[i]/f),log10(p.σVFK[i]*f),1000000)) for i = eachindex(p.σVFK)]
end

function simulate_force(traj_nom, p::SimParams)
    return p.force_ref .+ p.σf.*randn(6,p.T) # + p.Kreact*p.force_ref
end


# TODO: Skriv vettiga test för var funktion i denna filen!!!
function simulate_sensor(traj_nom, traj_act, search_traj, p::SimParams)
    T_TF_M      = p.T_TF_M
    Nsensors    = size(T_TF_M,3)
    meas        = zeros(3,p.T,Nsensors)
    for j = 1:Nsensors
        meas[:,:,j] = PFstateEstimator.distToSeam([0,0], search_traj[j], traj_act, T_TF_M[:,:,j], tries = 5)[2]'
        for i = 1:p.T
            meas[3,i,j] = xyθ(traj_nom[:,:,i]*T_TF_M[:,:,j], traj_act[:,:,i]*T_TF_M[:,:,j]) # TODO: minus sometimes works better, find out the convention
        end
    end
    nZ = meas .!= 0
    meas += (p.σm.*randn(3,p.T,Nsensors)).*nZ
    return meas
end



function run_tracking(name, simdata, simparams, pfparams, seed=1; savetrace = true)
    srand(seed)
    T       = simparams.T
    x̂       = zeros(4,4,T)
    se      = zeros(3,T)
    meas    = permutedims(simdata.meas,[1,3,2])
    tic()
    state = init_pf(simdata.traj_meas[:,:,1], pfparams.Npart*pfparams.NpartInit, pfparams.σW, meas[:,:,1], simparams.T_TF_M)
    if savetrace
        trace   = Array(PFtrace,T)
        for i = 1:T
            x̂[:,:,i], se[:,i], state, trace[i] = PFstateEstimator.pfStateEstimator(state, meas[:,:,i], simdata.force[:,i], simdata.traj_meas[:,:,i], simdata.traj_nom, simdata.search_traj, simparams.T_TF_M, pfparams, savetrace = true)
        end
    else
        for i = 1:T
            x̂[:,:,i], se[:,i], state           = PFstateEstimator.pfStateEstimator(state, meas[:,:,i], simdata.force[:,i], simdata.traj_meas[:,:,i], simdata.traj_nom, simdata.search_traj, simparams.T_TF_M, pfparams, savetrace = false)
        end
    end
    t = toq()
    et, er = tracking_error(x̂, simdata)
    println(name, " took ",t, " seconds. Error ",round(et,3), "   ", round(er,3))

    savetrace ? SimResult(name, x̂, se, et, er, simdata, simparams, pfparams, trace) : SimResult(name, x̂, se, et, er, simdata, simparams, pfparams)

end



function plot_tracking(simresult)
    x̂ = simresult.x̂
    se = simresult.se
    time = linspace(0,simresult.simparams.h*simresult.simparams.T,simresult.simparams.T)
    p1 = subplot(time,sq(x̂),c=:blue,lab="x̂",nc=1, title=simresult.name)
    subplot!(time,sq(simresult.simdata.traj_act), c=:red, lab="traj_act")
    subplot!(time,sq(simresult.simdata.traj_nom),c=:green,lab="traj_nom")
    subplot!(time,sq(x̂)+2se',c=:blue,lab="+2σx̂", l=:dash)
    subplot!(time,sq(x̂)-2se',c=:blue,lab="-2σx̂", l=:dash)

    p = subplot(time,traj2quat(x̂)',c=:blue,lab="x̂",nc=1, title=simresult.name*" rotation")
    subplot!(time,traj2quat(simresult.simdata.traj_act)', c=:red, lab="traj_act")
    subplot!(time,traj2quat(simresult.simdata.traj_nom)',c=:green,lab="traj_nom")

    return p1
end

plot_angle(simresult) = plot(Rangle(simresult.x̂,simresult.simdata.traj_act,true),nc=1, title=simresult.name, lab="Angle (degrees)")

function plot_meas(dim,simparams,simdata,base="M")
    if dim == 0
        dim = 1:size(simparams.T_TF_M,3)
    end
    for d in dim
        T_TF_M = simparams.T_TF_M[:,:,d]
        if base == "M"
            plot(simdata.meas[:,:,d]')
        elseif base == "RB"
            meas = zeros(3,simparams.T)
            for i = 1:simparams.T
                meas[:,i] = (trinv(simdata.traj_nom[:,:,i]*T_TF_M[:,:,d])*[simdata.meas[1:2,i,d];0;1])[1:3]
            end
            plot(meas[1:3,:]', lab="Sensor $d")
        end
    end
end

function plot_meas(dim, simresult::SimResult, base="M")
    plot_meas(dim,simresult.simparams,simresult.simdata,base)
    title!(simresult.name)
end

function get_dataframe(simresultlists...; onehot=false)
    # σW::Vector{T}       # noise in state update
    # σV::Vector{T}       # Noise in sensor reading
    # σVFK::Vector{T}     # Noise in FK
    # σt      # std for translation error in FK
    # σR      # std for rotation error in FK
    # σq      # std for resolver/encoder error in joints
    # σm      # Sensor noise
    # σf      # Force noise
    # force_ref
    frame   = DataFrame()
    Nexp    = length(simresultlists)
    MC      = length(simresultlists[1])
    et      = [simresultlists[i][mc].et for i in 1:Nexp, mc in 1:MC]
    et0     = [norm(simresultlists[i][mc].simdata.traj_act[1:3,4,1]-simresultlists[i][mc].simdata.traj_nom[1:3,4,1]) for i in 1:Nexp, mc in 1:MC]
    names   = [simresultlists[i][mc].name for i in 1:Nexp, mc in 1:MC]
    traj_type = [string(simresultlists[i][mc].simdata.traj_type) for i in 1:Nexp, mc in 1:MC]
    ex      = Float64[et[i,mc][1] for i in 1:Nexp, mc in 1:MC]
    ey      = Float64[et[i,mc][2] for i in 1:Nexp, mc in 1:MC]
    ez      = Float64[et[i,mc][3] for i in 1:Nexp, mc in 1:MC]
    er      = Float64[simresultlists[i][mc].er for i in 1:Nexp, mc in 1:MC]
    et0     = Float64[et0[i,mc] for i in 1:Nexp, mc in 1:MC]

    Npart   = [simresultlists[i][mc].pfparams.Npart for i in 1:Nexp, mc in 1:MC]
    σW1      = [simresultlists[i][mc].pfparams.σW[1] for i in 1:Nexp, mc in 1:MC]
    σW2      = [simresultlists[i][mc].pfparams.σW[2] for i in 1:Nexp, mc in 1:MC]
    σW3      = [simresultlists[i][mc].pfparams.σW[3] for i in 1:Nexp, mc in 1:MC]
    σV1      = [simresultlists[i][mc].pfparams.σV[1] for i in 1:Nexp, mc in 1:MC]
    σV2      = [simresultlists[i][mc].pfparams.σV[2] for i in 1:Nexp, mc in 1:MC]
    σV3      = [simresultlists[i][mc].pfparams.σV[3] for i in 1:Nexp, mc in 1:MC]
    σVFK1    = [simresultlists[i][mc].pfparams.σVFK[1] for i in 1:Nexp, mc in 1:MC]
    σVFK2    = [simresultlists[i][mc].pfparams.σVFK[2] for i in 1:Nexp, mc in 1:MC]
    σVFK3    = [simresultlists[i][mc].pfparams.σVFK[3] for i in 1:Nexp, mc in 1:MC]
    σVFK4    = [simresultlists[i][mc].pfparams.σVFK[4] for i in 1:Nexp, mc in 1:MC]
    σVFK5    = [simresultlists[i][mc].pfparams.σVFK[5] for i in 1:Nexp, mc in 1:MC]


    σt    = [simresultlists[i][mc].simparams.σt for i in 1:Nexp, mc in 1:MC]
    σR    = [simresultlists[i][mc].simparams.σR for i in 1:Nexp, mc in 1:MC]
    σq    = [simresultlists[i][mc].simparams.σq for i in 1:Nexp, mc in 1:MC]
    σm1    = [simresultlists[i][mc].simparams.σm[1] for i in 1:Nexp, mc in 1:MC]
    σm2    = [simresultlists[i][mc].simparams.σm[2] for i in 1:Nexp, mc in 1:MC]
    σm3    = [simresultlists[i][mc].simparams.σm[3] for i in 1:Nexp, mc in 1:MC]
    σf    = [simresultlists[i][mc].simparams.σf for i in 1:Nexp, mc in 1:MC]
    force_ref1    = [simresultlists[i][mc].simparams.force_ref[1] for i in 1:Nexp, mc in 1:MC]
    force_ref2    = [simresultlists[i][mc].simparams.force_ref[2] for i in 1:Nexp, mc in 1:MC]
    force_ref3    = [simresultlists[i][mc].simparams.force_ref[3] for i in 1:Nexp, mc in 1:MC]
    force_ref4    = [simresultlists[i][mc].simparams.force_ref[4] for i in 1:Nexp, mc in 1:MC]
    force_ref5    = [simresultlists[i][mc].simparams.force_ref[5] for i in 1:Nexp, mc in 1:MC]
    force_ref6    = [simresultlists[i][mc].simparams.force_ref[6] for i in 1:Nexp, mc in 1:MC]



    sens0   = [all(simresultlists[i][mc].simdata.meas .== 0) for i in 1:Nexp, mc in 1:MC]
    sens1   = [!sens0[i,mc] && size(simresultlists[i][mc].simparams.T_TF_M,3) == 1 for i in 1:Nexp, mc in 1:MC]
    sens2   = [!sens0[i,mc] && size(simresultlists[i][mc].simparams.T_TF_M,3) == 2 for i in 1:Nexp, mc in 1:MC]
    sens    = map(Int,sens1) + 2map(Int,sens2)

    frame[:names]       = names[:]
    frame[:ex]          = ex[:]
    frame[:ey]          = ey[:]
    frame[:ez]          = ez[:]
    frame[:er]          = er[:]
    frame[:et0]         = et0[:]

    frame[:Npart]       = Npart[:]
    frame[:logNpart]    = map(log10,Npart[:])
    frame[:sens]        = sens[:]
    frame[:sens0]       = sens0[:]
    frame[:sens1]       = sens1[:]
    frame[:sens2]       = sens2[:]
    frame[:traj_type]   = traj_type[:]
    frame[:σW1]         = σW1[:]
    frame[:σW2]         = σW2[:]
    frame[:σW3]         = σW3[:]
    frame[:σV1]         = σV1[:]
    frame[:σV2]         = σV2[:]
    frame[:σV3]         = σV3[:]
    frame[:σVFK1]       = σVFK1[:]
    frame[:σVFK2]       = σVFK2[:]
    frame[:σVFK3]       = σVFK3[:]
    frame[:σVFK4]       = σVFK4[:]
    frame[:σVFK5]       = σVFK5[:]
    frame[:σt]          = σt[:]
    frame[:σR]          = σR[:]
    frame[:σq]          = σq[:]
    frame[:σm1]         = σm1[:]
    frame[:σm2]         = σm2[:]
    frame[:σm3]         = σm3[:]
    frame[:σf]          = σf[:]
    frame[:force_r1]  = force_ref1[:]
    frame[:force_r2]  = force_ref2[:]
    frame[:force_r3]  = force_ref3[:]
    frame[:force_r4]  = force_ref4[:]
    frame[:force_r5]  = force_ref5[:]
    frame[:force_r6]  = force_ref6[:]

    pool!(frame, [:names])
    pool!(frame, [:traj_type])
    return frame
end

function plot_errors(simresultlists...)
    Nexp    = length(simresultlists)
    MC      = length(simresultlists[1])
    et      = [simresultlists[i][mc].et for i in 1:Nexp, mc in 1:MC]
    names   = [simresultlists[i][1].name for i in 1:Nexp]
    ex      = Float64[et[i,mc][1] for i in 1:Nexp, mc in 1:MC]
    ey      = Float64[et[i,mc][2] for i in 1:Nexp, mc in 1:MC]
    ez      = Float64[et[i,mc][3] for i in 1:Nexp, mc in 1:MC]
    er      = Float64[simresultlists[i][mc].er for i in 1:Nexp, mc in 1:MC]
    PyPlot.figure()
    PyPlot.subplot(2,2,1)
    PyPlot.boxplot(ex',labels=names), PyPlot.title("Error x")
    PyPlot.subplot(2,2,2)
    PyPlot.boxplot(ey',labels=names), PyPlot.title("Error y")
    PyPlot.subplot(2,2,3)
    PyPlot.boxplot(ez',labels=names), PyPlot.title("Error z")
    PyPlot.subplot(2,2,4)
    PyPlot.boxplot(er',labels=names), PyPlot.title("Error rot")
    # PyPlot.show()

    # subplot([mx my mz mr], t=:bar, title=["Mean error X" "Mean error y" "Mean error z" "Mean error r"], xlabel=names)
    # subplot([max may maz mar], t=:bar, title=["Max error X" "Max error y" "Max error z" "Max error r"], xlabel=names)
    nothing
end


function plot_resample(res)
    N = res.pfparams.Npart
    plot([res.trace[i].Neff for i = 1:res.simparams.T],lab="Effective number of particles", title=res.name)
    scatter!(N*[res.trace[i].resample for i = 1:res.simparams.T], c=:green, lab="Resample")
    scatter!(1.2N*[res.trace[i].reset for i = 1:res.simparams.T]-0.1N, c=:red, m=:rect, lab="Reset")
    scatter!(1.4N*[res.trace[i].runMeasUpdate for i = 1:res.simparams.T]-0.2N, c=:magenta, m=:circle, lab="Measurement update")
end

function plot_trace(res; xIndices = 1:3, yIndices = 1:3)
    T       = res.simparams.T
    N       = res.pfparams.Npart
    y       = res.simdata.meas[:,:,1]
    xreal   = [sq(res.simdata.traj_act)'; R2rpy(res.simdata.traj_act,deg=true)]
    pdata   = Void
    T_TF_M  = res.simparams.T_TF_M[:,:,1]
    for i = 1:T
        ti      = res.trace[i]
        x       = [sq(ti.x)'; R2rpy(ti.x,deg=true)]
        w       = ti.w
        a       = ti.resample_i
        yhat    = PFstateEstimator.distToSeam([0,0], res.simdata.search_traj[1], ti.x, T_TF_M, tries = 20)[2]'
        for j = 1:N
            yhat[3,j] = xyθ(res.simdata.traj_nom[:,:,i]*T_TF_M, ti.x[:,:,j]*T_TF_M)
        end
        pdata = plotPoints(x, w, y, yhat, N, a, i, xreal, 0, 0, pdata, density = true, xIndices = xIndices, yIndices = yIndices, leftOnly = true)
    end

end

function Plots.plot(res::SimResult, args...)
    pyplot()
    for arg in args
        arg == :meas && plot_meas(0,res)
        arg == :tracking && plot_tracking(res)
        arg == :resample && plot_resample(res)
        arg == :angle && plot_angle(res)
        #TODO: nom, act, meas, x̂
    end
    if !isempty([:nom, :act, :traj_meas] ∩ args)
        subplot(n=3,nc=1)
        for arg in args
            arg == :nom && subplot!(sq(res.simdata.traj_nom), c=:green, lab="traj_nom")
            arg == :act && subplot!(sq(res.simdata.traj_act), c=:red, lab="traj_act")
            arg == :traj_meas && subplot!(sq(res.simdata.traj_meas), c=:magenta, lab="traj_meas")
        end
    end
end

function tracking_error(x̂, simdata)
    et = rms(sq(x̂) - sq(simdata.traj_act))
    er = rms(Rangle(x̂,simdata.traj_act,true))
    return et,er
end

# """
# This function calculates an efficient search trajectory.
# """

function get_search_traj(traj_nom,T_TF_M, optimize = false)
    Nsensors = size(T_TF_M,3)
    search_traj = deepcopy(traj_nom)
    search_traj = cat(4,search_traj,search_traj)
    for ns = 1:Nsensors
        for i = 1:size(search_traj,3)
            search_traj[:,:,i,ns] = search_traj[:,:,i,ns]*T_TF_M[:,:,ns]
        end
    end
    search_traj = permutedims(squeeze(search_traj[1:3,4,:,:],2),[2 1 3])
    rettraj = Array(Array{Float64,2},Nsensors)
    if !optimize
        for ns = 1:Nsensors
            D = [conv(diff(diff(search_traj[:,1,ns])),ones(3)) conv(diff(diff(search_traj[:,2,ns])),ones(3)) conv(diff(diff(search_traj[:,3,ns])),ones(3))]
            inds = any(abs(D) .> 1e-3,2)
            rettraj[ns] = search_traj[:,:,ns]
        end
        return rettraj
    end
    for ns = 1:Nsensors
        # return search_traj
        N = size(search_traj,1)
        Tv = Convex.Variable(size(search_traj,1,2))
        diffs = Tv-search_traj[:,:,ns]
        f = Convex.sumsquares(diffs)

        for i = 1:N-2
            Di = abs(Tv[i,:] -2Tv[i+1,:] + Tv[i+2,:])
            # D[i,:] = [Di[1],Di[2],Di[3]]
            # f += 100*max(Di[1],Di[2],Di[3])
            f += 1*sum(Di)
        end
        p = Convex.minimize(f,[Convex.norm(diffs,Inf) < 0.05])
        Convex.solve!(p,SCS.SCSSolver(verbose=false, max_iters=100000, normalize=false, eps = 1e-5))
        if p.status != :Optimal
            warn("Optimization did not converge in first step, using full search_traj, status: ", p.status)
            rettraj[ns] = search_traj[:,:,ns]
            continue
        end
        T = Tv.value
        D = diff(diff(T))
        if false
            Plots.subplot(search_traj[:,:,ns],layout=[1,1,1])
            Plots.subplot!(T,c=:red)
            Plots.subplot(D,layout=[1,1,1])
        end

        diffs = Tv-search_traj[:,:,ns]
        f = Convex.sumsquares(diffs)


        c = Convex.norm(diffs,Inf) < 0.05 # Constrain the approximation error to be no more than this
        nulls = falses(N)
        for i = 2:N-1
            if abs(D[i-1]) < 0.0002
                f += 100*sum(Tv[i-1,:] -2Tv[i,:] + Tv[i+1,:])
                nulls[i] = true
            end
        end
        p = Convex.minimize(f,c)
        Convex.solve!(p,SCS.SCSSolver(verbose=false, max_iters=150000, normalize=false, eps = 1e-5))
        if p.status != :Optimal
            warn("Optimization did not converge in second step, using full search_traj, status: ", p.status)
            rettraj[ns] = search_traj[:,:,ns]
            continue
        end
        T = Tv.value
        rettraj[ns] = T[!nulls,:]

        if false
            Plots.subplot(search_traj[:,:,ns],layout=[1,1,1],c=:blue,lab="search_traj")
            Plots.subplot!(T,c=:red,lab="optresult")
            # Plots.subplot(D,layout=[1,1,1])

            Plots.subplot!(find(!nulls),rettraj[ns],c=:green,lab="rettraj")
            gui()
        end
    end
    return rettraj
end



vars(vs, _) = vs
vars(vs, s::Symbol) = isdefined(s) ? vs : push!(vs, s)
function vars(vs, e::Expr)
    for arg in e.args
        vars(vs, arg)
    end
    vs
end
extractvars(ex::Expr) = vars(Set{Symbol}(), ex)

# """
# Write formulas on the form
#
# `f = response + (var1 .== val1) - (var1 .== val2) ~ rhs`
# """

function dglm(f::Formula,  df::DataFrame, args...)
    l = f.lhs.args
    # Assert operator is minus
    @assert l[1] == :-
    response    = l[2].args[2]
    diffsym = symbol("diff_",response)

    cond1       = l[2].args[3].args
    cond1symb   = cond1[1]
    comparison1 = cond1[2]
    value1      = cond1[3]

    cond2       = l[3].args
    cond2symb   = cond2[1]
    comparison2 = cond2[2]
    value2      = cond2[3]
    # Determine the indices
    ind1 = eval(Expr(:comparison, df[cond1symb], comparison1, value1))
    ind2 = eval(Expr(:comparison, df[cond2symb], comparison2, value2))
    @assert (sum(ind1) == sum(ind2)) "The two conditions did not evaluate to true for the same number of elements"
    diff = df[response][ind1] - df[response][ind2]
    dfd = DataFrame()
    dfd[diffsym] = diff
    V = extractvars(f.rhs)
    for sym = names(df)
        # Only add the relevant symbols to the new dataframe
        if sym ∉ V && sym ∉ f.rhs.args # Super weird bug here. Sometimes the first condition evaluates to false, but the if-statement is still entered!!
            continue
        end
        # @bp all(df[sym][ind1] .!= df[sym][ind2])
        @assert all(df[sym][ind1] .== df[sym][ind2]) "The entries considered for modeling does not have equal values in the two selected datasets"
        dfd[sym] = df[sym][ind1]
    end
    glm($(diffsym) ~ $(f.rhs), dfd, args...)

end



using PyCall
@pyimport matplotlib2tikz
function savetikz(path, fig = PyPlot.gcf(), extra=[""])
    matplotlib2tikz.save(path,fig, figureheight = "\\figureheight", figurewidth = "\\figurewidth", extra = pybuiltin("set")(extra))
end

end
