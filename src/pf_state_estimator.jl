

"""
This module exports
PFparams, pfStateEstimator

`pfStateEstimator(meas, force, T_RB_TF, search_traj, T_TF_M, params::PFparams)`
Run state estimator based on particle filter\n
`Npart`: number of particles\n
`meas`: measurement from laser sensor\n
`T_RB_TF`: position of the robot tool flange from FK, in R4x4\n
`search_traj`: nominal trajectory in RTx3
"""
module PFstateEstimator
using Robotlib
# using Debug
using Plots
# using Devectorize
include("pf_funs.jl")
using StatsBase # logsumexp

export PFparams, PFtrace, PFstate, pfStateEstimator, init_pf

"""
σW::Vector{T}       # noise in state update\n
σV::Vector{T}       # Noise in sensor reading\n
σVFK::Vector{T}     # Noise in FK\n
Npart::Int              # Number of particles\n
NpartInit::Int          # Increasing number of particles with this factor during initialization
h::Float64              # Sample time\n
force_filter::Float64
"""
type PFparams{T}
    σW::Vector{T}       # noise in state update
    σV::Vector{T}       # Noise in sensor reading
    σVFK::Vector{T}     # Noise in FK
    Npart::Int              # Number of particles
    NpartInit::Int          # Increasing number of particles with this factor during initialization
    h::Float64              # Sample time
    force_filter::Float64   #
end

type PFtrace
    Neff::Float64
    resample::Bool
    x::Array{Float64,3}
    w::Array{Float64,1}
    reset::Bool
    resample_i::AbstractVector{Int}
    runMeasUpdate::Bool
end

type PFstate
    x::Array{Float64,3}
    w::Vector{Float64}
    expw::Vector{Float64}
    T_RB_TF_last::Matrix{Float64}
    meas_last::VecOrMat{Float64}
    force_last::Vector{Float64}
    force_filt::Vector{Float64}
    t::Int
end
# type PFtrace
# end

function init_pf(T_RB_TF, Npart, σW, meas, T_TF_M)
    # m_TF            = (T_RB_TF[1:3,1:3]*T_TF_M[1:3,1:3,1])*[meas[1:2,1];0]
    x_init          = T_RB_TF
    # x_init[1:3,4]  += 0.5*m_TF
    x,w,expw        = initParticles(x_init,Npart,σW)
    T_RB_TF_last = T_RB_TF
    meas_last       = zeros(meas)
    force_last      = zeros(6)
    force_filt      = zeros(6)
    return PFstate(x,w,expw,T_RB_TF_last,meas_last, force_last, force_filt,1)
end

# """
# `pfStateEstimator(meas, force, T_RB_TF, search_traj, T_TF_M, params::PFparams)`
# Run state estimator based on particle filter\n
# `Npart`: number of particles\n
# `meas`: measurement from laser sensor\n
# `T_RB_TF`: position of the robot tool flange from FK, in R4x4\n
# `search_traj`: nominal trajectory in RT×3
# """

function pfStateEstimator(s::PFstate, meas, force, T_RB_TF, traj_nom, search_traj, T_TF_M, params::PFparams; savetrace=false)

    x               = s.x
    w               = s.w
    expw            = s.expw
    T_RB_TF_last    = s.T_RB_TF_last
    meas_last       = s.meas_last
    force_last      = s.force_last
    force_filt      = s.force_filt
    σW              = params.σW
    σV              = params.σV
    σVFK            = params.σVFK
    Npart           = params.Npart
    NpartInit       = params.NpartInit
    h               = params.h
    force_filter    = params.force_filter
    #     m_TF = T_TF_M\[meas';0;1]

    force_filt = force_filter/h*abs(force-force_last) + (1-force_filter)*force_filt
    runMeasUpdate = any(meas .!= 0)

    # Resample--------------------------------------------------------------


    Neff = (1/sum(expw.^2))
    resample = Neff < Npart/2 || s.t == 2

    if resample
        j   = resample_systematic_exp(expw,Npart)
        x[:,:,1:Npart] = x[:,:,j]
        w[1:Npart]  = w[j]
        #x = x[:,:,j]
        #w = w[j]
        if s.t == 2
            x = x[:,:,1:Npart]
            w = w[1:Npart]
            expw = expw[1:Npart]
            s.x = x
            s.w = w
            s.expw = expw
        end
    else
        j = 1:size(x,3)
    end

    # Time update-----------------------------------------------------------
    increment   = T_RB_TF*trinv(T_RB_TF_last)
    # σWt     = (increment[1:3,4] + √(h)).*σW[1] + √(h)*σW[3]*force_filt[1:3] # TODO: This line is ad-hoc
    σWt         = σW[1].*ones(3)
    timeUpdate!(x,increment,σWt,σW[2],h)

    # Meas update-----------------------------------------------------------
    if runMeasUpdate
        for i = 1:size(T_TF_M,3)
            x_μ2, mx_μ, index, γ   = distToSeam(meas[1:2,i], search_traj[i], x, T_TF_M[:,:,i], tries = 1)
            wU      = measUpdate(mx_μ, meas[3,i], x, traj_nom, σV, index, γ, T_TF_M[:,:,i])
            # wU[wU .< -12] = -8 # Robustify against outliers
            # wU[(meas[1,i] != 0) & (x_μ2 .== 0)] -= 1 # If a measurement is obtained, but the seam not found for a particle, punish that particle
            # do NOT add opposite kind of error, no sensor will not work then
            w    += wU
        end
    end

    w    += measFKUpdate(x,T_RB_TF,force,σVFK)
    w,expw[:]  = normalizeWeights!(w)

    # # Reset the particle filter---------------------------------------------
    reset = any(isnan(w)) || any(isinf(w)) #|| (1/sum(expw.^2)) < 2
    # if reset
    #     m_TF            = (T_RB_TF[1:3,1:3]*T_TF_M[1:3,1:3,1])'*[meas[1:2,1];0]
    #     x_init          = T_RB_TF
    #     x_init[1:3,4]  -= m_TF
    #     x, w, expw      = initParticles(x_init,Npart,σW)
    # end


    # Calculate state estimate----------------------------------------------
    x̂ = zeros(3,4)
    for i = 1:size(x,3)
        x̂[1:3,1:4] += expw[i].*x[1:3,1:4,i] # calculate mean particle
    end
    toOrthoNormal!(x̂)

    # x̂ = x[:,:,findmax(w)[2]]
    x̂ = [x̂; 0 0 0 1]

    # Calculate position uncertainty----------------------------------------
    se      = zeros(3)
    for i = 1:size(x,3)
        se += expw[i].*(x[1:3,4,i]-x̂[1:3,4]).^2
    end
    se              = √(se)

    # Update filter states--------------------------------------------------
    T_RB_TF_last[:,:] = T_RB_TF
    meas_last[:]      = meas
    s.t              += 1
    savetrace && return x̂, se, s, PFtrace(Neff,resample, deepcopy(x), deepcopy(expw), reset, j, runMeasUpdate)
    return x̂, se, s
end



function normalizeWeights!(w)
    w -= logsumexp(w)
    expw = exp(w)
    return w, expw
end

function initParticles(x_init,Npart,σW)
    x = repeat(x_init,inner=[1,1],outer=[1,1,Npart])
    x[1:3,4,:] = squeeze(x[1:3,4,:],2) + 5*σW[1]*randn(3,Npart)
    I = eye(3)
    for i = 1:Npart
        k = randn(3)
        k /= norm(k) # Axis of rotation
        theta = 5*σW[2]*randn()
        kx = skew(k)
        R = I + sin(theta)*kx + (1-cos(theta))*(k*k'-I)
        x[1:3,1:3,i] = R*x[1:3,1:3,i]
    end
    expw = 1/Npart*ones(Npart)
    w = log(1/Npart)*ones(Npart)
    return x,w,expw
end

function orthoNormalize(x,j)
    x[:,:,1] = toOrthoNormal(x[:,:,1])
    for i = 2:length(j)
        if j[i-1] == j[i]
            continue
        end
        x[:,:,j[i]] = toOrthoNormal(x[:,:,j[i]])
    end
    return x
end
end
