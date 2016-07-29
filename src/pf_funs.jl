using Debug
import Plots

@inline function mul3!(T,a,b)
    for j = 1:3, i = 1:3
        T[i,j] = a[i,1]*b[1,j] + a[i,2]*b[2,j] + a[i,3]*b[3,j]
    end
    return T
end
@inline function mul4!(T,a,b)
    for j = 1:4, i = 1:3
        T[i,j] = a[i,1]*b[1,j] + a[i,2]*b[2,j] + a[i,3]*b[3,j] + a[i,4]*b[4,j]
    end
    return T
end

@inline function mul3(a,b)
    T = Matrix{Float64}(3,3)
    for j = 1:3, i = 1:3
        T[i,j] = a[i,1]*b[1,j] + a[i,2]*b[2,j] + a[i,3]*b[3,j]
    end
    return T
end
@inline function mul4(a,b)
    T = Matrix{Float64}(4,4)
    for j = 1:4, i = 1:3
        T[i,j] = a[i,1]*b[1,j] + a[i,2]*b[2,j] + a[i,3]*b[3,j] + a[i,4]*b[4,j]
    end
    T[4,1] = T[4,2] = T[4,3] = 0
    T[4,4] = 1
    return T
end

# """
# Calculates the weight update for a bunch of particles
# x_mu: difference between expected seam location and actual seam location,
# SigmaV: measurement noise covariance matrix
# """

function measUpdate(x_μ, theta, x,traj_nom, σV, index, γ, T_TF_M)
    N = size(x_μ,1)
    xtheta = zeros(N)
    for i = 1:N
        if index[i] == size(traj_nom,3)
            xtheta[i] = 0
        else
            frameP = γ*traj_nom[:,:,index[i]+1] + (1-γ)*traj_nom[:,:,index[i]]
            xtheta[i] = xyθ(frameP*T_TF_M,x[:,:,i]*T_TF_M)
        end
    end
    σ = 2σV.^2
    theta_mu2 = (theta-xtheta).^2
    r = -(x_μ[:,1].^2)./σ[1] - (x_μ[:,2].^2)./σ[2] - theta_mu2./σ[3]

    return r
end


# """
# Calculates the distance between where particle `i` thinks the seam should
# be and where it actually is, accoring to sensor.
# `m`: measurement in ℜ², (x,y)-coordinates from sensor
# `traj_nom`: nominal trajectory, in ℜ4x4xT
# `xp`: particles, in ℜ4x4xN
# `T_TCP_M`: tool to sensor transformation matrix, in ℜ4x4
# """


function gamma(i1::Int, i2, n, m̂i, search_traj)
    p₂ = search_traj[i2,:][:]
    p₁ = search_traj[i1,:][:]
    v = p₂-p₁
    γ = (n⋅(m̂i-p₁))/(n[1]v[1]+n[2]v[2]+n[3]v[3])
    p = p₁ + γ*v
    γ, v, p, i1
end

macro OKγ(γ)
    return :(γ <= 1 && γ >= 0)
end

@debug function distToSeam(m, search_traj, xp, T_TF_M; tries = 2)

    Npart = size(xp,3)
    Ntn = size(search_traj,1)

    # m̂ is where the seam should be according to the hypothesis
    m̂ = zeros(Npart,3)
    xpM = similar(xp)
    for n = 1:Npart
        xpM[:,:,n] = xp[:,:,n]*T_TF_M
        m̂[n,:] = (xpM[:,:,n]*[m[1], m[2], 0, 1])[1:3]'# The z-measurement is effectively 0 since we know the point is in the laser plane

    end

    # TODO: change distance from euclidian 2-norm to Mahalanobis distance
    ni, nd = knnsearch(m̂,xpM,search_traj)
    # @bp
    md = Array(Float64,Npart,3)
    # assert(all(all(isfinite(ni))))
    # assert(all(all(isfinite(nd))))
    index = zeros(Int,size(nd))
    # @bp
    for i = 1:Npart
        n = xpM[1:3,3,i] # Normal is the z-axis
        m̂i = (m̂[i,:]')[:]
        if ni[i] == Ntn # End of seam case
            γ, v, p, i1 = gamma(Ntn-1, Ntn, n, m̂i, search_traj)
            if γ > 1 # No measurement aquired
                nd[i] = 0
                md[i,:] = 0
                index[i] = Ntn
                continue
            end
        else
            γ, v, p, i1 = gamma(ni[i], min(ni[i]+1,Ntn), n, m̂i, search_traj)
        end
        if !@OKγ(γ) && ni[i] != 1# Go forward
            for j = 1:tries-1 # Forward had one try for free
                γ, v, p, i1 = gamma(min(ni[i]+j,Ntn), min(ni[i]+j+1,Ntn), n, m̂i, search_traj)
                if @OKγ(γ)
                    break
                end
            end
        end
        if !@OKγ(γ) && ni[i] != 1# # Go Backwards
            for j = 1:tries
                γ, v, p, i1 = gamma(max(ni[i]-j,1), max(ni[i]-j+1,1), n, m̂i, search_traj)
                if @OKγ(γ)
                    break
                end
            end
        end

        if (!@OKγ(γ) && ni[i] != 1) #|| (n⋅v)/norm(v) < 0.1 # Failed to find a measurement OR Angle is too small
            # @show (n⋅v)/norm(v)
            # @show γ
            # @bp norm(v) != 0
            nd[i] = 0
            md[i,:] = 0
            index[i] = i1
            continue
        end

        # Proceed to calculate measurement
        e = p-m̂i
        e = (xpM[1:3,1:3,i])'e
        nd[i] = e[1]^2 + e[2]^2
        md[i,:] = e
        index[i] = i1
        if abs(e[3]) > 1e-6 # Make sure that the error lies in the laser plane
            @show e[3]
            @bp
        end
    end

    x_μ2 = nd
    # @show sum(x_μ2 .== 0)
    return x_μ2, md, index, γ
end


# """
# `knnsearch(m̂,search_traj, range=0, center = 0)`
# returns the index and the distance²
# """

function knnsearch(m̂,xpM,search_traj)

    Npart,M = size(m̂)
    L = size(search_traj,1)
    idx = zeros(Int64,Npart)
    D = zeros(Npart)
    d² = Array(Float64,L)
    # Loop for each query point
    for n=1:Npart
        @inbounds @fastmath for l = 1:L # Transform back to the sensor frame so we are sure only to compare distance in the xy-plane, this hard coded loop is the most essential place for optimization in the entire filter, it reduces runtime to 1/7 over generic matrix multiplication
            a1 = search_traj[l,1] - m̂[n,1]
            a2 = search_traj[l,2] - m̂[n,2]
            a3 = search_traj[l,3] - m̂[n,3]
            b1 = a1*xpM[1,1,n] + a2*xpM[2,1,n] + a3*xpM[3,1,n]
            b2 = a1*xpM[1,2,n] + a2*xpM[2,2,n] + a3*xpM[3,2,n]
            b3 = a1*xpM[1,3,n] + a2*xpM[2,3,n] + a3*xpM[3,3,n]
            d²[l] = b1*b1 + b2*b2 + b3*b3
        end
        D[n],idx[n] = findmin(d²) # Find the smallest distance
    end
    return idx,D
end




function measFKUpdate(x,T_RB_TCP,f,sigmaVFK)
    N = size(x,3)
    d = sigmaVFK[3] + sigmaVFK[4]*norm(f)
    dR = sigmaVFK[5]
    x_mu2 = zeros(N)
    theta = zeros(N)
    # TODO: update this according to paper
    for i = 1:N
        x_mu2[i] = sum((T_RB_TCP[1:3,4] - x[1:3,4,i]).^2)  # this has been optimized to remove norm calculation which was squared in the next step
        costheta = (trace(T_RB_TCP[1:3,1:3]' * x[1:3,1:3,i])-1)/2
        if costheta > 1 # will never be < -1 because theta is small
            costheta = 1;
        end
        theta[i] = acos(costheta)
    end
    w = zeros(theta)
    outsidePlateauR = theta .> dR^2
    w[outsidePlateauR] -= ((theta[outsidePlateauR]-dR).^2)./(2sigmaVFK[2]^2) # Angle update
    outsidePlateau = x_mu2 .> d^2
    w[outsidePlateau] -= ((sqrt(x_mu2[outsidePlateau])-d).^2)./(2sigmaVFK[1]^2); # xyz-update
    return w

end


# """
# This function propagates all particles forward in time.
# xp: particles to be propagated
# increment: 4x4 transformation matrix describing the move done by the
# robot during the last time step.
# T\_RB\_TF(t-1)\\T\_RB\_TF(t) = T\_TF(t-1)\_TF(t)
# """

function timeUpdate!(xp, increment, sigmaW,sigmaWr, h)
    N = size(xp,3)
    I = eye(3)
    noise = randn(3,N).*(√(h)*sigmaW[1:3])
    R = eye(3)
    for n = 1:N
        xp[:,:,n] = increment*xp[:,:,n]
        xp[1:3,4,n] += noise[:,n] # Add state noise

        #         w = 0.01*pi/180*randn(3,1);
        #         W = [0, -w(3), w(2); w(3), 0, -w(1); -w(2), w(1), 0];
        #         R = expm(W); # this takes a lot of time!
        k = randn(3)
        k /= norm(k) # Axis of rotation
        theta = √(h)*sigmaWr*randn()
        kx = skew!(R,k)
        R = I + sin(theta)*kx + (1-cos(theta))*(k*k'-I)
        @assert !any(isnan(R[:]))
        xp[1:3,1:3,n] = R*xp[1:3,1:3,n]
    end

    return xp
end

@inline function skew!(R,s)
    R[1,1] = 0
    R[2,1] = s[3]
    R[3,1] = -s[2]
    R[1,2] = -s[3]
    R[2,2] = 0
    R[3,2] = s[1]
    R[1,3] = s[2]
    R[2,3] = -s[1]
    R[3,3] = 0
    return R
end


function resample_systematic_exp(w,M)
    N = size(w,1)
    bins = cumsum(w) # devec
    s = (rand()/M+0):1/M:bins[end]
    j = Array(Int64,M)
    bo = 1
    for i = 1:M
        for b = bo:N
            if s[i] <= bins[b]
                j[i] = b
                bo = b
                break
            end
        end
    end
    return j
end
