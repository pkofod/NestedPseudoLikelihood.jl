type NPLCacheVars{T}
    m::Vector{Vector{T}}   # index
    eᴾ::Vector{Vector{T}}  # expected value of shock given optimality
    P::Vector{Vector{T}}   # for use in policy step
    Pᵈ::Vector{Vector{T}}  # for use in ML step
    z̃::Vector{Matrix{T}}   # ML
    z̃ᵈ::Vector{Matrix{T}}  # used in gradient!
    Δz̃::Vector{Matrix{T}}  # policy
    Δz̃ᵈ::Vector{Matrix{T}} # ML
    ẽ::Vector{Vector{T}}   # unsure if needed
    Δẽ::Vector{Vector{T}}  # policy
    Δẽᵈ::Vector{Vector{T}} # ML
    Fᵁ::AbstractMatrix{T}  # Controlled transition probability matrix
    Wᴾ::AbstractMatrix{T}  # Valuated Zs and eᵖs
    Θᵏ::Vector{T}          # Parameter cache
end
function NPLCacheVars(nX, nXᵈ, nvar, J, F)
    m   = [zeros(nX) for j = 1:J-1] # index
    eᴾ  = [zeros(nX) for j = 1:J]   # expected value of shock given optimality
    P   = [zeros(nX) for j = 1:J]   # for use in policy step
    Pᵈ  = [zeros(nXᵈ)  for j = 1:J] # for use in ML step
    z̃   = [zeros(nX,  nvar)  for j = 1:J]   # ML
    z̃ᵈ  = [zeros(nXᵈ, nvar) for j = 1:J] # used in gradient!
    Δz̃  = [zeros(nX,  nvar) for j = 1:J-1]  # policy
    Δz̃ᵈ = [zeros(nXᵈ, nvar) for j = 1:J-1] # ML
    ẽ   = [zeros(nX) for i in 1:J] # unsure if needed
    Δẽ  = [zeros(nX) for j = 1:J-1]  # policy
    Δẽᵈ = [zeros(nXᵈ) for j = 1:J-1] # ML
    Fᵁ  = zeros(F[1])         # Controlled transition probability matrix
    Wᴾ  = zeros(nX, nvar+1) # Valuated Zs and eᵖs
    Θᵏ  = zeros(nvar)
    NPLCacheVars(m, eᴾ, P, Pᵈ,
                 z̃, z̃ᵈ, Δz̃, Δz̃ᵈ,
                 ẽ, Δẽ, Δẽᵈ,
                 Fᵁ, Wᴾ, Θᵏ)
end

type ConvergenceNPL <: ConvergenceInfo
    maxK::Int64
    # Outer loop
    flag::Bool
    outer::Int64
    norm_Θ::Float64
    # Inner loop
    ll::Float64
    norm_grad::Float64
    iter_maxlike::Int64
end
ConvergenceNPL() = ConvergenceNPL(0, false, 0, 0., 0., 0., 0)

inner_iterations(conv::ConvergenceNPL) = conv.iter_maxlike

type NPL <: EstimationMethod
  U::LinearUtility
  Θu::Matrix{Float64}
  Θy::Vector{Float64}
  Θf::Vector{Float64}
  ll::Vector{Float64}
  first_step::ConvergenceNPL
  K::Integer
  finite::Bool
  verbose::Bool
end

function NPL(U, initial; maxK::Integer = 0, verbose::Bool = false)
    K = maxK == 0 ? 80 : maxK
    NPL(U,
        #Θu
        zeros(K, size(U.Z[1],2)),
        #Θy
        Vector{Float64}(),
        #Θf
        Vector{Float64}(),
        # ll
        zeros(K),
        # first step results (should just be a P)
        initial,
        # If maxK is set, it will run exactly maxK times.
        K,
        # Is it NPL-K? (as opposed to NPL-∞)
        K > 0,
        # Print information
        verbose)
end


function Base.display{M<:NPL}(est_res::EstimationResults{M})
    @printf "Results of estimation\n"
    @printf " * Method: NPL\n"
    @printf " * loglikelihood: %s\n" est_res.loglikelihood
    if length(join(est_res.trace.Θ[end,:], ",")) < 40
       @printf " * Estimate: [%s]\n" join(est_res.trace.Θ[end,:], ",")
       @printf " * Std.err.: [%s]\n" join(stderr(est_res), ",")
   else
       @printf " * Estimate: [%s, ...]\n" join(est_res.trace.Θ[end,1:2], ",")
       @printf " * Std.err.: [%s, ...]\n" join(stderr(est_res)[1:2], ",")
   end
   @printf " Iterations\n"
   n_PI = size(est_res.trace.Θ, 1)
   if n_PI == 1
       @printf " * Policy iteration: %d\n" size(est_res.trace.Θ, 1)
   else
       @printf " * Policy iteration: %d\n" size(est_res.trace.Θ, 1)
   end
   @printf " * Maximum likelihood: %d\n" est_res.conv.iter_maxlike
end


type TraceNPL
    Θ::Matrix{Float64}
    norm_ΔΘ::Vector{Float64}
    ll::Vector{Float64}
    Δll::Vector{Float64}
    norm_g::Vector{Float64}
end

TraceNPL(K, n) = TraceNPL(fill(NaN, K, n),
                          fill(NaN, K),
                          fill(NaN, K),
                          fill(NaN, K),
                          fill(NaN, K))
