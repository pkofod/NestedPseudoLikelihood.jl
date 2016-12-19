module NestedPseudoLikelihood
using StructuralBase, MDPTools, StatsFuns, Optim
import StructuralBase: inner_iterations, fit
export fit_npl, fit

include("types.jl")
include("logit.jl")
include("initial_ccps.jl")


# The StatsBase conformant way of writing it
fit(E::NPL, U, X, D) = fit_npl(U, X, D, E)

# In case someone passes a single state
fit_npl(U, X::MDPTools.DiscreteState, D; kwargs...) = fit_npl(U, States(X), D; kwargs...)

# The
function fit_npl(U, X, D; method = Optim.Newton())
    E = NPL(U, fit_np!(U.P, D.a, D.x))
    fit_npl(U, X, D, E; method = method)
end
function fit_npl{Ts<:MDPTools.AbstractState}(U::LinearUtility,
                                            X::Ts,
                                            D::Data,
                                            E::NPL;
                                            ## Keyword arguments
                                            # Convergence parameters
                                            ε_outer = 1e-8, # ununsed
                                            ε_inner = 1e-8, # unused
                                            # Initial estimation
                                            method::Optim.Optimizer = Optim.Newton())
    # Get dimension of choice vector once and for all
    J = length(U.P)

    # Initialize cache variables such that most calculations are in-place
    c = NPLCacheVars(X.nX, length(D.xᵈ), U.nvar, J, X.F)

    # Keep track of convergence info
    conv_info = ConvergenceNPL()

    # Keep a trace of parameters and log-likelihood values
    tr = TraceNPL(E.K, U.nvar)

    # Define short-hand notation for general likelihood expressions at data
        llᵈ(x)          =     ll(x,          D, c, J)
       ∇llᵈ(x, grad)    =    ∇ll(x, grad,    D, c, J)
    ll_∇llᵈ(x, grad)    = ll_∇ll(x, grad,    D, c, J)
      ∇²llᵈ(x, hessian) =   ∇²ll(x, hessian, D, c, J)

    # Set options; could be a positional argument?
    options = Optim.Options(show_trace = E.verbose, extended_trace=E.verbose)

    for k = 1:E.K
        copy!(c.Θᵏ, U.Θ)

        ## Prepare inner nest
        # Solve for WP given current P
        Wᴾ!(c, U, X, J)

        # Update the variables used to construct the choice specific values
        discounted_vars!(c, U, X, J)
        # Update variables used in the conditional likelihood
          observed_vars!(D, c, J)

        # Optimize likelihood and apply policy update
        likelihood = TwiceDifferentiable(llᵈ, ∇llᵈ, ll_∇llᵈ, ∇²llᵈ, c.Θᵏ) # sort of a hack, could be waste
        res = optimize(likelihood, c.Θᵏ, method, options)

        # trace the results from the maximum likelihood step
        trace!(tr, conv_info, res, U, E, k)

        # Perform policy step given ᵏ variables to get Pᵏ⁺¹
        Ψ!(c, U)

        # If we're not estimating NPL-K, assess convergence
        !E.finite && assess_convergence(norm(c.Θᵏ - U.Θ, Inf), ε_outer, E, conv_info, k)
    end

    # Store final gradient and Hessian for users to inspect
    gradient = zeros(U.nvar)
    hessian = zeros(U.nvar, U.nvar)
    _ll = ll_∇llᵈ(c.Θᵏ, gradient)
            ∇²llᵈ(c.Θᵏ, hessian)

    # Return results
    objective = TwiceDifferentiable(llᵈ, ∇llᵈ, ll_∇llᵈ, ∇²llᵈ, c.Θᵏ)
    EstimationResults(E,
                      -_ll*D.nobs,
                      -gradient*D.nobs,
                      -hessian*D.nobs,
                      objective,
                      copy(U.Θ),
                      conv_info,
                      tr,
                      D.nobs,
                      objective)
end

"""
    Wᴾ!(c, U, X, J)
    Wᴾ!(  Fᵁ,   Wᴾ,   β,   F,   eᴾ,   Pᵏ,   Z, J)
Updates the matrix Wᴾ that contains the variables used to calculate the valuation
given a current Pᵏ and parameters Θᵏ, Wᴾz*Θᵏ+Wᵖe.
"""
Wᴾ!(c, U, X, J) =
         Wᴾ!(c.Fᵁ, c.Wᴾ, U.β, X.F, c.eᴾ, U.P, U.Z, J)
function Wᴾ!(  Fᵁ,   Wᴾ,   β,   F,   eᴾ,   Pᵏ,   Z, J)
    # Calculate E(ϵʲ|j optimal)
    map!(p->-log.(p), eᴾ, Pᵏ) # this should dispatch on some shock type

    # Calculate transition probability matrix given current policy Pᵏ
    Fᵁ .= β*sum(Pᵏ[ia].*F[ia] for ia = 1:J)

    # Update Wᴾ = [Wᵖz Wᵖe]
    Wᴾ .= (I-Fᵁ)\sum(Pᵏ[ia].*[Z[ia] eᴾ[ia]] for ia = 1:J)
end

discounted_vars!(c, U, X, J) =
         discounted_vars!(c.z̃, c.ẽ, c.Δz̃, c.Δẽ, c.Wᴾ, U, X, J)
function discounted_vars!(  z̃,   ẽ,   Δz̃,   Δẽ,   Wᴾ, U, X, J)
    @views for j = J:-1:1
        z̃[j] .= U.Z[j] + U.β * (X.F[j] * Wᴾ[:,1:end-1])
        ẽ[j] .= U.β * X.F[j] * Wᴾ[:,end]
        if j < J
            Δz̃[j] .= z̃[j]-z̃[J]
            Δẽ[j] .= ẽ[j]-ẽ[J]
        end
    end
end

function observed_vars!(D, c, J)
    # Only use variables at the observed states
    for j = 1:J
      c.z̃ᵈ[j] .= c.z̃[j][D.xᵈ,:]
        if j < J
            c.Δz̃ᵈ[j] .= c.Δz̃[j][D.xᵈ,:]
            c.Δẽᵈ[j] .= c.Δẽ[j][D.xᵈ]
        end
    end
end

"""
    Ψ!(c, U)
    Ψ!(Δz̃, Δẽ, m, U, P)

The best response operator that takes ᵏ variables and updates Pᵏ to Pᵏ⁺¹ in-place.
"""
Ψ!(c, U) =
         Ψ!(c.Δz̃, c.Δẽ, c.m, U, U.P)
# Note that we're updating the whole state space, not just xᵈ, as we need Pᵏ⁺¹
# at all x ∈ X to construct Fᵁ that is needed to calculate Wᴾ.
function Ψ!(Δz̃, Δẽ, m, U, P)
    J = length(P)
    for i = 1:J-1
        m[i] .= Δz̃[i]*U.Θ+Δẽ[i]
    end
    denom = 1+sum(exp.(m[i]) for i = 1:J-1)
    for i = 1:J-1
        P[i] .= exp.(m[i])./denom
    end
    P[J] .= 1-sum(P[ia] for ia = 1:J-1)
end

function assess_convergence(norm, ε_outer, E, conv_info, k)
    if norm < ε_outer
        conv_info.flag = true
        conv_info.outer = k
        conv_info.norm_Θ = norm
        E.Θu = E.Θu[1:k, :]
        E.ll = E.ll[1:k]
    end
end

function trace!(tr, conv_info, res, U, E, k)
    # Increment iteration counter
    conv_info.iter_maxlike += Optim.iterations(res)

    # Grab estimates
    copy!(U.Θ, Optim.minimizer(res))

    # Trace loglikelihood
    E.ll[k] = conv_info.ll

    # Trace parameter estimates
    E.Θu[k, :] = U.Θ
    tr.Θ[k,:] = U.Θ
end

end # module
