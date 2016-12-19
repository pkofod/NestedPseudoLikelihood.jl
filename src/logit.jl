function ll{T<:Real}(Θᵏ::Vector{T}, D, c, J)
    update_Pᵈ_at_xᵈ!(c.Pᵈ, D.xᵈ, Θᵏ, c.Δz̃ᵈ, c.Δẽᵈ, J)

    ll = zero(T)
    lengthxd = length(D.xᵈ)
    @inbounds for j = 1:J
        for i = 1:lengthxd
            ll -=  D.nxjᵈ[i,j]*log(c.Pᵈ[j][i])
        end
    end
    ll/D.nobs
end

∇ll(Θᵏ, grad, D, c, J) =
                  ∇ll(Θᵏ,            grad, D.x,  J, D.nobs, D.xᵈ, c, D)
function ∇ll{T<:Real}(Θᵏ::Vector{T}, grad,   xᵈ, J, n, Dxd, c, D)
    update_Pᵈ_at_xᵈ!(c.Pᵈ, Dxd, Θᵏ, c.Δz̃ᵈ, c.Δẽᵈ, J)

    z̄ = sum(c.Pᵈ[j].*c.z̃ᵈ[j] for j = 1:J)
    grad .= zero(T)
    @views for j in 1:J
        grad[:] -= sum(D.nxjᵈ[:, j].*(c.z̃ᵈ[j]-z̄),1)'
    end
    grad .= grad/n
end

ll_∇ll(Θᵏ, grad, D, c, J) =
                  ll_∇ll(Θᵏ,           grad, D.x,  J, D.nobs, c, D.nxjᵈ, D.xᵈ)
function ll_∇ll{T<:Real}(x::Vector{T}, grad,   xᵈ, J,   n,    c,   nxjᵈ, Dxd)

    K = length(x)
    lengthxd = length(Dxd)
    update_Pᵈ_at_xᵈ!(c.Pᵈ, Dxd, x, c.Δz̃ᵈ, c.Δẽᵈ, J)
    ll = zero(T)
    @inbounds for j = 1:J
        for i = 1:lengthxd
            ll -=  nxjᵈ[i,j]*log(c.Pᵈ[j][i])
        end
    end

    z̄ = sum(c.Pᵈ[j].*c.z̃ᵈ[j] for j = 1:J)
    grad .= zero(T)
    @views for j in 1:J
        grad[:] -= sum(nxjᵈ[:, j].*(c.z̃ᵈ[j]-z̄),1)'
    end
    grad .= grad/n
    ll/n
end

function ∇²ll{T<:Real}(Θᵏ::Vector{T}, hessian, D, c, J)
    update_Pᵈ_at_xᵈ!(c.Pᵈ, D.xᵈ, Θᵏ, c.Δz̃ᵈ, c.Δẽᵈ, J)
    K = size(c.z̃ᵈ[1],2)

    z̄ = sum(c.Pᵈ[j].*c.z̃ᵈ[j] for j = 1:J)
    nxx = sum(D.nxjᵈ,2)
    hessian .= zero(T)
    for j in 1:J
        hessian .+= (nxx.*c.Pᵈ[j].*(c.z̃ᵈ[j]-z̄))'*(c.z̃ᵈ[j]-z̄)
    end

    hessian .= hessian/D.nobs
end

update_Pᵈ_at_xᵈ!{T}(c, D, Θᵏ::Vector{T}, J) =
          update_Pᵈ_at_xᵈ!(c.Pᵈ, D.xᵈ, Θᵏ,          c.Δz̃ᵈ, c.Δẽᵈ, J)
function update_Pᵈ_at_xᵈ!{T}(Pᵈ,   xᵈ, Θᵏ::Vector{T}, Δz̃ᵈ,   Δẽᵈ, J)
    K = size(Δz̃ᵈ[1],2) # nvar
    nxᵈ = length(xᵈ)
    @inbounds for j = 1:J-1
        Pᵈ[j] .= Δẽᵈ[j]
        zj = Δz̃ᵈ[j]
        for k = 1:K
            @simd for i = 1:nxᵈ
                Pᵈ[j][i] += zj[i, k]*Θᵏ[k] # hoist
            end
        end
        Pᵈ[j] .= exp.(Pᵈ[j])
    end
    denom = 1+sum(Pᵈ[1:J-1])
    Pᵈ[end] .= one(T)
    for j = 1:J-1
        Pᵈ[j] .= Pᵈ[j]./denom
        Pᵈ[end] .-= Pᵈ[j]
    end
end
