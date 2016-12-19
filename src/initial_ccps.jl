# This should probably be in a seperate package

function fit_np!(P, a, x)
    map(x->fill!(x, zero(eltype(P[1]))), P)
    nxobs = zeros(length(P[1]))
    @inbounds for i_obs = 1:length(a)
        x_obs = x[i_obs]
        nxobs[x_obs] += 1
        P[a[i_obs]][x_obs] += 1
    end
    for ia = 1:length(P)
        for i = 1:length(P[1])
            if nxobs[i] == 0
                P[ia][i] = 0.5
            elseif P[ia][i] == 0
                P[ia][i] = 1e-6
            elseif P[ia][i] == nxobs[i]
                P[ia][i] = P[ia][i]/nxobs[i] - 1e-6
            else
                P[ia][i] = P[ia][i]/nxobs[i]
            end
        end
    end
    ConvergenceNPL() # should probably save this
end
