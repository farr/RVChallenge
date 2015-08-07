module AR1

export AR1Model, AR1State, residuals

immutable AR1Model
    mu::Float64
    sigma::Float64
    tau::Float64
end

immutable AR1State
    mu::Float64
    var::Float64
end

function observe(state::AR1State, x::Float64, dx::Float64)
    dx2 = dx*dx

    mu_new = (state.mu*dx2 + x*state.var)/(dx2 + state.var)
    var_new = 1.0/(1.0/dx2 + 1.0/state.var)

    AR1State(mu_new, var_new)
end

function advance(model::AR1Model, state::AR1State, dt::Float64)
    y = state.mu - model.mu

    alpha = exp(-dt/model.tau)
    
    y_new = alpha*y

    mu_new = y_new + model.mu

    a2 = alpha*alpha
    
    var_new = a2*state.var + (1.0 - a2)*model.sigma*model.sigma

    AR1State(mu_new, var_new)
end

function residuals(model::AR1Model, ts::Array{Float64, 1}, ys::Array{Float64, 1}, dys::Array{Float64, 1})
    n = size(ts, 1)

    state = AR1State(model.mu, model.sigma*model.sigma)
    resids = zeros(n)
    sresids = zeros(n)
    t = ts[1]

    for i in 1:n
        dt = ts[i] - t
        t = ts[i]
        state = advance(model, state, dt)
        
        resids[i] = ys[i] - state.mu
        sresids[i] = sqrt(state.var + dys[i]*dys[i])

        state = observe(state, ys[i], dys[i])
    end

    resids, sresids
end

end
