module AR1

export AR1Model, residuals, simulate

@doc """
Construct an AR(1) model with mean `mu`, standard deviation (at zero
lag) `sigma`, correlation time `tau`, and measurement uncertainty scale
`nu`.
""" ->
immutable AR1Model
    mu::Float64
    sigma::Float64
    tau::Float64
    nu::Float64
end

immutable AR1State
    mu::Float64
    var::Float64
end

function observe(model::AR1Model, state::AR1State, x::Float64, dx::Float64)
    dx2 = dx*dx*model.nu*model.nu

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

""" Returns the uncorrelated residuals and associated uncertainties
for the AR(1) model applied to the given data. """
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

        state = observe(model, state, ys[i], dys[i])
    end

    resids, sresids
end

""" Returns a sample from the given AR(1) model at times `ts` with
observational uncertainties `dxs` """
function simulate(model::AR1Model, ts::Array{Float64, 1}, dxs::Array{Float64, 1})
    xs = zeros(ts)
    xsout = zeros(ts)

    xs[1] = model.sigma*randn()
    xsout[1] = xs[1] + model.nu*dxs[1]*randn()
    
    for i in 2:size(xs,1)
        dt = ts[i] - ts[i-1]
        alpha = exp(-dt/model.tau)

        beta = model.sigma*sqrt(1.0 - alpha*alpha)
        
        xs[i] = alpha*xs[i-1] + beta*randn()
        xsout[i] = xs[i] + model.nu*dxs[i]*randn() + model.mu
    end

    xsout
end

end
