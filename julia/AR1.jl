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

""" Returns the uncorrelated residuals and associated uncertainties
for the AR(1) model applied to the given data. """
function residuals(model::AR1Model, ts::Array{Float64, 1}, ys::Array{Float64, 1}, dys::Array{Float64, 1})
    n = size(ts, 1)

    s2 = model.sigma*model.sigma
    nu2 = model.nu*model.nu

    rs = zeros(n)
    ss = zeros(n)

    t = ts[1]
    x = 0.0
    dx2 = s2
    k = 1.0
    
    for i in 1:n
        # Advance the filter
        dt = ts[i] - t
        t = ts[i]
        lam = exp(-dt/model.tau)
        x = lam*x
        dx2 = lam*lam*(dx2 - s2) + s2

        # Predict the observation
        rs[i] = ys[i] - x - model.mu
        ss2 = dx2 + nu2*dys[i]*dys[i]
        ss[i] = sqrt(ss2)

        # Update gain
        k = dx2 / ss2

        # Update state
        x += rs[i]*k
        dx2 -= ss2*k*k
    end

    rs, ss
end

""" Returns a sample from the given AR(1) model at times `ts` with
observational uncertainties `dxs` """
function simulate(model::AR1Model, ts::Array{Float64, 1}, dxs::Array{Float64, 1})
    xs = zeros(ts)
    xsout = zeros(ts)

    xs[1] = model.sigma*randn()
    xsout[1] = xs[1] + model.nu*dxs[1]*randn() + model.mu
    
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
