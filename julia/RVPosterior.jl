module RVPosterior

using AR1
using Ensemble
using Kepler

immutable PhysicalParams
    mu::Float64
    sigma::Float64
    tau::Float64
    nu::Float64
    betas::Array{Float64, 1}
    pers::Array{Float64, 1}
    vels::Array{Float64, 1}
    ecws::Array{Float64, 1}
    esws::Array{Float64, 1}
    chis::Array{Float64, 1}
end

nplanets(p::PhysicalParams) = size(p.pers,1)
npred(p::PhysicalParams) = size(p.betas,1)

ecc(p::PhysicalParams) = sqrt(p.ecws.*p.ecws + p.esws.*p.esws)
omega(p::PhysicalParams) = atan2(p.esws, p.ecws)

function rvs(p::PhysicalParams, ts::Array{Float64, 1})
    es = ecc(p)
    omegas = omega(p)

    rvs = Kepler.rv_model(ts, p.vels, es, omegas, p.chis, p.pers)
    sum(rvs, 2)    
end

function cresid(p, ts, vs, preds)
    npl = size(p.pers, 1)
    rs = zeros(ts)

    if npl > 0
        rv = rvs(p, ts)
    else
        rv = zeros(ts)
    end

    for i in eachindex(rs)
        rs[i] = vs[i] - rv[i]
        for j in 1:size(preds,2)
            rs[i] -= p.betas[j]*preds[i,j]
        end
    end

    rs
end

function uncorr_resid(p, ts, vs, dvs, preds)
    crs = cresid(p, ts, vs, preds)

    AR1.residuals(AR1.AR1Model(p.mu, p.sigma, p.tau, p.nu), ts, crs, dvs)
end

function loglike(p, ts, vs, dvs, preds)
    rs, srs = uncorr_resid(p, ts, vs, dvs, preds)
    
    lp = 0.0
    for i in 1:size(ts, 1)
        lp += -0.91893853320467274180 - log(srs[i]) - 0.5*rs[i]*rs[i]/(srs[i]*srs[i])
    end
    lp
end

immutable Posterior
    ts::Array{Float64, 1}
    vs::Array{Float64, 1}
    dvs::Array{Float64, 1}
    preds::Array{Float64, 2}
    npl::Int
end

function toparams(post, v)
    nplp = post.npl*5
    nbet = size(v,1) - 4 - nplp

    mu = v[1]
    sigma = exp(v[2])
    tau = exp(v[3])
    nu = exp(v[4])

    betas = v[5:5+nbet-1]

    i = 5+nbet
    Ps = zeros(post.npl)
    Ks = zeros(post.npl)
    ecws = zeros(post.npl)
    esws = zeros(post.npl)
    chis = zeros(post.npl)
    for j in 1:post.npl
        Ps[j] = exp(v[i])
        i += 1

        Ks[j] = exp(v[i])
        i += 1

        x = v[i:i+1]
        z = Ensemble.Parameterizations.unit_disk_value(x)
        ecws[j] = z[1]
        esws[j] = z[2]
        i += 2

        chis[j] = Ensemble.Parameterizations.bounded_value(v[i], 0.0, 1.0)
        i += 1
    end

    PhysicalParams(mu, sigma, tau, nu, betas, Ps, Ks, ecws, esws, chis)
end


function tov(post, p)
    nplp = post.npl*5
    nbet = size(p.betas, 1)

    n = 4 + nplp + nbet

    v = zeros(n)
    i = 1

    v[i] = p.mu
    i += 1
    
    v[i] = log(p.sigma)
    i += 1

    v[i] = log(p.tau)
    i += 1

    v[i] = log(p.nu)
    i += 1

    v[i:i+nbet-1] = p.betas
    i += nbet

    for j in 1:post.npl
        v[i] = log(p.pers[j])
        i += 1

        v[i] = log(p.vels[j])
        i += 1

        z = Float64[p.ecws[j], p.esws[j]]
        x = Ensemble.Parameterizations.unit_disk_param(z)
        v[i:i+1] = x
        i += 2

        v[i] = Ensemble.Parameterizations.bounded_param(p.chis[j], 0.0, 1.0)
        i += 1
    end

    v
end

function logprior(post, p, v)
    nplp = post.npl*5
    nbet = size(p.betas, 1)

    T = post.ts[end] - post.ts[1]
    vmin = minimum(post.vs)
    vmax = maximum(post.vs)

    V = vmax - vmin

    lp = 0.0

    # Jumping in log(sigma), log(tau), log(nu), but want flat priors
    # p(logsigma) = sigma*p(sigma)
    lp += log(p.sigma)
    lp += log(p.tau)
    lp += log(p.nu)

    if post.npl > 0
        for i in 1:post.npl
            # Similarly, want flat priors on P and velocity
            lp += log(p.pers[i])
            lp += log(p.vels[i])

            z = Float64[p.ecws[i], p.esws[i]]
            j = 5 + nbet + 2 + (i-1)*5
            ve = v[j:j+1]
            lp += Ensemble.Parameterizations.unit_disk_logjac(z, ve)

            chiv = v[5 + nbet + 4 + (i-1)*5]
            lp += Ensemble.Parameterizations.bounded_logjac(p.chis[i], chiv, 0.0, 1.0)
        end
    end

    lp
end

function lnprob(post, v)
    p = toparams(post, v)

    lp = logprior(post, p, v)

    if lp == -Inf
        return lp
    else
        return loglike(p, post.ts, post.vs, post.dvs, post.preds) + lp
    end
end

function addplanet(post, ps::Array{Float64, 2}, P0, K0)
    new_post = Posterior(post.ts, post.vs, post.dvs, post.preds, post.npl+1)

    nwalk = size(ps, 2)

    new_ps = zeros(size(ps,1)+5, nwalk)
    for i in 1:nwalk
        p = toparams(new_post, cat(1, ps[:,i], zeros(5)))
        p.pers[end] = P0*(1.0 + 1e-5*randn())
        p.vels[end] = K0*(1.0 + 1e-5*randn())
        p.ecws[end] = 0.1*rand()
        p.esws[end] = 0.1*rand()
        p.chis[end] = rand()

        new_ps[:,i] = tov(new_post, p)
    end

    lnps = Float64[lnprob(new_post, new_ps[:,i]) for i in 1:nwalk]

    ibest = indmax(lnps)
    pbest = new_ps[:,ibest]

    for i in 1:nwalk
        new_ps[:,i] = pbest + 1e-5*randn(size(pbest,1))
    end

    new_post, new_ps, lnps
end

function save_samples(io, ps)
    writedlm(io, reshape(permutedims(ps, [1, 3, 2]), (size(ps,1), size(ps,2)*size(ps,3))))
end

function load_samples(io, nwalk)
    ps = readdlm(io)

    ps = reshape(ps, size(ps,1), div(size(ps,2),nwalk), nwalk)

    permutedims(ps, [1, 3, 2])
end

end
