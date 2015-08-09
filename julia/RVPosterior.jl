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
    eccs::Array{Float64, 1}
    omegas::Array{Float64, 1}
    chis::Array{Float64, 1}
end

nplanets(p::PhysicalParams) = size(p.pers,1)
npred(p::PhysicalParams) = size(p.betas,1)

function rvs(p::PhysicalParams, ts::Array{Float64, 1})
    rvs = Kepler.rv_model(ts, p.vels, p.eccs, p.omegas, p.chis, p.pers)
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
    nbet = size(v, 1) - 4 - nplp

    mu = v[1]

    sigma = v[2]

    tau = v[3]

    nu = v[4]

    betas = v[5:5+nbet-1]

    if post.npl > 0
        pers = v[5+nbet:5:end]
        vels = v[5+nbet+1:5:end]
        eccs = v[5+nbet+2:5:end]
        omegas = v[5+nbet+3:5:end]
        chis = v[5+nbet+4:5:end]
    else
        pers = zeros(0)
        vels = zeros(0)
        eccs = zeros(0)
        omegas = zeros(0)
        chis = zeros(0)
    end

    PhysicalParams(mu, sigma, tau, nu, betas, pers, vels, eccs, omegas, chis)
end

function tov(post, p)
    nplp = post.npl*5
    nbet = size(p.betas, 1)

    n = 4 + nplp + nbet

    v = zeros(n)

    v[1] = p.mu
    v[2] = p.sigma
    v[3] = p.tau
    v[4] = p.nu

    v[5:5+nbet-1] = p.betas

    if post.npl > 0
        v[5+nbet:5:end] = p.pers
        v[5+nbet+1:5:end] = p.vels
        v[5+nbet+2:5:end] = p.eccs
        v[5+nbet+3:5:end] = p.omegas
        v[5+nbet+4:5:end] = p.chis
    end

    v
end

function logprior(post, p, v)
    if p.sigma < 0.0
        return -Inf
    end
    if p.tau < 0.0
        return -Inf
    end
    if p.nu < 0.0
        return -Inf
    end

    if post.npl > 0
        for i in 1:post.npl
            if p.pers[i] < 0.0
                return -Inf
            end
            if p.vels[i] < 0.0
                return -Inf
            end
            if p.eccs[i] < 0.0 || p.eccs[i] > 1.0
                return -Inf
            end
            if p.omegas[i] < 0.0 || p.omegas[i] > 2.0*pi
                return -Inf
            end
            if p.chis[i] < 0.0 || p.chis[i] > 1.0
                return -Inf
            end
        end
    end

    0.0
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
        p.pers[end] = P0 + 1e-5*randn()
        p.vels[end] = K0 + 1e-5*randn()
        p.eccs[end] = 0.1*rand()
        p.omegas[end] = 2.0*pi*rand()
        p.chis[end] = rand()

        new_ps[:,i] = tov(new_post, p)
    end

    lnps = Float64[lnprob(new_post, new_ps[:,i]) for i in 1:nwalk]

    pbest = copy(new_ps[:,indmax(lnps)])
    for i in 1:nwalk
        new_ps[:,i] = pbest + 1e-6*randn(size(pbest,1))
    end
    lnps = Float64[lnprob(new_post, new_ps[:,i]) for i in 1:nwalk]

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
