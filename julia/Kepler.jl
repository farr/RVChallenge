module Kepler

export rv_model

kepler_f(M, E, e) = E - e*sin(E) - M
kepler_fp(E, e) = one(E) - e*cos(E)
kepler_fpp(E,e) = e*sin(E)
kepler_fppp(E,e) = e*cos(E)

function kepler_solve_ea(n, e, t)
    M = mod(n*t, 2.0*pi)

    if M < pi
        E = M + 0.85*e
    else
        E = M - 0.85*e
    end

    f = kepler_f(M, E, e)

    while abs(f) > 1e-8
        fp = kepler_fp(E,e)
        disc = sqrt(abs(16.0*fp*fp - 20.0*f*kepler_fpp(E,e)))
        if fp > 0.0
            d = -5.0*f/(fp + disc)
        else
            d = -5.0*f/(fp - disc)
        end

        E += d
        f = kepler_f(M, E, e)
    end

    E
end

function kepler_solve_ta(n, e, t)
    E = kepler_solve_ea(n, e, t)

    f = 2.0*atan(sqrt((one(e)+e)/(one(e)-e))*tan(E/2.0))

    if f < 0.0
        f += 2.0*pi
    end

    f
end

""" 
`rv_model(ts, K, e, omega, chi, P)`

Returns the radial velocity at times `ts` for the system with
semi-amplitude `K`, eccentricity `e`, argument of pericentre `omega`,
phase at a `t=0` of `2*pi*chi`, and period `P`.  These parameter
arguments can also be vectorised, in which case the returned array has
size `(ntimes, nplanets)`.  
"""
function rv_model(ts::Array{Float64, 1},
                  K::Float64,
                  e::Float64,
                  omega::Float64,
                  chi::Float64,
                  P::Float64)

    nts = size(ts, 1)

    rvs = zeros(nts)

    t0 = -chi*P
    n = 2.0*pi/P

    ecw = e*cos(omega)
        
    for i in 1:nts
        t = ts[i]
        f = kepler_solve_ea(n, e, t-t0)

        rvs[i] = K*(cos(f + omega) + ecw)
    end

    rvs
end

function rv_model(ts::Array{Float64, 1},
                  Ks::Array{Float64, 1},
                  es::Array{Float64, 1},
                  omegas::Array{Float64, 1},
                  chis::Array{Float64, 1},
                  Ps::Array{Float64, 1})
    npl = size(Ks, 1)
    nts = size(ts, 1)

    rvs = zeros(nts, npl)

    for j in 1:npl
        rvs[:,j] = rv_model(ts, Ks[j], es[j], omegas[j], chis[j], Ps[j])
    end

    rvs
end

end
