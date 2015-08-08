import emcee
import matplotlib.pyplot as plt
import numpy as np
import plotutils.autocorr as ac
import plotutils.parameterizations as par
import plotutils.plotutils as pu
import posterior as pos
import scipy.signal as ss
import triangle

def guess_planet_params(P0, K, T):
    K = K*(1.0 + 1e-3*np.random.randn()) # part in 10^3
    P = P0*(1.0 + 1e-3*np.random.randn()) # part in 10^3
    e = np.random.uniform(low=0, high=0.1)
    omega = np.random.uniform(low=0, high=2*np.pi)
    chi = np.random.uniform(low=0, high=1)

    return np.array([np.log(K),
                     par.bounded_params(P, low=0, high=T),
                     par.bounded_params(e, low=0, high=1),
                     par.bounded_params(omega, low=0, high=2*np.pi),
                     par.bounded_params(chi, low=0, high=1)])

def remove_period_from_params(ps):
    return np.column_stack((ps[:,:-4], ps[:,-3:]))

def add_period_to_params(ps, Ps):
    return np.column_stack((ps[:,:-3], Ps, ps[:,-3:]))

def guess_params():
    return np.rand.randn(7)*1e-4

def get_residual(data, p0=None, Pfixed=None, N=1000, reset=True, sampler=None):
    assert (sampler is not None) or (p0 is not None), 'must provide p0 or sampler'
    
    T = data['jdb'][-1] - data['jdb'][0]

    if sampler is None:
        preds = np.column_stack((data['fwhm'], data['bis_span'], data['rhk']))
        npars = p0.shape[1]
        if Pfixed is None:
            nfixed = 0
        else:
            nfixed = len(Pfixed)
        nkep = (npars - 7 + nfixed)/5 - nfixed
        
        logpost = pos.Posterior(data['jdb'], data['vrad'], data['svrad'], preds, nkep, Pfixed)
        sampler = emcee.EnsembleSampler(128, npars, logpost)
        result = [p0]
    else:
        result = [sampler.chain[:,-1,:]]
        
    result = sampler.run_mcmc(result[0], N, thin=10)

    if reset:
        good_sel = sampler.lnprobability.flatten() > np.max(sampler.lnprobability) - npars/2.0
        if np.count_nonzero(good_sel) < 128:
            pass
        else:
            result = [np.random.permutation(sampler.flatchain[good_sel, :])[:128,:]]
        sampler.reset()

    result = sampler.run_mcmc(result[0], N, thin=10)

    logpost = sampler.lnprobfn.f
    
    resid = 0.0
    resid2 = 0.0
    for p in sampler.chain[:,-1,:]:
        pred = logpost.predict(p)[0]
        res = logpost.y - pred
        
        resid += res
    resid /= sampler.chain.shape[0]

    return sampler, resid

def convergence_plots(sampler):
    plt.figure()
    plt.plot(sampler.lnprobability.T)
    plt.figure()
    pu.plot_emcee_chains(sampler.chain)
    triangle.corner(sampler.flatchain)

    print 'Autocorrelation lengths: ', ac.emcee_chain_autocorrelation_lengths(sampler.chain)
    print 'Gelman-Rubin R: ', ac.emcee_gelman_rubin_r(sampler.chain)
    
def periodogram(sampler, resid, pmin=3):
    logpost = sampler.lnprobfn.f

    if logpost.nkep == 0 and logpost.Pfixed is None:
        plt.figure()
        plt.errorbar(logpost.t, resid, logpost.dy, fmt='.')
    else:
        pers = np.array([logpost.get_periods(p) for p in sampler.flatchain])
        pers = np.mean(pers, axis=0)

        for p in pers:
            plt.figure()
            plt.title('P = {}'.format(p))
            plt.errorbar(logpost.t % p, resid, logpost.dy, fmt='.')

    T = logpost.t[-1] - logpost.t[0]
    dt = np.min(np.diff(logpost.t))

    df = 1.0/T
    fmax = 1.0/(2*dt)

    fs = np.arange(df, fmax, df)
    rpsd = ss.lombscargle(logpost.t, resid, 2*np.pi*fs)
    psd = ss.lombscargle(logpost.t, logpost.y, 2*np.pi*fs)
    N = len(logpost.t)
    
    plt.figure()
    plt.title('Raw')
    plt.plot(1.0/fs, np.sqrt(4*psd/N)*1000, '-k')
    plt.xscale('log')
    plt.xlabel(r'$P$ (day)')
    plt.ylabel(r'$A$ ($\mathrm{m}/\mathrm{s}$)')

    plt.figure()
    plt.title('Cleaned')
    plt.plot(1.0/fs, np.sqrt(4*rpsd/N)*1000, '-b')
    plt.xscale('log')
    plt.xlabel(r'$P$ (day)')
    plt.ylabel(r'$A$ ($\mathrm{m}/\mathrm{s}$)')
    
    pers = 1.0/fs
    sel = pers > pmin
    ibest = np.argmax(rpsd[sel])
    pbest = pers[sel][ibest]
    
    print 'Maximum amplitude period is ', pbest, ' at ', 1000*np.sqrt(rpsd[sel][ibest]*4/len(resid))

    return fs, rpsd
