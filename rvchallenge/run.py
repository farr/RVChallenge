import emcee
import matplotlib.pyplot as plt
import numpy as np
import plotutils.autocorr as ac
import plotutils.plotutils as pu
import regres_ar1 as rar
import scipy.signal as ss

def get_residual(data, extra_preds=None, p0=None):
    preds = np.column_stack((data['fwhm'], data['bis_span'], data['rhk']))
    npars = 7
    if extra_preds is not None:
        preds = np.column_stack((preds, extra_preds))
        npars += extra_preds.shape[1]

    logpost = rar.AR1RegresPosterior(data['jdb'], data['vrad'], data['svrad'], preds)
    sampler = emcee.EnsembleSampler(128, npars, logpost)
    if p0 is None:
        result = [1e-4*np.random.randn(128, npars)]
    else:
        result = [p0]
    result = sampler.run_mcmc(result[0], 1000, thin=10)

    good_sel = sampler.lnprobability.flatten() > np.max(sampler.lnprobability) - 3.5
    if np.count_nonzero(good_sel) < 128:
        pass
    else:
        result = [np.random.permutation(sampler.flatchain[good_sel, :])[:128,:]]
    sampler.reset()

    result = sampler.run_mcmc(result[0], 1000, thin=10)

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

    print 'Autocorrelation lengths: ', ac.emcee_chain_autocorrelation_lengths(sampler.chain)
    print 'Gelman-Rubin R: ', ac.emcee_gelman_rubin_r(sampler.chain)
    
def periodogram(sampler, resid, pmin=3):
    logpost = sampler.lnprobfn.f

    plt.figure()
    plt.errorbar(logpost.t, resid, logpost.dy, fmt='.')

    T = logpost.t[-1] - logpost.t[0]
    dt = np.min(np.diff(logpost.t))

    df = 1.0/T
    fmax = 1.0/(2*dt)

    fs = np.arange(df, fmax, df)
    rpsd = ss.lombscargle(logpost.t, resid, 2*np.pi*fs)
    psd = ss.lombscargle(logpost.t, logpost.y, 2*np.pi*fs)
    N = len(logpost.t)
    
    plt.figure()
    plt.plot(1.0/fs, np.sqrt(4*psd/N)*1000, '-k', label='Raw')
    plt.plot(1.0/fs, np.sqrt(4*rpsd/N)*1000, '-b', label='Cleaned')
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.xlabel(r'$P$ (day)')
    plt.ylabel(r'$A$ ($\mathrm{m}/\mathrm{s}$)')

    pers = 1.0/fs
    sel = pers > pmin
    ibest = np.argmax(rpsd[sel])
    pbest = pers[sel][ibest]
    
    print 'Maximum amplitude period is ', pbest, ' at ', 1000*np.sqrt(rpsd[sel][ibest]*4/len(resid))

    return fs, rpsd
