import arfit.ar1_kalman as ak
import kepler
import numpy as np
import plotutils.parameterizations as par

class Posterior(object):
    def __init__(self, t, y, dy, pred, nkep, Pfixed=None):
        self.t = t
        self.y = y
        self.dy = dy
        self.pred = pred
        self.npred = pred.shape[1]
        self.nkep = nkep
        self.Pfixed = Pfixed
        if Pfixed is None:
            self.nfixed = 0
        else:
            self.nfixed = len(Pfixed)

    @property
    def T(self):
        return self.t[-1] - self.t[0]

    def ar1_pars(self, p):
        return p[:4]
        
    def coeffs(self, p):
        return p[4:4+self.npred]

    def kep_pars(self, p):
        return p[4+self.npred:4+self.npred+5*self.nkep].reshape((-1, 5))

    def kep_fixed_pars(self, p):
        return p[4+self.npred+5*self.nkep:].reshape((-1, 4))

    def kep_deparameterize(self, kps):
        dkps = np.zeros_like(kps)

        dkps[:,0] = par.bounded_values(kps[:,0], low=0, high=0.1) # bounded(K, 0, 100 m/s) => K
        dkps[:,1] = par.bounded_values(kps[:,1], low=0, high=self.T) # bounded(P, 0, T) => P
        dkps[:,2] = par.bounded_values(kps[:,2], low=0, high=1) # bounded(e, 0, 1) => e
        dkps[:,3] = par.bounded_values(kps[:,3], low=0, high=2*np.pi) # bounded(omega, 0, 2*pi) => omega
        dkps[:,4] = par.bounded_values(kps[:,4], low=0, high=1) # bounded(chi, 0, 1) => chi

        return dkps

    def kep_fixed_deparameterize(self, kps):
        kps_ext = np.zeros((kps.shape[0], 5))

        kps_ext[:,0] = kps[:,0]
        kps_ext[:,1] = par.bounded_params(self.Pfixed, low=0, high=self.T)
        kps_ext[:,2:] = kps[:,1:]

        return self.kep_deparameterize(kps_ext)
    
    def get_ar1(self, p):
        kep_rvs = self.get_rvs(p)
        resid = self.y - np.dot(self.pred, self.coeffs(p)) - kep_rvs

        s = np.std(self.y)
        
        return ak.AR1KalmanPosterior(self.t, resid, self.dy, sigma_low = 0.1*s, sigma_high=10*s)

    def get_rvs(self, p):
        rvs = 0.0
        
        if self.nkep > 0:
            kep_ps = self.kep_deparameterize(self.kep_pars(p))
            kep_rvs = kepler.rv_model(self.t, kep_ps[:,0], kep_ps[:,2], kep_ps[:,3], kep_ps[:,4], 2.0*np.pi/kep_ps[:,1])
            rvs += np.sum(kep_rvs, axis=0)

        if self.Pfixed is not None:
            kep_ps = self.kep_fixed_deparameterize(self.kep_fixed_pars(p))
            kep_rvs = kepler.rv_model(self.t, kep_ps[:,0], kep_ps[:,2], kep_ps[:,3], kep_ps[:,4], 2.0*np.pi/kep_ps[:,1])
            rvs += np.sum(kep_rvs, axis=0)

        return rvs

    def get_amps(self, p):
        amps = []

        if self.nkep > 0:
            amps = np.concatenate((amps, self.kep_deparameterize(self.kep_pars(p))[:,0]))
        if self.Pfixed is not None:
            amps = np.concatenate((amps, self.kep_fixed_deparameterize(self.kep_fixed_pars(p))[:,0]))

        return amps

    def get_periods(self, p):
        pers = []

        if self.nkep > 0:
            pers = np.concatenate((pers, self.kep_deparameterize(self.kep_pars(p))[:,1]))

        if self.Pfixed is not None:
            pers = np.concatenate((pers, self.Pfixed))

        return pers

    def get_eccs(self, p):
        es = []

        if self.nkep > 0:
            es = np.concatenate((es, self.kep_deparameterize(self.kep_pars(p))[:,2]))

        if self.Pfixed is not None:
            es = np.concatenate((es, self.kep_fixed_deparameterize(self.kep_fixed_pars(p))[:,1]))

        return es

    def __call__(self, p):
        ar1post = self.get_ar1(p)

        lp = 0.0

        # Flat prior on the predictors

        if self.nkep > 0:
            kp = self.kep_pars(p)

            # Flat prior on K in [0, 100 m/s]
            lp += np.sum(par.bounded_log_jacobian(kp[:,0], low=0, high=0.1))

            # Flat prior on P
            lp += np.sum(par.bounded_log_jacobian(kp[:,1], low=0, high=self.T))
            
            # Flat prior on e
            lp += np.sum(par.bounded_log_jacobian(kp[:,2], low=0, high=1))

            # Flat prior on omega
            lp += np.sum(par.bounded_log_jacobian(kp[:,3], low=0, high=2*np.pi))

            # Flat prior on chi
            lp += np.sum(par.bounded_log_jacobian(kp[:,4], low=0, high=1))

        if self.Pfixed is not None:
            kp = self.kep_fixed_pars(p)

            # Flat prior on K in [0, 100 m/s]
            lp += np.sum(par.bounded_log_jacobian(kp[:,0], low=0, high=0.1))

            # Flat prior on e
            lp += np.sum(par.bounded_log_jacobian(kp[:,1], low=0, high=1))

            # Flat prior on omega
            lp += np.sum(par.bounded_log_jacobian(kp[:,2], low=0, high=2*np.pi))

            # Flat prior on chi
            lp += np.sum(par.bounded_log_jacobian(kp[:,3], low=0, high=1))
            
        return lp + ar1post(self.ar1_pars(p))

    def predict(self, p):
        arp = self.get_ar1(p).predict(self.ar1_pars(p))

        kep_rvs = self.get_rvs(p)

        return arp[0] + np.dot(self.pred, self.coeffs(p)) + kep_rvs, arp[1]
