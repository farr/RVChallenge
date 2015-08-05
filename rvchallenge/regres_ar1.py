import arfit.ar1_kalman as ak
import numpy as np

class AR1RegresPosterior(object):
    def __init__(self, t, y, dy, pred):
        self.t = t
        self.y = y
        self.dy = dy
        self.pred = pred

    def coeffs(self, p):
        return p[4:]
        
    def get_ar1(self, p):
        resid = self.y - np.dot(self.pred, self.coeffs(p))

        return ak.AR1KalmanPosterior(self.t, resid, self.dy)

    def __call__(self, p):
        ar1post = self.get_ar1(p)

        # Flat prior on the predictors

        return ar1post(p[:4])

    def predict(self, p):
        arp = self.get_ar1(p).predict(p[:4])

        return arp[0] + np.dot(self.pred, self.coeffs(p)), arp[1]
