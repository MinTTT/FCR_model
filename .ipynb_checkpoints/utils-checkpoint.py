import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sciplot as sp
import seaborn as sns

class FCR_bac():
    def __init__(self, lambdas, const=None):
        if not const:
            self.phi_rb0 = 0.049
            self.gamma = 11.02
            self.lambda_c = 1.17
            self.phi_lacz_max = 21.8 * 10 ** 3
        self.lambda_i, self.lambda_f = lambdas
        self.phi_rb_i = self.phi_rb0 + self.lambda_i / self.gamma
        self.phi_rb_f = self.phi_rb0 + self.lambda_f / self.gamma
        self.sigma_f = self.lambda_f / self.phi_rb_f
        self.sigma_0 = self.lambda_i / self.phi_rb_i
        self.mp_0 = .5
        self.mrb_0 = self.mp_0 * self.phi_rb_i
        self.miu_f = self.lambda_f / (1. - self.lambda_f / self.lambda_c)
        self.t = np.linspace(0, 5, 500)
        self.t_ = np.linspace(-1, 0, 200, endpoint=False)
        self.j_r = None
        self.sm_t = None
        self.mrb_t = None
        self.sigma_t = None

    def ode_sigma(self, sm, t):
        sigma, m_rb = sm
        return [self.miu_f * sigma * ((1. - sigma / self.sigma_f) / (1. - sigma / self.gamma)),
                (self.phi_rb0 / (1. - sigma / self.gamma)) * sigma * m_rb]

    def integ_sigma(self, t=None):
        if t:
            self.t = t
        self.sm_t = odeint(self.ode_sigma, [self.sigma_0, self.mrb_0], self.t)
        self.sigma_t = self.sm_t[:, 0]
        self.mrb_t = self.sm_t[:, 1]
        self.j_r = self.sm_t[:, 0] * self.sm_t[:, 1]
        return None





if __name__ == '__main__':
    up_shift = FCR_bac([0.1, 0.5])
    up_shift.integ_sigma()
    sp.whitegrid()
    fig1, ax = plt.subplots(1, 1)
    sns.lineplot(x=up_shift.t, y=up_shift.j_r, label='J_Rb')
    sns.lineplot(x=up_shift.t, y=up_shift.mrb_t, label='M_Rb')
    # ax.set_yscale('log')
    fig1.show()