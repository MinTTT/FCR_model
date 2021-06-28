import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sciplot as sp
import seaborn as sns


class FCR_bac():
    def __init__(self, lambdas, const=None):
        if const is None:
            self.phi_rb0 = 0.049
            self.gamma = 11.02
            self.lambda_c = 1.4  # def 1.17
            self.phi_lacz_max = 21.8 * 10 ** 3
        else:
            self.phi_rb0, self.gamma, self.lambda_c, self.phi_lacz_max = const

        self.lambda_i, self.lambda_f = lambdas  # initial and final growth rate lambda
        self.phi_rb_i = self.phi_rb0 + self.lambda_i / self.gamma
        self.phi_rb_f = self.phi_rb0 + self.lambda_f / self.gamma
        self.sigma_f = self.lambda_f / self.phi_rb_f
        self.sigma_0 = self.lambda_i / self.phi_rb_i
        self.mass_0 = 1.  # initial total mass
        self.j_0 = self.mass_0 * self.lambda_i
        self.mp_0 = .5  # initial total protein mass
        self.j_r_0 = self.sigma_0 * self.phi_rb_i * self.mp_0
        self.alphaM_over_alpha = self.j_0 / self.j_r_0
        self.mrb_0 = self.mp_0 * self.phi_rb_i  # initial ribosome affiliated protein
        self.miu_f = self.lambda_f / (1. - self.lambda_f / self.lambda_c)
        self.t = np.linspace(0, 2, 700)
        # initialize the t < 0, lambda_t, mp, j_r, mass, j
        self.t_ = np.linspace(-2, 0, 200, endpoint=False)
        self.lambda_t_ = self.lambda_i * np.ones(len(self.t_))
        self.mp_t_ = self.mp_0 * np.exp(self.lambda_i * self.t_)
        self.j_r_t_ = self.mp_t_ * self.lambda_i
        self.mass_t_ = self.mass_0 * np.exp(self.lambda_i * self.t_)
        self.j_t_ = self.lambda_i * self.mass_t_
        # set the variables when t > 0
        self.j_r_t = None
        self.mass_t = None
        self.lambda_t = None
        self.sm_t = None
        self.mrb_t = None
        self.sigma_t = None
        self.j_t = None
        self.chi_cat = None
        self.chi_rb = None
        # t range in minus and plus
        self.total_lambda = None
        self.total_j = None
        self.total_mass = None
        self.total_j_r = None
        self.total_t = None

    def ode_sigma(self, sm, t):
        sigma, m_rb, gr, mass = sm
        return [self.miu_f * sigma * ((1. - sigma / self.sigma_f) / (1. - sigma / self.gamma)),
                (self.phi_rb0 / (1. - sigma / self.gamma)) * sigma * m_rb,
                gr * (self.miu_f * (1 - self.lambda_f / self.lambda_c) - gr),
                self.alphaM_over_alpha * sigma * m_rb]




    def integ_sigma(self, t=None):
        if ~(t is None):
            self.t = t
        self.sm_t = odeint(self.ode_sigma, [self.sigma_0, self.mrb_0, self.lambda_i, self.mass_0], self.t)
        self.sigma_t = self.sm_t[:, 0]
        self.mrb_t = self.sm_t[:, 1]
        self.lambda_t = self.sm_t[:, 2]
        self.mass_t = self.sm_t[:, 3]
        self.j_r_t = self.sigma_t * self.mrb_t
        self.j_t = self.j_r_t * self.alphaM_over_alpha
        self.chi_rb = self.phi_rb0 / (1. - self.sigma_t / self.gamma)
        self.chi_cat = 1. - self.chi_rb * self.sigma_t / self.lambda_c
        self.total_t = np.hstack((self.t_, self.t))
        self.total_mass = np.hstack((self.mass_t_, self.mass_t))
        self.total_j = np.hstack((self.j_t_, self.j_t))
        self.total_j_r = np.hstack((self.j_r_t_, self.j_r_t))
        self.total_lambda = np.hstack((self.lambda_t_, self.lambda_t))
        return None

#%%

up_shift = FCR_bac([1.0, 0.2])
up_shift.integ_sigma(t=np.linspace(0, 8, 5000))
sp.whitegrid()
fig1, ax = plt.subplots(2, 2)
sns.lineplot(x=up_shift.sigma_t, y=up_shift.chi_rb, label='$\chi_{\mathrm{Rb}}$', ax=ax[0, 0])
sns.lineplot(x=up_shift.sigma_t, y=up_shift.chi_cat, label='$\chi_{\mathrm{cat}}$', ax=ax[0, 0])
sns.lineplot(x=up_shift.t, y=up_shift.sigma_t, label='$\sigma_{\mathrm{t}}$', ax=ax[0, 1])
sns.lineplot(x=up_shift.total_t, y=up_shift.total_lambda, label='$\lambda$', ax=ax[1, 0])
sns.lineplot(x=up_shift.total_t, y=up_shift.total_j, label='Flux', ax=ax[1, 1])
ax[1, 1].set_yscale('log')
# ax.set_yscale('log')
fig1.show()
