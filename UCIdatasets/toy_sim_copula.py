import numpy as np
import torch
import os
from scipy.stats import norm as scipy_norm

import UCIdatasets as datasets


class TOY_SIM_COPULA:

    class Data:
        def __init__(self, data):
            self.x = data.float()
            self.N = self.x.shape[0]

    def __init__(self, n_samples=50000, new=False,
                 case='Copula',
                 rho_neg=-0.5,
                 rho_pos=0.3):
        self.case = case
        self.rho_neg = rho_neg
        self.rho_pos = rho_pos

        data = load_data(n_samples=n_samples, new=new, case=case,
                         rho_neg=rho_neg, rho_pos=rho_pos)

        N = data.shape[0]
        N_test = int(0.1 * N)
        N_validate = int(0.1 * N)
        data_test = data[-N_test:]
        data_validate = data[-N_validate - N_test:-N_test]
        data_train = data[0:-N_validate - N_test]

        self.trn = data_train
        self.val = data_validate
        self.tst = data_test

        data_trn_val = torch.cat([self.trn, self.val], dim=0).numpy()
        self.mu = data_trn_val.mean(axis=0)
        self.sig = data_trn_val.std(axis=0)

        self.trn = self.Data(self.trn)
        self.val = self.Data(self.val)
        self.tst = self.Data(self.tst)

        self.n_dims = self.trn.x.shape[1]
        self.cat_dims = {}
        self.A = get_adj_matrix()
        self.Z_Sigma = get_cov_matrix()

        # Analytical: E[Y|do(X=x)] = rho(x) * Phi^{-1}((x+1)/2)
        # Compare do(X=-0.5) vs do(X=0.5) as reference treatment contrast
        u0 = np.clip((-0.5 + 1) / 2, 1e-6, 1 - 1e-6)
        u1 = np.clip((0.5 + 1) / 2, 1e-6, 1 - 1e-6)
        ey0 = float(rho_neg * scipy_norm.ppf(u0))  # x=-0.5 < 0 → use rho_neg
        ey1 = float(rho_pos * scipy_norm.ppf(u1))  # x=0.5 >= 0 → use rho_pos

        self.EY0 = torch.tensor(ey0).float()
        self.EY1 = torch.tensor(ey1).float()
        self.ATE = self.EY1 - self.EY0
        self.EY0_l = self.EY0
        self.EY0_u = self.EY0
        self.EY1_l = self.EY1
        self.EY1_u = self.EY1
        self.ATE_l = self.ATE
        self.ATE_u = self.ATE

        # Placeholder fields kept for training-script compatibility
        self.p_U = torch.tensor(0.0)
        self.p_AU0 = torch.tensor(0.0)
        self.p_AU1 = torch.tensor(0.0)
        self.p_YA0U0 = torch.tensor(0.0)
        self.p_YA0U1 = torch.tensor(0.0)
        self.p_YA1U0 = torch.tensor(0.0)
        self.p_YA1U1 = torch.tensor(0.0)
        self.n_dgp = 0

        self.dataset_filepath = (
            datasets.dataroot
            + f'toy_sim_copula/toy_sim_copula_{case}_{n_samples}_{rho_neg}_{rho_pos}.'
        )

        print(f'True E[Y|do(X=-0.5)] = {ey0:.5f}')
        print(f'True E[Y|do(X= 0.5)] = {ey1:.5f}')
        print(f'True ATE              = {self.ATE.item():.5f}')


def get_adj_matrix():
    A = np.zeros((2, 2))
    A[1, 0] = 1  # Y depends on X
    return torch.from_numpy(A).float()


def get_cov_matrix():
    # Identity: flow reference is isotropic Gaussian; the flow learns piecewise structure
    return torch.eye(2).float()


def load_data(n_samples=50000, new=False,
              case='Copula',
              rho_neg=-0.5,
              rho_pos=0.3):
    filepath = (
        datasets.dataroot
        + f'toy_sim_copula/toy_sim_copula_{case}_{n_samples}_{rho_neg}_{rho_pos}.pt'
    )

    if not new:
        try:
            data = torch.load(filepath)
            print(f'Copula data loaded from {filepath}')
            return data
        except Exception:
            return load_data(n_samples=n_samples, new=True, case=case,
                             rho_neg=rho_neg, rho_pos=rho_pos)

    with torch.no_grad():
        # X ~ Uniform(-1, 1)
        x = torch.FloatTensor(n_samples).uniform_(-1, 1)

        # Map X to standard-normal latent via its uniform marginal CDF
        u_x = ((x + 1) / 2).clamp(1e-6, 1 - 1e-6)
        z_x = torch.tensor(scipy_norm.ppf(u_x.numpy()), dtype=torch.float32)

        # Piecewise rho determined by the sign of x
        rho_vals = torch.where(
            x < 0,
            torch.full_like(x, rho_neg),
            torch.full_like(x, rho_pos),
        )

        # Z_Y | Z_X ~ N(rho * Z_X, sqrt(1 - rho^2))  [Gaussian copula conditional]
        cond_mean = rho_vals * z_x
        cond_std = torch.sqrt(1.0 - rho_vals ** 2)
        y = torch.randn(n_samples) * cond_std + cond_mean  # Y ~ N(0,1) marginally

    data = torch.stack([x, y], dim=1)

    os.makedirs(datasets.dataroot + 'toy_sim_copula', exist_ok=True)
    torch.save(data, filepath)
    print(f'Copula data generated and saved to {filepath}')
    return data
