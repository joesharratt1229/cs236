import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        m , v = self.enc(x)
        z = ut.sample_gaussian(m, v)
        logits = self.dec(z)
        prior_mean , prior_variance = self.z_prior
        kl = ut.kl_normal(m, v, prior_mean, prior_variance)
        rec = -ut.log_bernoulli_with_logits(x, logits)

        nelbo = kl + rec
        return nelbo.mean(), kl.mean(), rec.mean()
        

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        prior_mean , prior_variance = self.z_prior
        m , v = self.enc(x)

        weighted_m , weighted_v , weighted_x = ut.duplicate(m, iw), ut.duplicate(v, iw), ut.duplicate(x, iw)

        weighted_z = ut.sample_gaussian(weighted_m, weighted_v)
        weighted_logits = self.dec(weighted_z)
        kl = ut.log_normal(weighted_z, weighted_m , weighted_v) - ut.log_normal_mixture(weighted_z, prior_mean, prior_variance)
        rec = - ut.log_bernoulli_with_logits(weighted_x, weighted_logits)

        nelbo = kl + rec
        niwae = -ut.log_mean_exp(-nelbo.reshape(iw, -1), dim = 0)

        return niwae.mean(), kl.mean(), rec.mean()

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
