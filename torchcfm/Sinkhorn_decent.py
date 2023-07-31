import torch
from geomloss import SamplesLoss


class SD(object):
    def __init__(self,  backend = None, blur = None, scaling = None) -> None:
        self.vector_field = None   #记录方向
        self.backend = backend
        self.blur = blur
        self.scaling = scaling
        self.potential_op = SamplesLoss(
            loss = 'sinkhorn', p = 2, blur = self.blur, potentials = True, 
            debias = False, backend = self.backend, scaling = self.scaling
        )


    def one_step_update(self, init_particles = None, init_mass = None, step_size = None, tgt_support = None, tgt_mass = None, **kw):
        self.particles = init_particles
        self.mass = init_mass
        self.particles.requires_grad = True
        first_var_ab, _ = self.potential_op(
            self.mass, self.particles, tgt_mass.contiguous(), tgt_support.contiguous()
        )
        first_var_aa, _ = self.potential_op(
            self.mass, self.particles, self.mass, self.particles 
        )
        first_var_ab_grad = torch.autograd.grad(
            torch.sum(first_var_ab), self.particles
        )[0]
        first_var_aa_grad = torch.autograd.grad(
            torch.sum(first_var_aa), self.particles
        )[0]
        with torch.no_grad():
            self.vector_field = first_var_ab_grad - first_var_aa_grad
            # self.vector.append(vector_field)
            self.particles = self.particles - step_size * self.vector_field

        self.particles.requires_grad = False
        # self.particles.grad.zero_()

    def get_state(self):
        return self.particles, self.mass, self.vector_field