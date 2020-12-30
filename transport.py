import numpy as np

# the transport function h is often characterized as a deep neural network.
# it sounds a bit like half a GAN, which transforms a gaussian into some desired distribution.
class Transport:
    def __init__(self, u):
        # u refers to the weights of the NN
        self.u = u

    def update(self, i, du, lr=1):
        # standard nn weight update eqn, except for the np.sqrt term. i suspect it's annealing.
        self.u = self.u - lr / np.sqrt(i + 1) * du


class AffineTransport(Transport):
    # in all the cases where this is called, u is initialized as [2.718, 1].
    # erm there's only 2 weights in the NN?
    def __init__(self, u):
        Transport.__init__(self, u)

    # a bunch of samples are taken from N(0,1), and each sample is fed to h. h will then produce outputs
    # such that the outputs follow a distribution which is a scaled and translated version of the original.
    # eps is the epsilon in the paper. it's sampled from N(0,1).
    def h(self, eps):
        return eps * np.exp(self.u[0]) + np.exp(self.u[1])

    def du(self, eps):
        du = np.zeros(2)
        du[0] = eps * np.exp(self.u[0])
        du[1] = 1.0 * np.exp(self.u[1])
        return du
