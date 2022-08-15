import sys
import abc
import numpy as np
import scipy.special as sp 

# Python 2 & 3 handle ABCs differently
if sys.version_info[0] < 3:
    ABC = abc.ABCMeta('ABC', (), {})
else:
    ABC = abc.ABC


class Prior(ABC):
    """
    Abstract base class for prior objects.
    All prior objects should inherit from this class.
    
    """

    @abc.abstractmethod
    def draw_samples(self, num_samples):
        pass

    @abc.abstractmethod
    def compute_logprob(self, a, b):
        pass


class UniformPrior(Prior):
    
    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

    def __repr__(self):
        return "Uniform"

    def draw_samples(self, num_samples):
        """
        Draw positive samples from a Gaussian distribution.
        Negative samples will not be returned.
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            numpy array of float: samples drawn from the appropriate
            Gaussian distribution. Array has length `num_samples`.
        """
        # sample from a uniform distribution in log space
        a_samples = np.random.uniform(self.minval, self.maxval, num_samples)
        b_samples = np.random.uniform(self.minval, self.maxval, num_samples)

        samples=np.vstack((a_samples,b_samples)).T

        return samples

    def compute_logprob(self, a, b):
        """
        Returns the log of the prior probability for a tuple of the beta 
        hyperparameters (a,b).
        """
        normalizer = self.maxval - self.minval

        a_logprob = -np.log(1/normalizer)
        b_logprob = -np.log(1/normalizer) 

        logprob=a_logprob+b_logprob

        return logprob

    


class GaussianPrior(Prior):

""" 
    Gaussian prior. If no_negatives is set to True, this is technically a Truncated Gaussian prior
  
    Args:
        mu (float): mean of the distribution
        sigma (float): standard deviation of the distribution
        no_negatives (bool): if True, only positive values will be drawn from
            this prior, and the probability of negative values will be 0 (default:True).
"""

    def __init__(self, mu, sigma, no_negatives=True):
        self.mu = mu
        self.sigma = sigma
        self.no_negatives = no_negatives

    def __repr__(self):
        return "Gaussian"

    def draw_samples(self, num_samples):
        """
        Draw positive samples from a Gaussian distribution.
        Negative samples will not be returned.
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            numpy array of float: samples drawn from the appropriate
            Gaussian distribution. Array has length `num_samples`.
        """

        a_samples = np.random.normal(
            self.mu, self.sigma, size=num_samples
        )
        b_samples = np.random.normal(
            self.mu, self.sigma, size=num_samples
        )

        a_num_bad = np.inf
        b_num_bad = np.inf

        if self.no_negatives:

            while a_num_bad != 0 or b_num_bad!=0:

                # identify indices corresponding to negative values 
                
                a_bad_samples = np.where(a_samples < 0)[0]
                a_num_bad = len(a_bad_samples)

                b_bad_samples = np.where(b_samples < 0)[0]
                b_num_bad = len(b_bad_samples)

                # replace negative values with newly drawn numbers from the Gaussian prior
                # the loop will continue until all values in the array are positive
                
                a_samples[a_bad_samples] = np.random.normal(
                    loc=self.mu, scale=self.sigma, size=a_num_bad
                )
                
                b_samples[b_bad_samples] = np.random.normal(
                    loc=self.mu, scale=self.sigma, size=b_num_bad
                )

        samples=np.vstack((a_samples,b_samples)).T

        return samples

    def compute_logprob(self, a, b):
        """
        Compute log(probability) of an array of numbers wrt a Gaussian distibution.
        Negative numbers return a probability of -inf.
        Args:
            element_array (float or np.array of float): array of numbers. We want the
                probability of drawing each of these from the appopriate Gaussian
                distribution
        Returns:
            numpy array of float: array of log(probability) values,
            corresponding to the probability of drawing each of the numbers
            in the input `element_array`.
        """
        a_logprob = -0.5*np.log(2.*np.pi*self.sigma) - 0.5*((a- self.mu) / self.sigma)**2
        b_logprob = -0.5*np.log(2.*np.pi*self.sigma) - 0.5*((b- self.mu) / self.sigma)**2

        logprob=a_logprob+b_logprob
        
        return logprob


class LogUniformPrior(Prior):
    """
    This is the probability distribution : p(x) = 1/x`
    The __init__ method should take in a "min" and "max" value
    of the distribution, which correspond to the domain of the prior.
    (If this is not implemented, the prior has a singularity at 0 and infinite
    integrated probability).
    Args:
        minval (float): the lower bound of this distribution
        maxval (float): the upper bound of this distribution
    """

    def __init__(self, minval, maxval):
        self.minval = minval
        self.maxval = maxval

        self.logmin = np.log(minval)
        self.logmax = np.log(maxval)

    def __repr__(self):
        return "Log Uniform"

    def draw_samples(self, num_samples):
        """
        Draw samples from this 1/x distribution.
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            np.array:  samples ranging from [``minval``, ``maxval``) as floats.
        """
        # sample from a uniform distribution in log space
        a_samples = np.random.uniform(self.logmin, self.logmax, num_samples)
        b_samples = np.random.uniform(self.logmin, self.logmax, num_samples)
        # convert from log space to linear space
        a_samples = np.exp(a_samples)
        b_samples = np.exp(b_samples)

        samples=np.vstack((a_samples,b_samples)).T

        return samples

    def compute_logprob(self, a, b):
        """
        Returns the log of the prior probability for a tuple of the beta 
        hyperparameters (a,b).
        """
        normalizer = self.logmax - self.logmin

        a_logprob = -np.log((a*normalizer))
        b_logprob = -np.log((b*normalizer)) 

        logprob=a_logprob+b_logprob

        return logprob

class LogNormalPrior(Prior):
    """
    Log-normal distribution prior. 
    Args:
        mu (float): Mean of the logarithm of the distribution
        sigma (positive float): Standard Deviation of the logarithm of the distribution
    
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return "LogNormal"

    def draw_samples(self, num_samples):
        """
        Draw samples from the log normal distribution
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            np.array:  samples as floats.
        """
        # sample from a uniform distribution in log space
        a_samples = np.random.normal(self.mu, self.sigma, num_samples)
        b_samples = np.random.normal(self.mu, self.sigma, num_samples)
        # convert from log space to linear space
        a_samples = np.exp(a_samples)
        b_samples = np.exp(b_samples)

        samples=np.vstack((a_samples,b_samples)).T

        return samples

    def compute_logprob(self, a, b):
        """
        Returns the log of the prior probability for a tuple of the beta 
        hyperparameters (a,b).
        """

        a_logprob = (-1*((np.log(a)-self.mu)**2)/(2*self.sigma**2))/(a*self.sigma*np.sqrt(2*np.pi))
        b_logprob = (-1*((np.log(b)-self.mu)**2)/(2*self.sigma**2))/(a*self.sigma*np.sqrt(2*np.pi))

        logprob=a_logprob+b_logprob

        return logprob


class GammaPrior(Prior):
    """
    A prior following a functional form from the Gamma distribution. 
    Args:
        k (float): shape parameter
        theta (float): scale parameter
    """

    def __init__(self, k, theta):
        self.k = k
        self.theta = theta

    def __repr__(self):
        return "Gamma"

    def draw_samples(self, num_samples):
        """
        Draw samples from the Gamma prior
        Args:
            num_samples (float): the number of samples to generate
        Returns:
            np.array:  samples as floats.
        """
        # sample from a uniform distribution in log space
        a_samples = np.random.gamma(self.k, self.theta,num_samples)
        b_samples = np.random.gamma(self.k, self.theta, num_samples)

        samples=np.vstack((a_samples,b_samples)).T

        return samples

    def compute_logprob(self, a, b):
        """
        Returns the log of the prior probability for a tuple of the beta 
        hyperparameters (a,b).
        """

        a_logprob = np.log(((a**(self.k-1))*(np.exp(-a/self.theta)))/(((self.theta)**self.k)*sp.gamma(a)))
        b_logprob = np.log(((b**(self.k-1))*(np.exp(-b/self.theta)))/(((self.theta)**self.k)*sp.gamma(b)))

        logprob=a_logprob+b_logprob

        return logprob
