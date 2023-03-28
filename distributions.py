"""
These are a few multivariate distributions we might use.
all implement a `logpdf` method to generate the log density function.
all implement a `sample` method to generate random variates.
"""
from abc import ABC, abstractmethod
from inspect import signature
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t

RNG_SEED: int = 1234


class Distribution(ABC):
    """
    Abstract Distribution Class
    """

    def __init__(self) -> None:
        self.rng = np.random.default_rng(RNG_SEED)
        self.n_features: int = 0

    # Note the rng object cannot be serialized as it's a generator
    def to_json(self) -> Dict[str, Any]:
        params = self.get_params()
        json_dict = {"dist_name": self.__class__.__name__, "params": params}

        return json_dict

    def set_params(self, params: Dict[str, Any]) -> None:
        # Force mypy to ignore this
        # Can't think of a "good" way to reinit all child classes
        self.__init__(**params)  # type: ignore

    def get_params(self) -> Dict[str, Any]:
        sig = signature(self.__class__)
        params = {p: sig.parameters[p].annotation for p in sig.parameters}
        params = {p: self.__getattribute__(p) for p in params}

        return params

    @abstractmethod
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        pass


class Univariate(Distribution):
    pass


class Multivariate(Distribution):
    pass


class MeanVarDist(Multivariate):
    def __init__(self, loc: np.ndarray, cov: np.ndarray) -> None:
        """
        loc: (..., n_features)
        cov: (..., n_features, n_features)
        """
        super().__init__()
        self.loc = loc
        self.cov = cov


class MultiNormDistribution(MeanVarDist):
    """
    Multivariate Normal Distribution
    """

    def __init__(self, loc: np.ndarray, cov: np.ndarray) -> None:
        super().__init__(loc, cov)
        # broadcast loc and cov
        self.broadcast = np.broadcast(self.loc[..., None], self.cov)
        self.broadcast_shape = self.broadcast.shape[:-2]
        self.n_features = self.broadcast.shape[-1]
        # compute cholesky decomposition of covariance
        self.chol = np.linalg.cholesky(self.cov)
        self.inv_cov = np.linalg.pinv(self.cov)
        _, self.log_det = np.linalg.slogdet(self.cov)

    def get_distance(self, x: np.ndarray) -> np.ndarray:
        z = x - self.loc
        z = np.einsum("...i,...ij,...j->...", z, self.inv_cov, z)
        return np.array(z)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        z = self.get_distance(x)
        b = self.log_det + self.n_features * np.log(2 * np.pi)
        b = -(b + z) / 2.0
        return np.array(b)

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        if isinstance(size, int):
            size = (size,)
        shape = size + self.broadcast_shape + (self.n_features,)
        # sample standard normal
        z = self.rng.standard_normal(size=shape)
        z = self.loc + np.einsum("...ij,...j->...i", self.chol, z)
        return np.array(z)


class MultiTDistribution(MeanVarDist):
    """
    Multivariate T Distribution
    """

    def __init__(self, loc: np.ndarray, cov: np.ndarray, dof: float = 1.00) -> None:
        """
        loc: (..., n_features)
        cov: (..., n_features, n_features)
        dof = np.inf will use a normal distribution
        """
        super().__init__(loc, cov)
        self.dof = dof
        # broadcast loc and cov
        self.loc = loc
        self.cov = cov
        self.broadcast = np.broadcast(loc[..., None], cov)
        self.broadcast_shape = self.broadcast.shape[:-2]
        self.n_features = self.broadcast.shape[-1]
        # setup normal distribution centered at 0
        e = np.zeros(self.broadcast_shape + (self.n_features,))
        self.normal = MultiNormDistribution(e, self.cov)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        z = self.normal.get_distance(x - self.loc)
        a = (self.n_features + self.dof) / 2.0
        z = a * np.log1p(z / self.dof)
        b = gammaln(a) - gammaln(self.dof / 2.0)
        b = b - self.n_features * np.log(np.pi * self.dof) / 2.0
        b = b - (self.normal.log_det / 2.0) - z
        return np.array(b)

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        if isinstance(size, int):
            size = (size,)
        shape = size + self.broadcast_shape
        if np.isinf(self.dof):
            # reduces to normal distribution
            x = np.ones(shape)
        else:
            x = self.rng.chisquare(self.dof, size=shape)
            x = np.sqrt(x / self.dof)
        z = self.normal.sample(size=size)
        z = self.loc + z / x[..., None]
        return np.array(z)


class GaussianKDE(Multivariate):
    """
    Gaussian KDE
    """

    def __init__(
        self,
        samples: np.ndarray,
        weights: Optional[np.ndarray] = None,
        bandwidth: float = 1.00,
    ) -> None:
        """
        samples: (n_samples, n_features)
        weights: (n_samples,)
        """
        super().__init__()
        self.samples = samples
        self.n_samples, self.n_features = self.samples.shape
        if weights is None:
            self.weights = np.ones(self.n_samples, dtype=float)
        else:
            self.weights = weights
        assert self.weights.shape == (self.n_samples,)
        self.weights = self.weights / np.sum(self.weights)
        self.cov = np.cov(self.samples, aweights=self.weights, rowvar=False)
        # Scott's Rule
        self.factor = pow(np.sum(self.weights**2), 1.0 / (self.n_features + 4))
        self.factor = self.factor * bandwidth
        self.kde_cov = self.cov * (self.factor**2)
        self.normals = MultiNormDistribution(self.samples, self.kde_cov)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        z = self.normals.logpdf(x[..., None, :])
        z_max = np.max(z, axis=-1, keepdims=True)
        z = np.dot(np.exp(z - z_max), self.weights)
        z = np.log(z) + np.squeeze(z_max, axis=-1)
        return np.array(z)

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        # will simulate n_variates per each sample
        return self.normals.sample(size)


class MixtureDistribution(Multivariate):
    """
    Mixture Distribution
    """

    def __init__(self, components: List[Distribution], weights: List[float]) -> None:
        super().__init__()
        self.components = components
        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)
        assert len(self.components) == len(self.weights)
        self.n_components = len(self.components)
        self.n_features = self.components[0].n_features
        for component in self.components:
            assert self.n_features == component.n_features

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        # TODO: refactor to use safe `logsumexp`
        p = np.array(0.0, dtype=float)
        for i, component in enumerate(self.components):
            p = p + self.weights[i] * np.exp(component.logpdf(x))
        p = np.log(p)
        return np.array(p)

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        choice = self.rng.choice(self.n_components, p=self.weights, size=size)
        sim = np.array(0.0, dtype=float)
        for i, component in enumerate(self.components):
            z = component.sample(size)
            # broadcast and copy
            sim, z = np.broadcast_arrays(sim, z)
            sim, z = 1 * sim, 1 * z
            sim[choice == i] = z[choice == i]
        sim = np.array(sim, dtype=float)
        return np.array(sim)


class Gamma(Univariate):
    def __init__(self, shape: float, scale: float) -> None:
        super().__init__()
        self.shape = shape
        self.scale = scale
        self.dist: Any = self.rng.gamma

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        res: np.ndarray = self.dist(self.shape, self.scale, size)
        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


class Binomial(Univariate):
    def __init__(self, n: int, p: float) -> None:
        super().__init__()
        self.n = n
        self.p = p
        self.dist: Any = self.rng.binomial

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        res: np.ndarray = self.dist(self.n, self.p, size)
        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


def rescale(
    x: np.ndarray, new_min: float, new_max: float, axis: int = -1
) -> np.ndarray:
    """
    Shift a distribution of variables to a new min and max.
    Default axis is the last one.
    """
    old_rng = x.max(axis) - x.min(axis)
    new_rng = new_max - new_min
    rescaled: np.ndarray = ((x - x.min(axis)) * new_rng) / old_rng + new_min
    return rescaled


class Beta(Univariate):
    def __init__(self, a: float, b: float, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.a = a
        self.b = b
        self.min_val = min_val
        self.max_val = max_val
        self.dist: Any = self.rng.beta

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        res: np.ndarray = self.dist(self.a, self.b, size)
        res = rescale(res, self.min_val, self.max_val)
        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


class Poisson(Univariate):
    def __init__(self, lam: float, min_val: int = 0):
        super().__init__()
        self.lam = lam
        self.min_val = min_val
        self.dist: Any = self.rng.poisson

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        res: np.ndarray = self.dist(self.lam, size)
        res[res < self.min_val] = self.min_val
        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


class UniformInt(Univariate):
    def __init__(self, low: int, high: int):
        super().__init__()
        self.low = low
        self.high = high
        self.dist: Any = self.rng.choice

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        res: np.ndarray = self.dist(range(self.low, self.high + 1), size)
        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


class Normal(Univariate):
    def __init__(self, mu: float, sd: float):
        super().__init__()
        self.mu = mu
        self.sd = sd
        self.dist: Any = self.rng.normal

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        res: np.ndarray = self.dist(self.mu, self.sd, size)

        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


class StudentT(Univariate):
    def __init__(self, loc: float, scale: float, dof: float):
        self.loc = loc
        self.scale = scale
        self.dof = dof

        dist = student_t
        dist.random_state = np.random.RandomState(RNG_SEED)
        self.dist: Any = dist.rvs

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        res: np.ndarray = self.dist(self.dof, self.loc, self.scale, size=size)
        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass


class UniformPiecewise(Univariate):
    def __init__(self, parts: List[Tuple[Tuple[float, float], float]]):
        """
        Specify a set of ranges to draw from uniformly, and then combine together
        according to specified probabilities for each range

        parts: [(low period, high period), draw probability]
            e.g. [((0, 1), 0.5), ((1, 5), 0.25), ((5, 10), 0.25)]
        """
        super().__init__()

        total_p = sum([part[1] for part in parts])
        assert np.allclose(total_p, 1), f"sum of prob is {total_p} but must be 1"

        for i in range(1, len(parts)):
            low = parts[i][0][0]
            prior_high = parts[i - 1][0][1]
            assert low == prior_high, "ranges must be non-overlapping and fully covered"

        self.parts = parts
        self.dist: Any = self.rng.uniform

    def sample(self, size: Union[int, Tuple[int, ...]] = 10000) -> np.ndarray:
        sample: List[np.ndarray] = []
        remaining = np.array(size)
        for i, part in enumerate(self.parts):
            low = part[0][0]
            high = part[0][1]
            prob = part[1]
            n = (np.array(prob) * size).astype(int)
            if i == (len(self.parts) - 1):  # final element
                sample.append(self.dist(low, high, remaining))
            else:
                sample.append(self.dist(low, high, n))
                remaining -= n
        res: np.ndarray = np.concatenate(sample)
        return res

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        pass
