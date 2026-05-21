import numpy as np
import scipy.sparse
from scipy.special import expit


class BaseSmoothOracle(object):
    def func(self, x):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = b.shape[0]

    def func(self, x):
        Ax = self.matvec_Ax(x)
        z = -self.b * Ax
        log_1_exp = np.logaddexp(0, z)
        loss = np.sum(log_1_exp) / self.m
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        Ax = self.matvec_Ax(x)
        sigma = -self.b * expit(-self.b * Ax)
        grad_loss = self.matvec_ATx(sigma) / self.m
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        Ax = self.matvec_Ax(x)
        t = expit(-self.b * Ax)
        s = t * (1 - t)
        H_loss = self.matmat_ATsA(s) / self.m
        H_reg = self.regcoef * np.eye(x.shape[0])
        return H_loss + H_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._cached_Ax = None
        self._cached_x = None
        self._cached_Ad = None
        self._cached_d = None
        self._cached_x_hat = None
        self._cached_Ax_hat = None

    def _get_Ax(self, x):
        if self._cached_x is not None and np.array_equal(x, self._cached_x):
            return self._cached_Ax
        Ax = self.matvec_Ax(x)
        self._cached_x = x.copy()
        self._cached_Ax = Ax
        return Ax

    def _get_Ax_and_Ad(self, x, d):
        Ax = self._get_Ax(x)
        if self._cached_d is not None and np.array_equal(d, self._cached_d):
            Ad = self._cached_Ad
        else:
            Ad = self.matvec_Ax(d)
            self._cached_d = d.copy()
            self._cached_Ad = Ad
        return Ax, Ad

    def func(self, x):
        Ax = self._get_Ax(x)
        z = -self.b * Ax
        log_1_exp = np.logaddexp(0, z)
        loss = np.sum(log_1_exp) / self.m
        reg = 0.5 * self.regcoef * np.dot(x, x)
        return loss + reg

    def grad(self, x):
        Ax = self._get_Ax(x)
        sigma = -self.b * expit(-self.b * Ax)
        grad_loss = self.matvec_ATx(sigma) / self.m
        grad_reg = self.regcoef * x
        return grad_loss + grad_reg

    def hess(self, x):
        Ax = self._get_Ax(x)
        t = expit(-self.b * Ax)
        s = t * (1 - t)
        H_loss = self.matmat_ATsA(s) / self.m
        H_reg = self.regcoef * np.eye(x.shape[0])
        return H_loss + H_reg

    def func_directional(self, x, d, alpha):
        Ax, Ad = self._get_Ax_and_Ad(x, d)
        x_hat = x + alpha * d
        if self._cached_x_hat is not None and np.array_equal(x_hat, self._cached_x_hat):
            Ax_hat = self._cached_Ax_hat
        else:
            Ax_hat = Ax + alpha * Ad
            self._cached_x_hat = x_hat.copy()
            self._cached_Ax_hat = Ax_hat
        z = -self.b * Ax_hat
        log_1_exp = np.logaddexp(0, z)
        loss = np.sum(log_1_exp) / self.m
        reg = 0.5 * self.regcoef * np.dot(x_hat, x_hat)
        return loss + reg

    def grad_directional(self, x, d, alpha):
        Ax, Ad = self._get_Ax_and_Ad(x, d)
        x_hat = x + alpha * d
        if self._cached_x_hat is not None and np.array_equal(x_hat, self._cached_x_hat):
            Ax_hat = self._cached_Ax_hat
        else:
            Ax_hat = Ax + alpha * Ad
            self._cached_x_hat = x_hat.copy()
            self._cached_Ax_hat = Ax_hat
        sigma = -self.b * expit(-self.b * Ax_hat)
        grad_hat = self.matvec_ATx(sigma) / self.m + self.regcoef * x_hat
        return np.dot(grad_hat, d)


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    def matvec_Ax(x):
        return A.dot(x)

    def matvec_ATx(x):
        return A.T.dot(x)

    def matmat_ATsA(s):
        if scipy.sparse.issparse(A):
            As = A.multiply(s.reshape(-1, 1))
            return A.T.dot(As).toarray()
        else:
            As = A * s.reshape(-1, 1)
            return A.T @ As

    if oracle_type == 'usual':
        oracle_cls = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle_cls = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
    return oracle_cls(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    n = x.shape[0]
    grad = np.zeros(n)
    f0 = func(x)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        grad[i] = (func(x_plus) - f0) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    n = x.shape[0]
    H = np.zeros((n, n))
    f0 = func(x)
    f_plus = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus[i] = func(x_plus)
    for i in range(n):
        for j in range(n):
            x_pp = x.copy()
            x_pp[i] += eps
            x_pp[j] += eps
            f_pp = func(x_pp)
            H[i, j] = (f_pp - f_plus[i] - f_plus[j] + f0) / (eps * eps)
    H = (H + H.T) / 2
    return H
