import numpy as np
from numpy.linalg import LinAlgError
import scipy
from collections import defaultdict
import time


class LineSearchTool(object):
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def _armijo_backtrack(self, oracle, x_k, d_k, alpha_init):
        alpha = alpha_init
        phi0 = oracle.func_directional(x_k, d_k, 0.0)
        grad0 = oracle.grad_directional(x_k, d_k, 0.0)
        while True:
            phi_alpha = oracle.func_directional(x_k, d_k, alpha)
            if phi_alpha <= phi0 + self.c1 * alpha * grad0:
                break
            alpha *= 0.5
        return alpha

    def _wolfe_search(self, oracle, x_k, d_k, alpha_init=1.0, maxiter=100):
        """
        Находит шаг, удовлетворяющий сильным условиям Вульфа,
        используя метод расширения интервала и бинарного поиска.
        """
        alpha = alpha_init
        phi0 = oracle.func_directional(x_k, d_k, 0.0)
        grad0 = oracle.grad_directional(x_k, d_k, 0.0)
        c1 = self.c1
        c2 = self.c2
        abs_grad0 = abs(grad0)

        # --- Фаза 1: увеличение шага, пока выполняется условие Армихо ---
        phi_alpha = oracle.func_directional(x_k, d_k, alpha)
        while phi_alpha <= phi0 + c1 * alpha * grad0:
            grad_alpha = oracle.grad_directional(x_k, d_k, alpha)
            if abs(grad_alpha) <= c2 * abs_grad0:
                return alpha
            # Увеличиваем шаг
            alpha *= 2.0
            phi_alpha = oracle.func_directional(x_k, d_k, alpha)

        # --- Фаза 2: бинарный поиск на интервале [alpha/2, alpha] ---
        lo = alpha / 2.0
        hi = alpha
        for _ in range(maxiter):
            mid = (lo + hi) / 2.0
            phi_mid = oracle.func_directional(x_k, d_k, mid)
            if phi_mid > phi0 + c1 * mid * grad0:
                hi = mid
            else:
                grad_mid = oracle.grad_directional(x_k, d_k, mid)
                if abs(grad_mid) <= c2 * abs_grad0:
                    return mid
                if grad_mid >= 0:
                    hi = mid
                else:
                    lo = mid
        # Если не нашли, возвращаем левую границу (удовлетворяет Армихо)
        return lo

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        if self._method == 'Constant':
            return self.c
        elif self._method == 'Armijo':
            alpha0 = previous_alpha if previous_alpha is not None else self.alpha_0
            return self._armijo_backtrack(oracle, x_k, d_k, alpha0)
        elif self._method == 'Wolfe':
            alpha0 = previous_alpha if previous_alpha is not None else 1.0
            return self._wolfe_search(oracle, x_k, d_k, alpha0)
        else:
            raise ValueError('Unknown method {}'.format(self._method))


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = time.time()

    if display:
        print("Starting gradient descent...")

    # Начальная точка
    if trace:
        history['time'].append(time.time() - start_time)
        history['func'].append(oracle.func(x_k))
        grad_norm0 = np.linalg.norm(oracle.grad(x_k))
        history['grad_norm'].append(grad_norm0)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    grad0 = oracle.grad(x_k)
    grad_norm0_sq = np.dot(grad0, grad0)
    if grad_norm0_sq == 0:
        if display:
            print("Initial gradient is zero, stopping.")
        return x_k, 'success', history

    for it in range(max_iter):
        grad = oracle.grad(x_k)
        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq <= tolerance * grad_norm0_sq:
            if display:
                print(f"Converged at iteration {it+1}")
            return x_k, 'success', history

        d_k = -grad
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=None)
        if alpha is None:
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k

        # Проверяем точность после обновления
        grad_new = oracle.grad(x_k)
        grad_new_norm_sq = np.dot(grad_new, grad_new)
        if grad_new_norm_sq <= tolerance * grad_norm0_sq:
            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(grad_new))
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            if display:
                print(f"Converged after update at iteration {it+1}")
            return x_k, 'success', history

        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            return x_k, 'computational_error', history

        if trace:
            history['time'].append(time.time() - start_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(grad_new))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f"Iter {it+1}: f={oracle.func(x_k):.6e}, ||g||={np.linalg.norm(grad_new):.6e}, alpha={alpha:.4e}")

    # Финальная проверка после исчерпания итераций
    grad_final = oracle.grad(x_k)
    grad_final_norm_sq = np.dot(grad_final, grad_final)
    if grad_final_norm_sq <= tolerance * grad_norm0_sq:
        if trace:
            # Добавляем последнюю точку, если её ещё нет
            if len(history['func']) == 0 or history['func'][-1] != oracle.func(x_k):
                history['time'].append(time.time() - start_time)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(grad_final))
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = time.time()

    if display:
        print("Starting Newton's method...")

    if trace:
        history['time'].append(time.time() - start_time)
        history['func'].append(oracle.func(x_k))
        grad_norm0 = np.linalg.norm(oracle.grad(x_k))
        history['grad_norm'].append(grad_norm0)
        if x_k.size <= 2:
            history['x'].append(x_k.copy())

    grad0 = oracle.grad(x_k)
    grad_norm0_sq = np.dot(grad0, grad0)
    if grad_norm0_sq == 0:
        if display:
            print("Initial gradient is zero, stopping.")
        return x_k, 'success', history

    for it in range(max_iter):
        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            return x_k, 'computational_error', history

        grad = oracle.grad(x_k)
        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            return x_k, 'computational_error', history

        grad_norm_sq = np.dot(grad, grad)
        if grad_norm_sq <= tolerance * grad_norm0_sq:
            if display:
                print(f"Converged at iteration {it+1}")
            return x_k, 'success', history

        # Защита от расходимости
        if np.linalg.norm(x_k) > 1e10:
            return x_k, 'computational_error', history

        try:
            H = oracle.hess(x_k)
            if np.any(np.isnan(H)) or np.any(np.isinf(H)):
                return x_k, 'computational_error', history
            cho = scipy.linalg.cho_factor(H, lower=True)
            d_k = scipy.linalg.cho_solve(cho, -grad)
        except LinAlgError:
            return x_k, 'newton_direction_error', history
        except Exception:
            return x_k, 'computational_error', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=None)
        if alpha is None:
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k

        # Проверяем точность после обновления
        grad_new = oracle.grad(x_k)
        grad_new_norm_sq = np.dot(grad_new, grad_new)
        if grad_new_norm_sq <= tolerance * grad_norm0_sq:
            if trace:
                history['time'].append(time.time() - start_time)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(grad_new))
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
            if display:
                print(f"Converged after update at iteration {it+1}")
            return x_k, 'success', history

        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            return x_k, 'computational_error', history

        if trace:
            history['time'].append(time.time() - start_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(grad_new))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())

        if display:
            print(f"Iter {it+1}: f={oracle.func(x_k):.6e}, ||g||={np.linalg.norm(grad_new):.6e}, alpha={alpha:.4e}")

    # Финальная проверка
    grad_final = oracle.grad(x_k)
    grad_final_norm_sq = np.dot(grad_final, grad_final)
    if grad_final_norm_sq <= tolerance * grad_norm0_sq:
        if trace:
            if len(history['func']) == 0 or history['func'][-1] != oracle.func(x_k):
                history['time'].append(time.time() - start_time)
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(np.linalg.norm(grad_final))
                if x_k.size <= 2:
                    history['x'].append(x_k.copy())
        return x_k, 'success', history
    return x_k, 'iterations_exceeded', history
