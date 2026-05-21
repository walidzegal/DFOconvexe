#DFOconvexe
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable


# ──────────────────────────────────────────────
# 1. PROJECTION
# ──────────────────────────────────────────────

def proj_box(x, lb, ub):
    return np.clip(x, lb, ub)


# ──────────────────────────────────────────────
# 2. MODÈLE (INTERPOLATION)
# ──────────────────────────────────────────────

@dataclass
class Model:
    c: float
    g: np.ndarray
    x_center: np.ndarray

    def __call__(self, y):
        return self.c + self.g @ (y - self.x_center)

    def gradient(self):
        return self.g


def build_model(f, x_k, Delta, proj):
    """
    Modèle DFO pur (interpolation linéaire)
    m(y) = c + g^T (y - x_k)
    """
    n = len(x_k)

    # ── points d'interpolation ──
    Y = [x_k.copy()]

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0

        y_plus = proj(x_k + Delta * e)
        y_minus = proj(x_k - Delta * e)

        Y.append(y_plus)

        if np.linalg.norm(y_minus - x_k) > 1e-10:
            Y.append(y_minus)

    # garder n+1 points
    Y = Y[:n+1]

    # ── évaluation ──
    f_vals = np.array([f(y) for y in Y])

    # ── système interpolation ──
    M = np.zeros((len(Y), n + 1))
    for i, y in enumerate(Y):
        M[i, 0] = 1.0
        M[i, 1:] = y - x_k

    # résolution robuste
    params = np.linalg.lstsq(M, f_vals, rcond=None)[0]

    c = params[0]
    g = params[1:]

    return Model(c=c, g=g, x_center=x_k.copy())


# ──────────────────────────────────────────────
# 3. MESURE DE STATIONNARITÉ
# ──────────────────────────────────────────────

def stationarity_measure(x_k, g_k, proj):
    return np.linalg.norm(proj(x_k - g_k) - x_k)


# ──────────────────────────────────────────────
# 4. SOUS-PROBLÈME TRUST-REGION
# ──────────────────────────────────────────────

def solve_subproblem(model, x_k, Delta, proj, n_iter=50):

    y = x_k.copy()
    best_y = y.copy()
    best_val = model(y)

    alpha = Delta / n_iter

    def proj_TR(z):
        zc = proj(z)
        d = zc - x_k
        if np.linalg.norm(d) > Delta:
            zc = x_k + Delta * d / np.linalg.norm(d)
        return zc

    for _ in range(n_iter):
        g = model.gradient()
        y_new = proj_TR(y - alpha * g)
        val = model(y_new)

        if val < best_val:
            best_val = val
            best_y = y_new

        y = y_new

    return best_y


# ──────────────────────────────────────────────
# 5. ALGORITHME CDFO-TR
# ──────────────────────────────────────────────

@dataclass
class Result:
    x_opt: np.ndarray
    f_opt: float
    history_x: list = field(default_factory=list)
    history_f: list = field(default_factory=list)
    history_delta: list = field(default_factory=list)
    history_pi: list = field(default_factory=list)
    converged: bool = False


def cdfo_tr(f, x0, proj,
            Delta0=1.0,
            eps=1e-6,
            max_iter=200):

    x_k = proj(x0.copy())
    Delta = Delta0

    res = Result(x_opt=x_k, f_opt=f(x_k))

    for k in range(max_iter):

        f_k = f(x_k)

        # ── modèle ──
        model = build_model(f, x_k, Delta, proj)

        # ── stationnarité ──
        pi = stationarity_measure(x_k, model.g, proj)

        res.history_x.append(x_k.copy())
        res.history_f.append(f_k)
        res.history_delta.append(Delta)
        res.history_pi.append(pi)

        if pi < eps:
            res.converged = True
            break

        # ── sous-problème ──
        x_new = solve_subproblem(model, x_k, Delta, proj)

        f_new = f(x_new)

        pred = model(x_k) - model(x_new)
        actual = f_k - f_new

        rho = actual / pred if abs(pred) > 1e-12 else 0

        # ── acceptation ──
        if rho > 0.1:
            x_k = x_new

        # ── mise à jour Δ ──
        if rho > 0.75:
            Delta = min(2 * Delta, 10)
        elif rho < 0.1:
            Delta *= 0.5

    res.x_opt = x_k
    res.f_opt = f(x_k)

    return res


# ──────────────────────────────────────────────
# 6. EXEMPLE
# ──────────────────────────────────────────────

def example():

    A = np.array([[2, 1],
                  [1, 3],
                  [0.5, 2]])

    b = np.array([3, 5, 4])

    def f(x):
        r = A @ x - b
        return 0.5 * np.dot(r, r)

    lb = np.array([0, 0])
    ub = np.array([2, 2])
    proj = lambda x: proj_box(x, lb, ub)

    x0 = np.array([0.3, 0.3])

    res = cdfo_tr(f, x0, proj)

    print("Solution :", res.x_opt)
    print("Valeur :", res.f_opt)
    print("Convergence :", res.converged)

    return res, f


# ──────────────────────────────────────────────
# 7. VISUALISATION
# ──────────────────────────────────────────────

def plot(res, f):

    path = np.array(res.history_x)

    xs = np.linspace(0, 2, 200)
    ys = np.linspace(0, 2, 200)
    X, Y = np.meshgrid(xs, ys)

    Z = np.array([[f(np.array([x, y])) for x in xs] for y in ys])

    plt.figure(figsize=(6,5))
    plt.contourf(X, Y, Z, levels=30, cmap='plasma')

    plt.plot(path[:,0], path[:,1], 'w-o')
    plt.scatter(path[0,0], path[0,1], c='cyan', label='start')
    plt.scatter(res.x_opt[0], res.x_opt[1], c='red', label='opt')

    plt.legend()
    plt.title("CDFO-TR path")
    plt.show()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    res, f = example()
    plot(res, f)
