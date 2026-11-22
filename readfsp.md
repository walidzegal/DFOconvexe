# FSP en Python et en C++

Réimplémentations autonomes de l'algorithme **FSP** (*Projection-based curve pattern search for
black-box optimization over smooth convex sets*, Jia, Lapucci & Mansueto, Optimization Methods and
Software, 2026), à partir du dépôt https://github.com/pierlumanzu/FSP (GPL-3.0 : ce code dérivé
est donc aussi à distribuer sous GPL-3.0, avec citation de l'article).

## Algorithme (ce que fait `solve`)

- Minimise une boîte noire `f` sur un convexe lisse, via un opérateur de projection `proj_op`.
- Directions : `±e_i`, `±1` (2n+2 directions), parcourues circulairement à partir de `resume_idx_d`.
- Point candidat : `proj(x + α d)`. Succès si `f(x_new) < f(x) − σ α²` (σ = 1e-3).
- Succès : `α ← max(1e-6, α/0.975)`. Échec sur toutes les directions : `α ← δ α` (δ = 0.5).
- Mode `Standard` : à partir de la 2ᵉ itération, on accepte la première direction réussie
  (la 1ʳᵉ itération évalue tout et garde la meilleure). Mode `Parallel` : toutes les directions
  sont évaluées à chaque itération, on garde la meilleure.
- Arrêt : `max_it`, `max_nf` ou `α ≤ min_alpha` (1e-7).

## Python (numpy seulement)

```bash
cd python
python main.py Standard      # ou Parallel ; option --csv sortie.csv
```

```python
from fsp import FSP
x, hist = FSP(max_nf=10_000).solve(f, proj_op, x0)   # f(x)->float, proj_op(x)->(x_proj, bool)
```

## C++17 (header-only)

```bash
cd cpp
make && ./fsp_bench Standard          # ou : cmake -B build && cmake --build build
```

```cpp
#include "fsp.hpp"
fsp::Options opt; opt.max_nf = 10000;
auto res = fsp::FSP(opt).solve(f, proj, x0);   // res.x, res.history, res.nf
```

## Différences avec le dépôt d'origine

- Vecteurs 1-D (pas de vecteurs colonnes) ; historique en listes (pas de pandas).
- `HS29_Original` : la projection sur l'ellipsoïde est calculée **exactement** (équation séculaire
  + bissection) au lieu de passer par `cvxopt`.
- Pas de génération du tableau LaTeX ni de sérialisation pickle.
- Python et C++ donnent les mêmes optima ; le nombre d'itérations peut différer de quelques unités
  sur 2 problèmes (HS22 c=5, HS43 en mode Parallel) à cause de l'ordre des sommes flottantes.
