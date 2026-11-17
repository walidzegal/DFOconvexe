// Recherche directe projetée avec backtracking curviligne (sans dérivées).
//
// Schéma : à chaque itération, le Poll génère des directions d ; pour chacune,
// on teste l'arc admissible x(alpha) = Proj_C(x_k + alpha d) avec backtracking sur alpha.
// Succès -> Delta augmente ; échec -> Delta diminue ; arrêt quand Delta < tol ou budget épuisé.
//
// Compilation : g++ -O2 -std=c++17 rdp_curviligne.cpp -o rdp_curviligne

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <utility>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;  // liste de vecteurs (lignes)

// ---------- Utilitaires ----------
static double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static double norme(const Vec& a) { return std::sqrt(dot(a, a)); }

// ---------- Projecteurs sur des convexes simples ----------
// centre vide => centre à l'origine
Vec proj_boule(const Vec& x, const Vec& centre = {}, double rayon = 1.0) {
    const size_t n = x.size();
    Vec v(n);
    for (size_t i = 0; i < n; ++i) v[i] = x[i] - (centre.empty() ? 0.0 : centre[i]);
    const double nv = norme(v);
    if (nv <= rayon) return x;
    Vec y(n);
    for (size_t i = 0; i < n; ++i)
        y[i] = (centre.empty() ? 0.0 : centre[i]) + rayon * v[i] / nv;
    return y;
}

Vec proj_boite(const Vec& x, const Vec& lb, const Vec& ub) {
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = std::clamp(x[i], lb[i], ub[i]);
    return y;
}

// ---------- Directions du Poll ----------
// Base orthonormale aléatoire Q (Gram-Schmidt modifié, deux passes),
// puis ensemble générateur positif {±q_i}. Retourne 2n directions.
Mat directions_poll(int n, std::mt19937_64& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    Mat Q;
    Q.reserve(n);
    while (static_cast<int>(Q.size()) < n) {
        Vec v(n);
        for (auto& vi : v) vi = gauss(rng);
        for (int pass = 0; pass < 2; ++pass) {  // ré-orthogonalisation
            for (const auto& q : Q) {
                const double c = dot(v, q);
                for (int i = 0; i < n; ++i) v[i] -= c * q[i];
            }
        }
        const double nv = norme(v);
        if (nv < 1e-12) continue;  // quasi dépendant : on retire un autre vecteur
        for (auto& vi : v) vi /= nv;
        Q.push_back(std::move(v));
    }
    Mat D = Q;
    for (const auto& q : Q) {
        Vec m(n);
        for (int i = 0; i < n; ++i) m[i] = -q[i];
        D.push_back(std::move(m));
    }
    return D;
}

// ---------- Algorithme ----------
struct Resultat {
    Vec x;
    double fx;
    int n_eval;
    std::vector<std::pair<int, double>> historique;
};

using Fonction = std::function<double(const Vec&)>;
using Projecteur = std::function<Vec(const Vec&)>;

Resultat rdp_curviligne(const Fonction& f, const Projecteur& proj, const Vec& x0,
                        double delta0 = 1.0, double tol = 1e-8, int max_eval = 5000,
                        double theta = 0.5, double gamma = 2.0, double delta_max = 10.0,
                        double c_dec = 1e-4, int max_backtrack = 5,
                        unsigned long long seed = 0) {
    std::mt19937_64 rng(seed);
    Vec x = proj(x0);  // Initialisation
    double fx = f(x);
    int n_eval = 1;
    double delta = delta0;
    std::vector<std::pair<int, double>> historique{{n_eval, fx}};

    while (delta >= tol && n_eval < max_eval) {  // Arrêt ?
        bool succes = false;
        for (const Vec& d : directions_poll(static_cast<int>(x.size()), rng)) {  // Poll
            double alpha = delta;
            for (int k = 0; k <= max_backtrack; ++k) {  // Backtracking sur alpha
                if (n_eval >= max_eval) break;
                Vec z(x.size());
                for (size_t i = 0; i < x.size(); ++i) z[i] = x[i] + alpha * d[i];
                Vec y = proj(z);  // Arc admissible
                const double fy = f(y);
                ++n_eval;
                if (fy < fx - c_dec * delta * delta) {  // Décroissance suffisante ?
                    x = std::move(y);
                    fx = fy;
                    succes = true;
                    break;
                }
                alpha *= theta;
            }
            if (succes || n_eval >= max_eval) break;
        }
        delta = succes ? std::min(gamma * delta, delta_max) : theta * delta;
        historique.emplace_back(n_eval, fx);
    }
    return {x, fx, n_eval, historique};
}

// ---------- Exemple : minimiser ||x - c||^2 sur la boule unité ----------
int main() {
    const int n = 5;
    const Vec c(n, 2.0);
    Fonction f = [&c](const Vec& x) {
        double s = 0.0;
        for (size_t i = 0; i < x.size(); ++i) s += (x[i] - c[i]) * (x[i] - c[i]);
        return s;
    };
    Projecteur proj = [](const Vec& x) { return proj_boule(x, {}, 1.0); };

    Resultat r = rdp_curviligne(f, proj, Vec(n, 0.0));

    const double nc = norme(c);
    Vec x_ref(n);
    for (int i = 0; i < n; ++i) x_ref[i] = c[i] / nc;  // solution exacte

    Vec diff(n);
    for (int i = 0; i < n; ++i) diff[i] = r.x[i] - x_ref[i];

    std::printf("f* trouvé   : %.10f\n", r.fx);
    std::printf("f* exact    : %.10f\n", f(x_ref));
    std::printf("erreur ||x - x*|| : %.2e\n", norme(diff));
    std::printf("évaluations : %d\n", r.n_eval);
    return 0;
}
