// Algorithme hybride MADS (style NOMAD) + recherche directe projetée curviligne.
//
// Idées reprises de NOMAD / MADS :
//   - taille de sondage Δp et taille de maillage Δm = min(Δp, Δp²)
//   - Poll avec base orthonormale aléatoire (Householder) et ensemble générateur positif {±h_j}
//   - Poll opportuniste, avec la dernière direction gagnante essayée en premier
//   - étape Search spéculative : extrapolation le long du dernier pas réussi
//
// Idées reprises de la recherche directe projetée curviligne :
//   - les contraintes convexes C sont traitées par projection : arc admissible x(α) = Π_C(x + α d)
//   - backtracking sur α, avec décroissance suffisante f(y) < f(x) - c Δp²
//   - le backtracking s'arrête quand α < Δm : l'arc explore l'écart entre Δp et Δm
//
// Compilation : g++ -O2 -std=c++17 hybride_nomad_curviligne.cpp -o hybride

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;
using Fonction = std::function<double(const Vec&)>;
using Projecteur = std::function<Vec(const Vec&)>;

// ---------- Utilitaires ----------
static double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}
static double norme(const Vec& a) { return std::sqrt(dot(a, a)); }
static Vec axpy(const Vec& x, double a, const Vec& d) {  // x + a d
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] + a * d[i];
    return y;
}
static Vec moins(const Vec& a, const Vec& b) { return axpy(a, -1.0, b); }

// ---------- Projecteurs sur des convexes simples ----------
Vec proj_boule(const Vec& x, const Vec& centre = {}, double rayon = 1.0) {
    Vec v = centre.empty() ? x : moins(x, centre);
    const double nv = norme(v);
    if (nv <= rayon) return x;
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = (centre.empty() ? 0.0 : centre[i]) + rayon * v[i] / nv;
    return y;
}

Vec proj_boite(const Vec& x, const Vec& lb, const Vec& ub) {
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = std::clamp(x[i], lb[i], ub[i]);
    return y;
}

// ---------- Directions du Poll (Householder, à la OrthoMADS) ----------
// H = I - 2 u u^T est orthogonale pour ||u|| = 1 ; colonne j : e_j - 2 u_j u.
Mat directions_poll(int n, std::mt19937_64& rng) {
    std::normal_distribution<double> gauss(0.0, 1.0);
    Vec u(n);
    double nu = 0.0;
    while (nu < 1e-12) {
        for (auto& ui : u) ui = gauss(rng);
        nu = norme(u);
    }
    for (auto& ui : u) ui /= nu;

    Mat D;
    D.reserve(2 * n);
    for (int j = 0; j < n; ++j) {
        Vec h(n);
        for (int i = 0; i < n; ++i) h[i] = (i == j ? 1.0 : 0.0) - 2.0 * u[j] * u[i];
        D.push_back(h);
    }
    for (int j = 0; j < n; ++j) {
        Vec m(n);
        for (int i = 0; i < n; ++i) m[i] = -D[j][i];
        D.push_back(std::move(m));
    }
    return D;
}

// ---------- Algorithme hybride ----------
struct Options {
    double delta0 = 1.0, tol = 1e-8, delta_max = 10.0;
    int max_eval = 5000;
    double theta = 0.5, gamma = 2.0, c_dec = 1e-4;
    int max_backtrack = 5;
    bool use_search = true;      // étape Search spéculative
    bool use_ordering = true;    // dernière direction gagnante en premier
    bool use_mesh_floor = true;  // backtracking jusqu'à Δm = min(Δp, Δp²)
    unsigned long long seed = 0;
};

struct Resultat {
    Vec x;
    double fx;
    int n_eval;
    std::vector<std::pair<int, double>> historique;  // (n_eval, f) à chaque succès
};

Resultat hybride_nomad_curviligne(const Fonction& f, const Projecteur& proj, const Vec& x0,
                                  const Options& o = Options()) {
    std::mt19937_64 rng(o.seed);
    Vec x = proj(x0);
    double fx = f(x);
    int n_eval = 1;
    double delta = o.delta0;  // taille de sondage Δp
    Vec d_last, pas;
    bool a_d_last = false, a_pas = false;
    std::vector<std::pair<int, double>> hist{{n_eval, fx}};

    // Évalue y ; accepte si décroissance suffisante.
    auto essai = [&](const Vec& y, double rho) -> bool {
        const double fy = f(y);
        ++n_eval;
        if (fy < fx - rho) {
            x = y;
            fx = fy;
            hist.emplace_back(n_eval, fx);
            return true;
        }
        return false;
    };

    while (delta >= o.tol && n_eval < o.max_eval) {
        const double rho = o.c_dec * delta * delta;
        const Vec x_prev = x;
        bool succes = false, par_search = false;

        // --- Search : extrapolation projetée le long du dernier pas réussi ---
        if (o.use_search && a_pas) {
            const Vec y = proj(axpy(x, o.gamma, pas));
            if (norme(moins(y, x)) > 0.0 && essai(y, rho)) succes = par_search = true;
        }

        // --- Poll : directions + arc projeté avec backtracking ---
        if (!succes) {
            Mat D = directions_poll(static_cast<int>(x.size()), rng);
            if (o.use_ordering && a_d_last) D.insert(D.begin(), d_last);
            const double delta_m = o.use_mesh_floor ? std::min(delta, delta * delta) : 0.0;

            // FSP : une taille de pas COMMUNE alpha pour tout le Poll. Si aucune direction
            // ne réussit, alpha <- theta * alpha pour TOUT le Poll (backtracking commun),
            // puis on régénère l'ensemble des points projetés.
            double alpha = delta;
            for (int k = 0; k <= o.max_backtrack && !succes; ++k, alpha *= o.theta) {
                if (n_eval >= o.max_eval || alpha < delta_m) break;
                std::set<Vec> vus;  // points projetés déjà tentés à ce niveau de alpha
                for (const Vec& d : D) {
                    const Vec y = proj(axpy(x, alpha, d));  // arc admissible
                    if (norme(moins(y, x)) == 0.0 || !vus.insert(y).second) continue;
                    if (essai(y, rho)) {
                        succes = true;
                        d_last = d;
                        a_d_last = true;
                        break;
                    }
                    if (n_eval >= o.max_eval) break;
                }
            }
        }

        // --- Mise à jour de Δp ---
        if (succes) {
            pas = moins(x, x_prev);
            a_pas = true;
            if (par_search) {
                d_last = pas;
                for (auto& v : d_last) v /= norme(pas);
                a_d_last = true;
            }
            delta = std::min(o.gamma * delta, o.delta_max);
        } else {
            a_pas = false;
            delta *= o.theta;
        }
    }
    return {x, fx, n_eval, hist};
}

// ---------- Banc d'essai ----------
struct Probleme {
    std::string nom;
    Fonction f;
    Projecteur proj;
    Vec x0;
    double f_star;
    int max_eval;
};

static int evals_pour_atteindre(const Resultat& r, double f_star, double eps) {
    for (const auto& [ne, fv] : r.historique)
        if (fv - f_star <= eps) return ne;
    return -1;
}

int main() {
    std::vector<Probleme> problemes;

    {  // 1) ||x - c||² sur la boule unité (n = 5)
        const int n = 5;
        const Vec c(n, 2.0);
        Fonction f = [c](const Vec& x) {
            double s = 0.0;
            for (size_t i = 0; i < x.size(); ++i) s += (x[i] - c[i]) * (x[i] - c[i]);
            return s;
        };
        Vec x_ref(n);
        for (int i = 0; i < n; ++i) x_ref[i] = c[i] / norme(c);
        problemes.push_back({"Boule n=5", f, [](const Vec& x) { return proj_boule(x, {}, 1.0); },
                             Vec(n, 0.0), f(x_ref), 5000});
    }
    {  // 2) quadratique mal conditionnée, minimum au coin de la boîte [-1,1]^10
        const int n = 10;
        Fonction f = [](const Vec& x) {
            double s = 0.0;
            for (size_t i = 0; i < x.size(); ++i) s += (i + 1.0) * (x[i] - 3.0) * (x[i] - 3.0);
            return s;
        };
        problemes.push_back({"Coin boite n=10", f,
                             [n](const Vec& x) { return proj_boite(x, Vec(n, -1.0), Vec(n, 1.0)); },
                             Vec(n, 0.0), 220.0, 5000});
    }
    {  // 3) Rosenbrock n = 4 sur [-2,2]^4 (minimum intérieur en (1,...,1))
        const int n = 4;
        Fonction f = [](const Vec& x) {
            double s = 0.0;
            for (size_t i = 0; i + 1 < x.size(); ++i)
                s += 100.0 * std::pow(x[i + 1] - x[i] * x[i], 2) + std::pow(1.0 - x[i], 2);
            return s;
        };
        problemes.push_back({"Rosenbrock n=4", f,
                             [n](const Vec& x) { return proj_boite(x, Vec(n, -2.0), Vec(n, 2.0)); },
                             Vec(n, -1.0), 0.0, 20000});
    }

    Options pur;  // se rapproche de la recherche directe curviligne simple
    pur.use_search = pur.use_ordering = pur.use_mesh_floor = false;
    const Options hyb;  // tout activé

    const int n_graines = 10;
    const double eps = 1e-6;
    std::printf("%-16s %-8s %14s %10s %12s\n", "Problème", "Variante", "écart final(méd)",
                "atteint", "évals(méd)");

    for (const auto& p : problemes) {
        for (int v = 0; v < 2; ++v) {
            std::vector<double> ecarts;
            std::vector<int> evals;
            int atteints = 0;
            for (int s = 0; s < n_graines; ++s) {
                Options o = v == 0 ? pur : hyb;
                o.max_eval = p.max_eval;
                o.seed = static_cast<unsigned long long>(s);
                const Resultat r = hybride_nomad_curviligne(p.f, p.proj, p.x0, o);
                ecarts.push_back(r.fx - p.f_star);
                const int e = evals_pour_atteindre(r, p.f_star, eps);
                if (e >= 0) {
                    ++atteints;
                    evals.push_back(e);
                }
            }
            std::sort(ecarts.begin(), ecarts.end());
            std::sort(evals.begin(), evals.end());
            const double ecart_med = 0.5 * (ecarts[n_graines / 2 - 1] + ecarts[n_graines / 2]);
            std::printf("%-16s %-8s %14.2e %7d/%d ", p.nom.c_str(), v == 0 ? "pur" : "hybride",
                        ecart_med, atteints, n_graines);
            if (atteints > 0)
                std::printf("%12d\n", evals[evals.size() / 2]);
            else
                std::printf("%12s\n", "-");
        }
    }
    return 0;
}
