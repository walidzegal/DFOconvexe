// Hybride MADS/NOMAD + recherche directe projetée curviligne, appliqué à SOLAR1 (MAXNRG_H1).
// Version FSP : le Poll utilise une taille de pas commune et un backtracking commun.
//
// SOLAR1 : n = 9 (x6 entier), bornes seules comme contraintes connues (convexe = boîte),
//          5 contraintes boîte noire c_j <= 0 (traitées par une barrière progressive simplifiée).
// Sortie du simulateur : f c1 c2 c3 c4 c5.
//
// Utilisation : ./hybride_solar1 /chemin/vers/solar [seed=0] [max_eval=3600] [hybride|pur]
// Compilation : g++ -O2 -std=c++17 hybride_solar1.cpp -o hybride_solar1

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;
static const double INF = std::numeric_limits<double>::infinity();

// ======================= Problème SOLAR1 =======================
struct Solar1 {
    std::string exe;
    int n = 9;
    int idx_int = 5;  // x6 entier
    Vec lb{1, 1, 20, 1, 1, 1, 1, 0, 0};
    Vec ub{40, 40, 250, 30, 30, INF, 89, 20, 20};
    Vec x0{8, 8, 150, 7, 7, 250, 45, 0.5, 5};
    Vec echelle;  // échelle de normalisation par variable

    Solar1(std::string e) : exe(std::move(e)) {
        echelle.resize(n);
        for (int i = 0; i < n; ++i)
            echelle[i] = std::isfinite(ub[i]) ? ub[i] - lb[i] : std::max(1.0, std::fabs(x0[i]));
    }

    // Projection sur la boîte puis arrondi de la variable entière.
    Vec proj(Vec x) const {
        for (int i = 0; i < n; ++i) x[i] = std::clamp(x[i], lb[i], ub[i]);
        x[idx_int] = std::round(x[idx_int]);
        return x;
    }
};

struct Eval {
    double f = INF, h = INF;
    bool ok = false;
};

class Evaluateur {
public:
    Evaluateur(const Solar1& p) : p_(p) {
        fichier_ = (std::filesystem::temp_directory_path() / ("x_solar_" + std::to_string(getpid()) + ".txt")).string();
    }
    ~Evaluateur() { std::remove(fichier_.c_str()); }
    int n_eval() const { return n_eval_; }

    Eval operator()(const Vec& x) {
        auto it = cache_.find(x);
        if (it != cache_.end()) return it->second;  // un point en cache ne coûte pas de BBE
        {
            std::ofstream out(fichier_);
            out.precision(17);
            for (size_t i = 0; i < x.size(); ++i) out << (i ? " " : "") << x[i];
            out << "\n";
        }
        ++n_eval_;
        Eval e;
        const std::string cmd = "\"" + p_.exe + "\" 1 \"" + fichier_ + "\" -seed=0 2>/dev/null";
        if (FILE* pipe = popen(cmd.c_str(), "r")) {
            std::string sortie;
            char buf[512];
            while (std::fgets(buf, sizeof buf, pipe)) sortie += buf;
            pclose(pipe);
            Vec val;
            const char* s = sortie.c_str();
            char* fin = nullptr;
            for (;;) {
                const double v = std::strtod(s, &fin);
                if (fin == s) break;
                val.push_back(v);
                s = fin;
            }
            if (val.size() >= 6 && std::isfinite(val[0])) {
                e.ok = true;
                e.f = val[0];
                double h2 = 0.0;
                for (int j = 1; j <= 5; ++j) {
                    if (!std::isfinite(val[j])) { e.ok = false; break; }
                    if (val[j] > 0) h2 += val[j] * val[j];
                }
                e.h = std::sqrt(h2);
            }
        }
        if (!e.ok) { e.f = INF; e.h = INF; }
        cache_[x] = e;
        return e;
    }

private:
    const Solar1& p_;
    std::string fichier_;
    std::map<Vec, Eval> cache_;
    int n_eval_ = 0;
};

// ======================= Directions du Poll (Householder) =======================
Mat directions_poll(int n, std::mt19937_64& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    Vec u(n);
    double nu = 0.0;
    while (nu < 1e-12) {
        for (auto& v : u) v = g(rng);
        nu = 0.0;
        for (double v : u) nu += v * v;
        nu = std::sqrt(nu);
    }
    for (auto& v : u) v /= nu;
    Mat D;
    for (int j = 0; j < n; ++j) {
        Vec h(n);
        for (int i = 0; i < n; ++i) h[i] = (i == j ? 1.0 : 0.0) - 2.0 * u[j] * u[i];
        D.push_back(h);
    }
    for (int j = 0; j < n; ++j) {
        Vec m(n);
        for (int i = 0; i < n; ++i) m[i] = -D[j][i];
        D.push_back(m);
    }
    return D;
}

// ======================= Algorithme hybride =======================
struct Options {
    double delta0 = 0.1;  // en unités normalisées (fraction de l'étendue)
    double delta_max = 1.0, tol = 1e-7;
    int max_eval = 3600;
    double theta = 0.5, gamma = 2.0, c_dec = 1e-4;
    int max_backtrack = 5;
    bool use_search = true, use_ordering = true, use_mesh_floor = true;
    unsigned long long seed = 0;
};

struct Pt {
    Vec x;
    double f = INF, h = INF;
    bool valide = false;
};

struct Resultat {
    Pt best;  // meilleur point faisable (ou meilleur infaisable si aucun)
    int n_eval = 0;
    int bbe_premier_faisable = -1;
    std::vector<std::pair<int, double>> historique;  // (bbe, meilleur f faisable)
};

Resultat hybride_solar(const Solar1& pb, const Options& o) {
    Evaluateur eval(pb);
    std::mt19937_64 rng(o.seed);
    const int n = pb.n;

    Pt xF, xI;  // incumbents faisable / infaisable (barrière progressive)
    double hmax = INF;
    std::vector<double> h_infaisables;

    Resultat res;
    auto note = [&]() {
        if (xF.valide) res.historique.emplace_back(eval.n_eval(), xF.f);
    };

    // Point initial
    {
        Vec x = pb.proj(pb.x0);
        Eval e = eval(x);
        Pt p{x, e.f, e.h, e.ok};
        if (e.ok && e.h == 0.0) { xF = p; res.bbe_premier_faisable = eval.n_eval(); }
        else if (e.ok) { xI = p; }
        note();
    }

    enum Sortie { AUCUN, PLEIN, AMELIORE };
    // Évalue y et met à jour les incumbents selon les règles de la barrière progressive.
    auto essai = [&](const Vec& y, double rho) -> Sortie {
        const Eval e = eval(y);
        if (!e.ok) return AUCUN;
        if (e.h == 0.0) {
            if (!xF.valide || e.f < xF.f - rho) {
                xF = {y, e.f, 0.0, true};
                if (res.bbe_premier_faisable < 0) res.bbe_premier_faisable = eval.n_eval();
                note();
                return PLEIN;
            }
            return AUCUN;
        }
        if (e.h > hmax) return AUCUN;
        h_infaisables.push_back(e.h);
        if (!xI.valide) { xI = {y, e.f, e.h, true}; return PLEIN; }
        if (e.f < xI.f && e.h <= xI.h) { xI = {y, e.f, e.h, true}; return PLEIN; }     // domine
        if (e.h < xI.h && e.f <= xI.f) { xI = {y, e.f, e.h, true}; return PLEIN; }
        if (e.h < xI.h) { xI = {y, e.f, e.h, true}; return AMELIORE; }               // h améliore, f se dégrade
        return AUCUN;
    };

    // Point candidat sur l'arc admissible : y = Proj(x + alpha * E d), pas minimal de 1 pour x6.
    auto candidat = [&](const Vec& x, double alpha, const Vec& d) {
        Vec y = x;
        for (int i = 0; i < n; ++i) {
            double dx = alpha * pb.echelle[i] * d[i];
            if (i == pb.idx_int && dx != 0.0)
                dx = std::copysign(std::max(1.0, std::round(std::fabs(dx))), dx);
            y[i] += dx;
        }
        return pb.proj(y);
    };

    double delta = o.delta0;
    Vec dernier_pas, d_last;
    bool a_pas = false, a_dlast = false;

    while (delta >= o.tol && eval.n_eval() < o.max_eval) {
        std::vector<Pt> centres;
        if (xF.valide) centres.push_back(xF);
        if (xI.valide) centres.push_back(xI);
        if (centres.empty()) break;  // aucun point évaluable

        Sortie sortie = AUCUN;
        const Vec x_avant = centres[0].x;
        const double rho = o.c_dec * delta * delta * std::max(1.0, std::fabs(centres[0].f));

        // --- Search : extrapolation projetée le long du dernier pas réussi ---
        if (o.use_search && a_pas) {
            Vec z = centres[0].x;
            for (int i = 0; i < n; ++i) z[i] += o.gamma * dernier_pas[i];
            const Vec y = pb.proj(z);
            if (y != centres[0].x) {
                sortie = essai(y, rho);
                if (sortie != AUCUN) { dernier_pas = Vec(n); for (int i = 0; i < n; ++i) dernier_pas[i] = y[i] - x_avant[i]; }
            }
        }

        // --- Poll FSP : taille de pas COMMUNE à tout le Poll ---
        // On essaie toutes les directions avec le même alpha ; si aucune ne réussit,
        // alpha <- theta * alpha pour TOUT le Poll (backtracking commun), puis on
        // régénère l'ensemble des points projetés. (Avant : backtracking par direction.)
        const double delta_m = o.use_mesh_floor ? std::min(delta, delta * delta) : 0.0;
        for (size_t c = 0; c < centres.size() && sortie == AUCUN; ++c) {
            const Vec xc = centres[c].x;
            Mat D = directions_poll(n, rng);
            if (o.use_ordering && a_dlast) D.insert(D.begin(), d_last);
            double alpha = delta;
            for (int k = 0; k <= o.max_backtrack && sortie == AUCUN; ++k, alpha *= o.theta) {
                if (eval.n_eval() >= o.max_eval || alpha < delta_m) break;
                std::set<Vec> vus;  // points projetés déjà tentés à ce niveau de alpha
                for (const Vec& d : D) {
                    const Vec y = candidat(xc, alpha, d);
                    if (y == xc || !vus.insert(y).second) continue;
                    sortie = essai(y, rho);
                    if (sortie != AUCUN) {
                        d_last = d; a_dlast = true;
                        dernier_pas = Vec(n);
                        for (int i = 0; i < n; ++i) dernier_pas[i] = y[i] - xc[i];
                        a_pas = true;
                        break;
                    }
                    if (eval.n_eval() >= o.max_eval) break;
                }
            }
        }

        // --- Mise à jour de delta et de h_max ---
        if (sortie == PLEIN) {
            delta = std::min(o.gamma * delta, o.delta_max);
        } else if (sortie == AMELIORE) {
            // delta inchangé
        } else {
            a_pas = false;
            delta *= o.theta;
            if (xI.valide) {  // h_max descend au plus grand h infaisable en dessous
                double nouveau = -INF;
                for (double h : h_infaisables)
                    if (h < hmax && h >= xI.h) nouveau = std::max(nouveau, h);
                if (nouveau > -INF) hmax = nouveau;
            }
        }
    }

    res.best = xF.valide ? xF : xI;
    res.n_eval = eval.n_eval();
    return res;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage : %s /chemin/vers/solar [seed=0] [max_eval=3600] [hybride|pur]\n", argv[0]);
        return 1;
    }
    Solar1 pb(argv[1]);
    Options o;
    if (argc > 2) o.seed = std::strtoull(argv[2], nullptr, 10);
    if (argc > 3) o.max_eval = std::atoi(argv[3]);
    const std::string mode = argc > 4 ? argv[4] : "hybride";
    if (mode == "pur") o.use_search = o.use_ordering = o.use_mesh_floor = false;

    const Resultat r = hybride_solar(pb, o);

    std::printf("mode=%s seed=%llu\n", mode.c_str(), o.seed);
    std::printf("faisable : %s | f = %.6f | h = %.3e | BBE = %d | BBE 1er faisable = %d\n",
                r.best.h == 0.0 ? "oui" : "non", r.best.f, r.best.h, r.n_eval, r.bbe_premier_faisable);
    std::printf("x = [");
    for (double v : r.best.x) std::printf(" %.10g", v);
    std::printf(" ]\n");
    std::printf("f_best(BBE) :");
    for (int b : {500, 1000, 1500, 2000, 2500, 3000, 3600}) {
        double fb = INF;
        for (const auto& [ne, fv] : r.historique) if (ne <= b) fb = std::min(fb, fv);
        std::printf(" %d:%.1f", b, fb);
    }
    std::printf("\n");
    return 0;
}