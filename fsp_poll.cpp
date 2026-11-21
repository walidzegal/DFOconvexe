// fsp_poll.cpp
// Poll de type FSP : UNE taille de pas commune alpha pour tout le Poll.
// Si aucun point projeté ne réussit, on réduit alpha pour TOUT le Poll
// (backtracking commun), puis on régénère et réévalue l'ensemble des points.
// Contraste avec le backtracking direction par direction.
//
// Le coeur (namespace fsp) ne dépend pas de NOMAD : il reçoit
//   - les directions de Poll (celles de NOMAD : OrthoMADS, etc.),
//   - le projecteur P_C,
//   - un évaluateur par lot (à brancher sur NOMAD, voir plus bas).
// Compilation autonome du test : g++ -O2 -std=c++17 fsp_poll.cpp -o fsp_poll

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

namespace fsp {

using Vec = std::vector<double>;

struct Eval {          // résultat d'une évaluation de la boîte noire
    double f;          // objectif
    double h;          // violation agrégée des contraintes (0 = faisable)
};

using Projector = std::function<Vec(const Vec&)>;
using BatchEval = std::function<std::vector<Eval>(const std::vector<Vec>&)>;

struct Params {
    double theta    = 0.5;    // facteur de réduction commun du Poll
    double alphaMin = 1e-9;   // en dessous : échec du Poll
    double distTol  = 1e-12;  // point projeté trop proche de x : ignoré
};

struct Outcome {
    bool   success = false;
    Vec    x;                 // meilleur point trouvé (ou incumbent si échec)
    Eval   e{0, 0};
    double alphaUsed = 0;     // alpha qui a donné le succès
    int    nBacktrack = 0;    // nombre de réductions communes de alpha
    int    nEval = 0;         // évaluations réelles (hors cache)
};

// Critère de succès de type barrière progressive (dominance au sens (h, f)).
inline bool dominates(const Eval& t, const Eval& inc) {
    const bool le     = (t.h <= inc.h) && (t.f <= inc.f);
    const bool strict = (t.h <  inc.h) || (t.f <  inc.f);
    return le && strict;
}

inline double dist(const Vec& a, const Vec& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(s);
}

// Un Poll FSP complet autour de l'incumbent (x, ex).
// `cache` est partagé entre les backtrackings (et peut l'être entre itérations).
inline Outcome poll(const Vec& x, const Eval& ex,
                    const std::vector<Vec>& dirs, double alpha0,
                    const Projector& P, const BatchEval& evalBatch,
                    std::map<Vec, Eval>& cache, const Params& prm = {}) {
    Outcome out;
    out.x = x;
    out.e = ex;

    for (double a = alpha0; a >= prm.alphaMin; a *= prm.theta, ++out.nBacktrack) {
        // 1) Tous les points du Poll avec la MÊME taille a, puis projection.
        std::vector<Vec> pts;
        for (const Vec& d : dirs) {
            Vec y(x.size());
            for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] + a * d[i];
            y = P(y);
            if (dist(y, x) < prm.distTol) continue;   // projection ramène en x
            if (cache.count(y)) continue;             // déjà évalué
            if (std::find(pts.begin(), pts.end(), y) != pts.end()) continue;
            pts.push_back(std::move(y));
        }
        if (pts.empty()) continue;                    // rien de nouveau : on réduit

        // 2) Évaluation de tout le Poll.
        const std::vector<Eval> res = evalBatch(pts);
        out.nEval += static_cast<int>(pts.size());
        for (size_t k = 0; k < pts.size(); ++k) cache[pts[k]] = res[k];

        // 3) Succès si au moins un point domine l'incumbent ; on garde le meilleur.
        int best = -1;
        for (size_t k = 0; k < pts.size(); ++k) {
            const Eval& ref = (best < 0) ? ex : res[best];
            if (dominates(res[k], ref)) best = static_cast<int>(k);
        }
        if (best >= 0) {
            out.success   = true;
            out.x         = pts[best];
            out.e         = res[best];
            out.alphaUsed = a;
            return out;
        }
        // 4) Échec : on sort de la boucle for et a <- theta * a pour TOUT le Poll.
    }
    return out;  // échec complet : NOMAD réduira alors le maillage comme d'habitude
}

}  // namespace fsp

// ---------------------------------------------------------------------------
// Test autonome (sans NOMAD) : min sum (x_i - 2)^2 sur la boule de rayon 1.
// Optimum analytique : x* = (1/sqrt(n)) * (1,...,1), f* = n*(2 - 1/sqrt(n))^2.
// ---------------------------------------------------------------------------
int main() {
    using namespace fsp;
    const int n = 5;
    const double radius = 1.0;

    Projector P = [&](const Vec& y) {
        double nrm = 0;
        for (double v : y) nrm += v * v;
        nrm = std::sqrt(nrm);
        Vec z = y;
        if (nrm > radius) for (double& v : z) v *= radius / nrm;
        return z;
    };
    BatchEval ev = [&](const std::vector<Vec>& pts) {
        std::vector<Eval> r;
        for (const Vec& y : pts) {
            double f = 0;
            for (double v : y) f += (v - 2.0) * (v - 2.0);
            r.push_back({f, 0.0});   // points projetés : toujours faisables
        }
        return r;
    };

    // Directions de Poll : ±e_i (à remplacer par celles de NOMAD).
    std::vector<Vec> dirs;
    for (int i = 0; i < n; ++i) {
        Vec d(n, 0.0); d[i] = 1.0;  dirs.push_back(d);
        d[i] = -1.0;                dirs.push_back(d);
    }

    Vec x(n, 0.0);
    Eval ex = ev({x})[0];
    std::map<Vec, Eval> cache;
    double alpha = 1.0;
    int totalEval = 1, iter = 0;

    while (iter < 500) {
        Outcome o = poll(x, ex, dirs, alpha, P, ev, cache);
        totalEval += o.nEval;
        if (!o.success) break;
        x = o.x; ex = o.e;
        alpha = std::min(2.0 * o.alphaUsed, 1.0);   // agrandissement après succès
        ++iter;
    }

    const double fstar = n * std::pow(2.0 - 1.0 / std::sqrt(double(n)), 2);
    std::cout << "iterations Poll : " << iter << "\n"
              << "evaluations     : " << totalEval << "\n"
              << "f trouve        : " << ex.f << "\n"
              << "f* analytique   : " << fstar << "\n";
    return 0;
}
