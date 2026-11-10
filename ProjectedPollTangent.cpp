#include "Nomad/nomad.hpp"
#include "Algos/Mads/Mads.hpp"
#include "Algos/Mads/PollMethodBase.hpp"
#include "Math/MatrixUtils.hpp"

#include "Evaluator.hpp"
#include "Problem.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <iomanip>
#include <list>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using Vec = std::vector<double>;

// ProjectedPollTangent.cpp
// Walid Zegal
// Adaptation de ProjectedPollTangent a SOLAR1 / MAXNRG_H1.

static constexpr size_t N = 9;
static constexpr double LB[N] = {
    1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
};
static constexpr double UB[N] = {
    40.0, 40.0, 250.0, 30.0, 30.0,
    1.0e20, 89.0, 20.0, 20.0
};

static constexpr double TOL_SAME = 1e-12;
static constexpr double TOL_ACTIVE = 1e-10;
static constexpr double TOL_TANGENT = 1e-12;


// Meilleure evaluation observee sous le seuil de faisabilite choisi.
// Le calcul de h utilise la norme L2 des parties positives des 5 sorties PB.
static constexpr double H_TOL_FINAL = 1e-10;
static bool g_bestFound = false;
static double g_bestF = std::numeric_limits<double>::infinity();
static double g_bestH = std::numeric_limits<double>::infinity();
static Vec g_bestX;

// Declaration anticipee, car projectOnTangentCone l'utilise.
static bool tangentOK(const Vec& x, const Vec& d, double eps = 1e-9);

static Vec projBox(const Vec& x)
{
    Vec y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = std::max(LB[i], std::min(UB[i], x[i]));
    return y;
}

// Projection provisoire de C sur la boite SOLAR1.
static Vec projectOnC(const Vec& x)
{
    return projBox(x);
}

static bool inC(const Vec& x, double eps = 1e-9)
{
    for (size_t i = 0; i < x.size(); ++i)
        if (x[i] < LB[i] - eps || x[i] > UB[i] + eps)
            return false;
    return true;
}

static double dot(const Vec& a, const Vec& b)
{
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

static double norm2(const Vec& x) { return dot(x, x); }
static double norm(const Vec& x) { return std::sqrt(norm2(x)); }

static Vec projectBoxTangent(const Vec& x, const Vec& d)
{
    Vec z(d);
    for (size_t i = 0; i < z.size(); ++i) {
        if (x[i] >= UB[i] - TOL_ACTIVE && z[i] > 0.0) z[i] = 0.0;
        if (x[i] <= LB[i] + TOL_ACTIVE && z[i] < 0.0) z[i] = 0.0;
    }
    return z;
}

static double g2(const Vec& x)
{
    const double pi = 3.14159265358979323846;
    return pi * x[2] * x[2] * (x[8] * x[8] - x[7] * x[7])
           * x[6] / 180.0 - 1.95e6;
}

static bool c2Active(const Vec& x)
{
    return g2(x) >= -TOL_ACTIVE;
}

static Vec gradG2(const Vec& x)
{
    const double k = 3.14159265358979323846 / 180.0;
    Vec g(N, 0.0);
    g[2] = 2.0 * k * x[2] * (x[8] * x[8] - x[7] * x[7]) * x[6];
    g[6] = k * x[2] * x[2] * (x[8] * x[8] - x[7] * x[7]);
    g[7] = -2.0 * k * x[2] * x[2] * x[7] * x[6];
    g[8] =  2.0 * k * x[2] * x[2] * x[8] * x[6];
    return g;
}

static bool c3Active(const Vec& x)
{
    return (2.0 * x[0] - x[2]) >= -TOL_ACTIVE;
}

static bool c4Active(const Vec& x)
{
    return (x[7] - x[8]) >= -TOL_ACTIVE;
}

static Vec projectC2Tangent(const Vec& x, const Vec& d)
{
    Vec z(d);
    if (!c2Active(x)) return z;
    const Vec a = gradG2(x);
    const double adotd = dot(a, z);
    const double a2 = norm2(a);
    if (adotd > 0.0 && a2 > 0.0) {
        const double alpha = adotd / a2;
        for (size_t i = 0; i < N; ++i) z[i] -= alpha * a[i];
    }
    return z;
}

static Vec projectC3Tangent(const Vec& x, const Vec& d)
{
    Vec z(d);
    if (!c3Active(x)) return z;
    const double violation = 2.0 * z[0] - z[2];
    if (violation > 0.0) {
        z[0] -= 2.0 * violation / 5.0;
        z[2] += violation / 5.0;
    }
    return z;
}

static Vec projectC4Tangent(const Vec& x, const Vec& d)
{
    Vec z(d);
    if (!c4Active(x)) return z;
    const double violation = z[7] - z[8];
    if (violation > 0.0) {
        z[7] -= violation / 2.0;
        z[8] += violation / 2.0;
    }
    return z;
}

static Vec projectOnTangentCone(const Vec& x, const Vec& d)
{
    Vec z = projectBoxTangent(x, d);
    z = projectC3Tangent(x, z);
    z = projectC4Tangent(x, z);
    z = projectC2Tangent(x, z);

    if (!tangentOK(x, z)) {
        std::cerr << "[ProjectedPollTangent] ATTENTION : "
                  << "la projection sequentielle ne satisfait pas "
                  << "simultanement toutes les contraintes tangentielles."
                  << std::endl;
    }
    return z;
}

static bool tangentOK(const Vec& x, const Vec& d, double eps)
{
    for (size_t i = 0; i < d.size(); ++i) {
        if (x[i] >= UB[i] - TOL_ACTIVE && d[i] > eps) return false;
        if (x[i] <= LB[i] + TOL_ACTIVE && d[i] < -eps) return false;
    }
    if (c3Active(x) && 2.0 * d[0] - d[2] > eps) return false;
    if (c4Active(x) && d[7] - d[8] > eps) return false;
    if (c2Active(x) && dot(gradG2(x), d) > eps) return false;
    return true;
}

static void printVec(const std::string& label, const Vec& x)
{
    std::cout << label << "[";
    for (size_t i = 0; i < x.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << x[i];
    }
    std::cout << "]" << std::endl;
}

static void printDirection(const std::string& label, const NOMAD::Direction& d)
{
    std::cout << label << "[";
    for (size_t i = 0; i < d.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << d[i].todouble();
    }
    std::cout << "]" << std::endl;
}

class My_Evaluator : public NOMAD::Evaluator
{
public:
    explicit My_Evaluator(const std::shared_ptr<NOMAD::EvalParameters>& p)
        : NOMAD::Evaluator(p, NOMAD::EvalType::BB) {}
    ~My_Evaluator() override = default;
    bool eval_x(NOMAD::EvalPoint& x, const NOMAD::Double& hMax,
                bool& countEval) const override;
};

bool My_Evaluator::eval_x(NOMAD::EvalPoint& x,
                          const NOMAD::Double& hMax,
                          bool& countEval) const
{
    (void)hMax;
    double solarPoint[N];
    for (size_t i = 0; i < N; ++i) solarPoint[i] = x[i].todouble();

    std::vector<Problem> problems;
    create_problems(problems);
    const Problem* pb = find_problem(problems, "1");
    if (pb == nullptr) {
        std::cerr << "[SOLAR1] Probleme MAXNRG_H1 introuvable." << std::endl;
        countEval = false;
        return false;
    }

    std::ostringstream outputStream;
    ::Evaluator solarEvaluator(*pb, outputStream);
    solarEvaluator.set_x(solarPoint);

    const int xIndex = 0;
    const int seed = 0;
    const double fidelity = 1.0;
    const int replications = 1;
    bool simulationCompleted = false;
    bool solarCountEval = false;
    bool verbose = false;
    std::string errorMessage;

    solarEvaluator.eval_x(xIndex, seed, fidelity, replications,
                          simulationCompleted, solarCountEval,
                          errorMessage, verbose);

    if (!simulationCompleted) {
        std::cerr << "[SOLAR1] Echec de la simulation : "
                  << errorMessage << std::endl;
        countEval = solarCountEval;
        return false;
    }

    solarEvaluator.display_outputs();
    const std::string outputs = outputStream.str();
    if (outputs.empty()) {
        std::cerr << "[SOLAR1] Aucune sortie retournee par SOLAR." << std::endl;
        countEval = solarCountEval;
        return false;
    }

    x.setBBO(outputs);

    // SOLAR1 retourne : objectif f, puis 5 contraintes PB.
    // Pour une contrainte PB, seule la partie positive constitue une violation.
    std::istringstream values(outputs);
    double fValue = 0.0;
    double cValue = 0.0;
    double hSquared = 0.0;

    if (values >> fValue) {
        bool completeOutputs = true;
        for (size_t i = 0; i < 5; ++i) {
            if (!(values >> cValue)) {
                completeOutputs = false;
                break;
            }
            const double violation = std::max(0.0, cValue);
            hSquared += violation * violation;
        }

        if (completeOutputs) {
            const double hValue = std::sqrt(hSquared);

            // Retenir le plus petit objectif parmi les points dont h respecte
            // le seuil annonce H_TOL_FINAL.
            if (hValue <= H_TOL_FINAL &&
                (!g_bestFound || fValue < g_bestF)) {
                g_bestFound = true;
                g_bestF = fValue;
                g_bestH = hValue;
                g_bestX.assign(N, 0.0);
                for (size_t i = 0; i < N; ++i)
                    g_bestX[i] = x[i].todouble();
            }
        }
    }

    countEval = solarCountEval;
    return true;
}

    void initAllParams(const std::shared_ptr<NOMAD::AllParameters>& p)
{
    p->setAttributeValue("DIMENSION", N);

    NOMAD::BBInputTypeList inputTypes = {
        NOMAD::BBInputType::CONTINUOUS,
        NOMAD::BBInputType::CONTINUOUS,
        NOMAD::BBInputType::CONTINUOUS,
        NOMAD::BBInputType::CONTINUOUS,
        NOMAD::BBInputType::CONTINUOUS,
        NOMAD::BBInputType::INTEGER,
        NOMAD::BBInputType::CONTINUOUS,
        NOMAD::BBInputType::CONTINUOUS,
        NOMAD::BBInputType::CONTINUOUS
    };

    p->setAttributeValue("BB_INPUT_TYPE", inputTypes);

    p->setAttributeValue("MAX_BB_EVAL", 400 * N);
    NOMAD::Point x0(N);
    x0[0] = 8.0;   x0[1] = 8.0;   x0[2] = 150.0;
    x0[3] = 7.0;   x0[4] = 7.0;   x0[5] = 250.0;
    x0[6] = 45.0;  x0[7] = 0.5;   x0[8] = 5.0;
    p->setAttributeValue("X0", x0);

    p->setAttributeValue(
        "LOWER_BOUND",
        NOMAD::ArrayOfDouble(std::vector<double>(LB, LB + N)));
    p->setAttributeValue(
        "UPPER_BOUND",
        NOMAD::ArrayOfDouble(std::vector<double>(UB, UB + N)));

    NOMAD::BBOutputTypeList out;
    out.emplace_back(NOMAD::BBOutputType::OBJ);
    for (size_t i = 0; i < 5; ++i)
        out.emplace_back(NOMAD::BBOutputType::PB);
    p->setAttributeValue("BB_OUTPUT_TYPE", out);

    NOMAD::DirectionTypeList types = {NOMAD::DirectionType::USER_POLL};
    p->setAttributeValue("DIRECTION_TYPE", types);
    p->setAttributeValue("QUAD_MODEL_SEARCH", false);
    p->setAttributeValue("NM_SEARCH", false);
    p->setAttributeValue("DISPLAY_DEGREE", 3);
    p->setAttributeValue("DISPLAY_STATS",
                         NOMAD::ArrayOfString("bbe ( sol ) obj h"));
    p->checkAndComply();
}

bool myProjectedPollTangentCallback(const NOMAD::Step& step,
                                    std::list<NOMAD::Direction>& dirs,
                                    const size_t n)
{
    auto mads = dynamic_cast<const NOMAD::Mads*>(step.getRootAlgorithm());
    if (nullptr == mads)
        throw NOMAD::Exception(__FILE__, __LINE__, "No Mads available.");

    auto callingPoll = dynamic_cast<const NOMAD::PollMethodBase*>(&step);
    if (nullptr == callingPoll)
        throw NOMAD::Exception(__FILE__, __LINE__, "No poll method available.");

    auto frameCenter = callingPoll->getFrameCenter();
    if (nullptr == frameCenter)
        throw NOMAD::Exception(__FILE__, __LINE__, "No frame center available.");
    if (n != N)
        throw NOMAD::Exception(__FILE__, __LINE__, "Dimension pb.");

    auto mesh = mads->getMegaIterationBarrier()->getCurrentIncumbentFeas()
                    ? callingPoll->getIterationMesh()
                    : callingPoll->getIterationMesh();
    if (nullptr == mesh)
        throw NOMAD::Exception(__FILE__, __LINE__, "No mesh available.");

    Vec center(n), delta(n);
    const NOMAD::ArrayOfDouble deltaFrame = mesh->getDeltaFrameSize();
    for (size_t i = 0; i < n; ++i) {
        center[i] = (*frameCenter)[i].todouble();
        delta[i] = deltaFrame[i].todouble();
    }

    std::cout << "[TANGENT POLL] Callback appele" << std::endl;
    printVec("[TANGENT POLL] Frame center = ", center);
    printVec("[TANGENT POLL] Delta frame = ", delta);

    double** M = new double*[n];
    double** Q = new double*[n];
    double** R = new double*[n];
    for (size_t i = 0; i < n; ++i) {
        M[i] = new double[n]; Q[i] = new double[n]; R[i] = new double[n];
        for (size_t j = 0; j < n; ++j) M[i][j] = (i == j ? 1.0 : 0.0);
    }

    std::string errorMsg;
    const bool qrOK = NOMAD::qr_factorization(errorMsg, M, Q, R,
                                               static_cast<int>(n),
                                               static_cast<int>(n));
    if (!qrOK) {
        for (size_t i = 0; i < n; ++i) { delete[] M[i]; delete[] Q[i]; delete[] R[i]; }
        delete[] M; delete[] Q; delete[] R;
        std::cerr << "QR failure: " << errorMsg << std::endl;
        return false;
    }

    std::vector<Vec> accepted;
    size_t rejectedZero = 0, rejectedCenter = 0, rejectedDup = 0, candidates = 0;
    NOMAD::Direction dir(n);

    for (size_t i = 0; i < n; ++i) {
        for (int sign = 1; sign >= -1; sign -= 2) {
            ++candidates;
            Vec raw(n);
            for (size_t j = 0; j < n; ++j) {
                dir[j] = sign * Q[j][i];
                raw[j] = dir[j].todouble();
            }

            Vec dTan = projectOnTangentCone(center, raw);
            const double dTanNorm = norm(dTan);
            if (dTanNorm <= TOL_TANGENT) { ++rejectedZero; continue; }
            for (size_t j = 0; j < n; ++j) dTan[j] /= dTanNorm;

            printDirection(" Direction QR = ", dir);
            printVec(" Direction tangentielle = ", dTan);
            std::cout << " Direction dans T_C = "
                      << (tangentOK(center, dTan) ? "oui" : "NON")
                      << std::endl;

            Vec trial(n);
            for (size_t j = 0; j < n; ++j)
                trial[j] = center[j] + delta[j] * dTan[j];
            Vec projected = projectOnC(trial);

            double dc = 0.0;
            for (size_t j = 0; j < n; ++j) dc += std::abs(projected[j] - center[j]);
            if (dc < TOL_SAME) { ++rejectedCenter; continue; }

            bool duplicate = false;
            for (const auto& a : accepted) {
                double dd = 0.0;
                for (size_t j = 0; j < n; ++j) dd += std::abs(projected[j] - a[j]);
                if (dd < TOL_SAME) { duplicate = true; break; }
            }
            if (duplicate) { ++rejectedDup; continue; }
            NOMAD::Direction outDir(n);
            bool zeroDelta = false;

            for (size_t j = 0; j < n; ++j) {
                if (std::abs(delta[j]) <= 1e-300) {
                    zeroDelta = true;
                    break;
                }

                outDir[j] = (projected[j] - center[j]) / delta[j];
            }

            if (zeroDelta)
                continue;

            // Rejeter les directions finales quasi nulles avant envoi a NOMAD
            double normInf = 0.0;

            for (size_t j = 0; j < n; ++j) {
                normInf = std::max(
                    normInf,
                    std::abs(outDir[j].todouble())
                );
            }

            if (normInf <= 1e-12) {
                ++rejectedZero;

                std::cout
                    << "[TANGENT POLL] Direction rejetee : "
                    << "norme infinie presque nulle = "
                    << normInf
                    << "\n";

                continue;
            }

            accepted.push_back(projected);
            dirs.push_back(outDir);

            printVec(" Point final projete = ", projected);
            printDirection(" Direction envoyee a NOMAD = ", outDir);
            dirs.push_back(outDir);
            printVec(" Point final projete = ", projected);
            printDirection(" Direction envoyee a NOMAD = ", outDir);
        }
    }

    for (size_t i = 0; i < n; ++i) { delete[] M[i]; delete[] Q[i]; delete[] R[i]; }
    delete[] M; delete[] Q; delete[] R;

    std::cout << "[TANGENT POLL] Candidats QR = " << candidates << "\n"
              << "[TANGENT POLL] Rejets direction nulle = " << rejectedZero << "\n"
              << "[TANGENT POLL] Rejets au centre = " << rejectedCenter << "\n"
              << "[TANGENT POLL] Rejets doublons = " << rejectedDup << "\n"
              << "[TANGENT POLL] Directions generees = " << dirs.size()
              << std::endl;
    return true;
}

int main()
{
    NOMAD::MainStep mainStep;
    auto params = std::make_shared<NOMAD::AllParameters>();
    initAllParams(params);
    mainStep.setAllParameters(params);

    std::unique_ptr<My_Evaluator> evaluator(
        new My_Evaluator(params->getEvalParams()));
    mainStep.setEvaluator(std::move(evaluator));

    mainStep.start();
    auto mads = std::dynamic_pointer_cast<NOMAD::Mads>(
        mainStep.getAlgo(NOMAD::StepType::ALGORITHM_MADS));
    if (nullptr == mads)
        throw NOMAD::Exception(__FILE__, __LINE__,
                               "Cannot access to Mads algorithm");

    mads->addCallback(NOMAD::CallbackType::USER_METHOD_POLL,
                      myProjectedPollTangentCallback);
    mainStep.run();
    mainStep.end();

    std::cout << "\n=============================================\n";
    std::cout << "       MEILLEURE SOLUTION TROUVEE\n";
    std::cout << "       (h <= " << H_TOL_FINAL << ")\n";
    std::cout << "=============================================\n";

    if (g_bestFound) {
        std::cout << std::setprecision(17);
        std::cout << "x = (";
        for (size_t i = 0; i < g_bestX.size(); ++i) {
            if (i) std::cout << " ";
            std::cout << g_bestX[i];
        }
        std::cout << ")\n";
        std::cout << "f = " << g_bestF << "\n";
        std::cout << "h = " << g_bestH << "\n";
    }
    else {
        std::cout << "Aucune evaluation avec h <= "
                  << H_TOL_FINAL << " n'a ete trouvee.\n";
    }

    std::cout << "=============================================\n";
    return 0;
}
