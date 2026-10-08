/*---------------------------------------------------------------------------------*/
/*  Recherche directe projetée avec arcs admissibles dans NOMAD 4                  */
/*                                                                                 */
/*  Basé sur l'exemple officiel examples/advanced/library/CustomPollMethod         */
/*  (NOMAD 4, LGPL v3+, GERAD / Polytechnique Montréal).                           */
/*                                                                                 */
/*  Idée : le callback USER_METHOD_POLL remplace le Poll standard. Pour chaque     */
/*  direction q_i d'une base orthonormale aléatoire (±q_i) et pour chaque pas      */
/*  de backtracking alpha_j = theta^j, on construit l'arc admissible               */
/*        x(alpha_j) = Pi_C( x_k + alpha_j * Delta o q_i )                         */
/*  et on renvoie à NOMAD la direction effective                                   */
/*        d_ij = ( x(alpha_j) - x_k ) ./ Delta                                     */
/*  (division composante par composante par la taille de cadre Delta^f).           */
/*                                                                                 */
/*  Problème test : min ||x - c||^2  sur la boule  C = { ||x|| <= R }              */
/*  Solution exacte : x* = R c / ||c||.                                            */
/*---------------------------------------------------------------------------------*/
#include "Nomad/nomad.hpp"
#include "Algos/EvcInterface.hpp"
#include "Algos/Mads/Mads.hpp"
#include "Algos/Mads/MadsMegaIteration.hpp"
#include "Algos/Mads/SearchMethodAlgo.hpp"
#include "Algos/SubproblemManager.hpp"
#include "Cache/CacheBase.hpp"
#include "Type/EvalSortType.hpp"
#include "Algos/AlgoStopReasons.hpp"
#include "Util/AllStopReasons.hpp"
#include "Math/MatrixUtils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

/*----------------------------------------*/
/*               Le problème              */
/*----------------------------------------*/
const size_t N = 5;
const double RAYON = 1.0;       // C = boule de rayon RAYON centrée à l'origine
const double CIBLE = 2.0;       // c = (CIBLE, ..., CIBLE)

// Paramètres du backtracking curviligne
const double THETA = 0.5;       // facteur de réduction de alpha
const int    MAX_BACKTRACK = 3; // alpha_j = THETA^j, j = 0..MAX_BACKTRACK

/*----------------------------------------*/
/*     Projecteur sur le convexe C        */
/*----------------------------------------*/
// Remplacer ce corps pour un autre convexe (boîte, polyèdre, SOC, ...).
std::vector<double> projeter(const std::vector<double>& x)
{
    double nrm = 0.0;
    for (double xi : x) nrm += xi * xi;
    nrm = std::sqrt(nrm);

    if (nrm <= RAYON) return x;

    std::vector<double> p(x.size());
    for (size_t i = 0; i < x.size(); ++i) p[i] = RAYON * x[i] / nrm;
    return p;
}

/*----------------------------------------*/
/*              Évaluateur                */
/*----------------------------------------*/
// Sorties : f(x) = ||x - c||^2 (OBJ) et c(x) = ||x||^2 - R^2 <= 0 (EB).
// La contrainte EB rejette les points hors de C qui pourraient apparaître
// après l'arrondi sur le maillage : on n'évalue PAS f o Pi_C, ce qui évite
// les plateaux artificiels observés avec l'enveloppe de projection.
class My_Evaluator : public NOMAD::Evaluator
{
public:
    explicit My_Evaluator(const std::shared_ptr<NOMAD::EvalParameters>& evalParams)
    : NOMAD::Evaluator(evalParams, NOMAD::EvalType::BB)
    {}

    ~My_Evaluator() override = default;

    bool eval_x(NOMAD::EvalPoint& x, const NOMAD::Double& hMax, bool& countEval) const override;
};

bool My_Evaluator::eval_x(NOMAD::EvalPoint& x,
                          const NOMAD::Double& hMax,
                          bool& countEval) const
{
    double f = 0.0, nrm2 = 0.0;
    for (size_t i = 0; i < N; ++i)
    {
        const double xi = x[i].todouble();
        f    += (xi - CIBLE) * (xi - CIBLE);
        nrm2 += xi * xi;
    }
    const double c = nrm2 - RAYON * RAYON;

    NOMAD::Double F(f), C(c);
    x.setBBO(F.tostring() + " " + C.tostring());
    countEval = true;
    return true;
}

/*----------------------------------------*/
/*              Paramètres                */
/*----------------------------------------*/
void initAllParams(const std::shared_ptr<NOMAD::AllParameters>& allParams)
{
    allParams->setAttributeValue("DIMENSION", N);
    allParams->setAttributeValue("MAX_BB_EVAL", 400 * N);

    // Point de départ faisable (dans C)
    allParams->setAttributeValue("X0", NOMAD::Point(N, 0.0));

    allParams->setAttributeValue("LOWER_BOUND", NOMAD::ArrayOfDouble(N, -10.0));
    allParams->setAttributeValue("UPPER_BOUND", NOMAD::ArrayOfDouble(N, 10.0));

    // Sorties : objectif + contrainte extrême (barrière)
    NOMAD::BBOutputTypeList bbOutputTypes;
    bbOutputTypes.emplace_back(NOMAD::BBOutputType::OBJ);
    bbOutputTypes.emplace_back(NOMAD::BBOutputType::EB);
    allParams->setAttributeValue("BB_OUTPUT_TYPE", bbOutputTypes);

    // Poll entièrement remplacé par le poll projeté.
    // On peut ajouter NOMAD::DirectionType::ORTHO_2N à la liste pour combiner.
    NOMAD::DirectionTypeList dtList = { NOMAD::DirectionType::USER_POLL };
    allParams->setAttributeValue("DIRECTION_TYPE", dtList);

    // Désactiver Search pour isoler l'effet du Poll
    allParams->setAttributeValue("QUAD_MODEL_SEARCH", false);
    allParams->setAttributeValue("NM_SEARCH", false);

    allParams->setAttributeValue("DISPLAY_DEGREE", 2);
    allParams->setAttributeValue("DISPLAY_STATS", NOMAD::ArrayOfString("bbe ( sol ) obj"));

    allParams->checkAndComply();
}

/*----------------------------------------*/
/*   Callback : Poll projeté curviligne   */
/*----------------------------------------*/
bool myProjectedPollCallback(const NOMAD::Step& step,
                             std::list<NOMAD::Direction>& dirs,
                             const size_t n)
{
    auto mads = dynamic_cast<const NOMAD::Mads*>(step.getRootAlgorithm());
    if (nullptr == mads)
        throw NOMAD::Exception(__FILE__, __LINE__, "No Mads available.");

    auto callingPoll = dynamic_cast<const NOMAD::PollMethodBase*>(&step);
    if (nullptr == callingPoll)
        throw NOMAD::Exception(__FILE__, __LINE__, "No poll method available.");

    // Centre du cadre x_k
    auto frameCenter = callingPoll->getFrameCenter();
    if (nullptr == frameCenter)
        throw NOMAD::Exception(__FILE__, __LINE__, "No frame center available.");

    auto pbParams = mads->getPbParams();
    if (pbParams->getAttributeValue<size_t>("DIMENSION") != n)
        throw NOMAD::Exception(__FILE__, __LINE__, "Dimension pb.");

    // Taille de cadre Delta^f (une valeur par coordonnée)
    auto mesh = step.getIterationMesh();
    if (nullptr == mesh)
        throw NOMAD::Exception(__FILE__, __LINE__, "No mesh available.");
    NOMAD::ArrayOfDouble boxSize = mesh->getDeltaFrameSize();

    std::vector<double> xk(n), delta(n);
    for (size_t i = 0; i < n; ++i)
    {
        xk[i]    = (*frameCenter)[i].todouble();
        delta[i] = boxSize[i].todouble();
    }

    // Base orthonormale aléatoire par QR (comme dans l'exemple officiel)
    NOMAD::Direction dirUnit(n, 0.0);
    NOMAD::Direction::computeDirOnUnitSphere(dirUnit);
    while (dirUnit[0] == 0)
        NOMAD::Direction::computeDirOnUnitSphere(dirUnit);

    auto** M = new double*[n];
    auto** Q = new double*[n];
    auto** R = new double*[n];
    for (size_t i = 0; i < n; ++i)
    {
        M[i] = new double[n];
        Q[i] = new double[n];
        R[i] = new double[n];
        M[i][0] = dirUnit[i].todouble();
        for (size_t j = 1; j < n; ++j) M[i][j] = (i == j) ? 1.0 : 0.0;
    }

    std::string error_msg;
    const bool success = NOMAD::qr_factorization(error_msg, M, Q, R,
                                                 static_cast<int>(n),
                                                 static_cast<int>(n));
    if (!success || !error_msg.empty())
    {
        std::cerr << "QR decomposition for projected poll has failed" << std::endl;
        for (size_t i = 0; i < n; ++i) { delete[] M[i]; delete[] Q[i]; delete[] R[i]; }
        delete[] M; delete[] Q; delete[] R;
        return false;
    }

    dirs.clear();

    // Pour chaque ±q_i et chaque alpha_j = THETA^j : arc projeté
    for (size_t i = 0; i < n; ++i)
    {
        for (int signe = 1; signe >= -1; signe -= 2)
        {
            double alpha = 1.0;
            for (int j = 0; j <= MAX_BACKTRACK; ++j, alpha *= THETA)
            {
                std::vector<double> y(n);
                for (size_t k = 0; k < n; ++k)
                    y[k] = xk[k] + alpha * delta[k] * signe * Q[k][i];

                const std::vector<double> p = projeter(y);

                // Direction effective d = (p - x_k) ./ Delta
                NOMAD::Direction d(n);
                double norme2 = 0.0;
                for (size_t k = 0; k < n; ++k)
                {
                    const double dk = (p[k] - xk[k]) / delta[k];
                    d[k] = dk;
                    norme2 += dk * dk;
                }

                // Ignorer les directions nulles (x_k au bord, q_i vers l'extérieur)
                if (norme2 > 1e-24)
                    dirs.push_back(d);
            }
        }
    }

    for (size_t i = 0; i < n; ++i) { delete[] M[i]; delete[] Q[i]; delete[] R[i]; }
    delete[] M; delete[] Q; delete[] R;

    return true;
}

/*----------------------------------------*/
/*                 main                   */
/*----------------------------------------*/
int main()
{
    NOMAD::MainStep TheMainStep;

    auto params = std::make_shared<NOMAD::AllParameters>();
    initAllParams(params);
    TheMainStep.setAllParameters(params);

    std::unique_ptr<My_Evaluator> ev(new My_Evaluator(params->getEvalParams()));
    TheMainStep.setEvaluator(std::move(ev));

    // Enregistrer le Poll projeté (nécessite DIRECTION_TYPE USER_POLL)
    TheMainStep.addCallback<NOMAD::MadsCallbackType::USER_METHOD_POLL>(myProjectedPollCallback);

    TheMainStep.start();
    TheMainStep.run();
    TheMainStep.end();

    // Solution exacte pour comparaison
    const double nc = CIBLE * std::sqrt(static_cast<double>(N));
    const double xs = RAYON * CIBLE / nc;
    double fstar = 0.0;
    for (size_t i = 0; i < N; ++i) fstar += (xs - CIBLE) * (xs - CIBLE);
    std::cout << "\nSolution exacte : f* = " << fstar << std::endl;

    return 0;
}
