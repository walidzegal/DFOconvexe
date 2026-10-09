/*--------------------------------------------------------------------------*/
/*  ProjectedPollMethod : derive de CustomPollMethod (NOMAD 4.5.0)          */
/*  Poll 2n (QR) dont les points d'essai sont projetes sur un ensemble      */
/*  convexe C = boite [LB,UB] inter boule(centre BALL_C, rayon BALL_R).     */
/*  Licence NOMAD (LGPL v3) : voir l'en-tete de l'exemple CustomPollMethod. */
/*--------------------------------------------------------------------------*/
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

#include <vector>
#include <cmath>
#include <algorithm>

/*----------------------------------------*/
/*               The problem              */
/*----------------------------------------*/
const int N = 6;

// Etape 1 : mettre false pour retrouver exactement le poll de customPollMethod.
// Etape 2 : true pour activer la projection.
static const bool USE_PROJECTION = true;

// Ensemble convexe : boite + boule
static const double LB = -10.0;
static const double UB = 10.0;
static const double BALL_R = 2.0;           // centre = origine
static const double TOL_SAME = 1e-12;

using Vec = std::vector<double>;

/*----------------------------------------*/
/*       Projection sur l'ensemble C      */
/*----------------------------------------*/
static Vec projBox(const Vec& x)
{
    Vec y(x);
    for (auto& v : y) v = std::min(std::max(v, LB), UB);
    return y;
}

static Vec projBall(const Vec& x)
{
    double nrm = 0.0;
    for (double v : x) nrm += v * v;
    nrm = std::sqrt(nrm);
    if (nrm <= BALL_R) return x;
    Vec y(x);
    for (auto& v : y) v *= BALL_R / nrm;
    return y;
}

// Projection sur l'intersection boite inter boule par l'algorithme de Dykstra
static Vec projectOnC(const Vec& x0, int maxIter = 500, double tol = 1e-13)
{
    const size_t n = x0.size();
    Vec x(x0), p(n, 0.0), q(n, 0.0);
    for (int k = 0; k < maxIter; ++k)
    {
        Vec xo(x), y(n), z(n);
        for (size_t i = 0; i < n; ++i) y[i] = x[i] + p[i];
        Vec xb = projBox(y);
        for (size_t i = 0; i < n; ++i) { p[i] = y[i] - xb[i]; z[i] = xb[i] + q[i]; }
        x = projBall(z);
        double d = 0.0;
        for (size_t i = 0; i < n; ++i) { q[i] = z[i] - x[i]; d += std::abs(x[i] - xo[i]); }
        if (d < tol) break;
    }
    return x;
}

static bool inC(const Vec& x, double eps = 1e-9)
{
    double nrm = 0.0;
    for (double v : x)
    {
        if (v < LB - eps || v > UB + eps) return false;
        nrm += v * v;
    }
    return std::sqrt(nrm) <= BALL_R + eps;
}

/*----------------------------------------*/
/*               Evaluator                */
/*----------------------------------------*/
class My_Evaluator : public NOMAD::Evaluator
{
public:
    explicit My_Evaluator(const std::shared_ptr<NOMAD::EvalParameters>& evalParams)
    : NOMAD::Evaluator(evalParams, NOMAD::EvalType::BB)
    {}

    ~My_Evaluator() override = default;

    bool eval_x(NOMAD::EvalPoint &x, const NOMAD::Double &hMax, bool &countEval) const override;
};

bool My_Evaluator::eval_x(NOMAD::EvalPoint &x,
                          const NOMAD::Double &hMax,
                          bool &countEval) const
{
    if (N % 2 != 0)
    {
        throw NOMAD::Exception(__FILE__,__LINE__,"Dimension N should be an even number");
    }

    // Securite : NOMAD ne connait pas la boule, on refuse les points hors de C
    if (USE_PROJECTION)
    {
        Vec v(N);
        for (size_t i = 0; i < N; ++i) v[i] = x[i].todouble();
        if (!inC(v))
        {
            countEval = false;
            return false;
        }
    }

    double f = 0;
    for ( size_t i = 1 ; i <= N/2 ; ++i ) {
        f += pow ( 10 * (x[2*i-1].todouble() - pow(x[2*i-2].todouble(),2) ) , 2 );
        f += pow ( 1 - x[2*i-2].todouble() , 2 );
    }
    NOMAD::Double F(f);
    x.setBBO(F.tostring());
    countEval = true;

    return true;
}

void initAllParams(const std::shared_ptr<NOMAD::AllParameters>& allParams)
{
    allParams->setAttributeValue("DIMENSION", N);
    allParams->setAttributeValue("MAX_BB_EVAL", 400*N);
    allParams->setAttributeValue("X0", NOMAD::Point(N, 0.0) );

    allParams->setAttributeValue("LOWER_BOUND", NOMAD::ArrayOfDouble(N, LB));
    allParams->setAttributeValue("UPPER_BOUND", NOMAD::ArrayOfDouble(N, UB));

    NOMAD::BBOutputTypeList bbOutputTypes;
    bbOutputTypes.emplace_back(NOMAD::BBOutputType::OBJ);
    allParams->setAttributeValue("BB_OUTPUT_TYPE", bbOutputTypes);

    // Etape 1 (identique a customPollMethod) : USER_POLL + ORTHO_2N.
    // Etape 2 (projection) : USER_POLL seul, sinon ORTHO_2N genere des points hors de C.
    NOMAD::DirectionTypeList dtList;
    if (USE_PROJECTION)
        dtList = {NOMAD::DirectionType::USER_POLL};
    else
        dtList = {NOMAD::DirectionType::USER_POLL, NOMAD::DirectionType::ORTHO_2N};
    allParams->setAttributeValue("DIRECTION_TYPE", dtList);

    allParams->setAttributeValue("QUAD_MODEL_SEARCH", false);
    allParams->setAttributeValue("NM_SEARCH", false);

    allParams->setAttributeValue("DISPLAY_DEGREE", 3);
    allParams->setAttributeValue("DISPLAY_STATS", NOMAD::ArrayOfString("bbe ( sol ) obj"));

    allParams->checkAndComply();
}

/*----------------------------------------*/
/*       Callback : poll projete          */
/*----------------------------------------*/
bool myProjectedPollCallback(const NOMAD::Step& step, std::list<NOMAD::Direction> & dirs, const size_t n)
{
    auto mads = dynamic_cast<const NOMAD::Mads*>(step.getRootAlgorithm());
    if (nullptr == mads)
    {
        throw NOMAD::Exception(__FILE__,__LINE__,"No Mads available.");
    }
    auto callingPoll = dynamic_cast<const NOMAD::PollMethodBase*>(&step);
    if (nullptr == callingPoll)
    {
        throw NOMAD::Exception(__FILE__,__LINE__,"No poll method available.");
    }
    auto frameCenter = callingPoll->getFrameCenter();

    auto pbParams = mads->getPbParams();
    auto nPb = pbParams->getAttributeValue<size_t>("DIMENSION");
    if (nPb != n)
    {
        throw NOMAD::Exception(__FILE__,__LINE__,"Dimension pb.");
    }

    auto mesh = step.getIterationMesh();
    if (nullptr == mesh)
    {
        throw NOMAD::Exception(__FILE__,__LINE__,"No mesh available.");
    }
    NOMAD::ArrayOfDouble boxSize = mesh->getDeltaFrameSize();

    dirs.clear();
    NOMAD::Direction dirUnit(n, 0.0);
    NOMAD::Direction::computeDirOnUnitSphere(dirUnit);
    while (dirUnit[0] == 0)
    {
        NOMAD::Direction::computeDirOnUnitSphere(dirUnit);
    }

    // Matrice M pour la factorisation QR
    auto ** M = new double*[n];
    for (size_t i = 0; i < n; ++i)
    {
        M[i] = new double [n];
        M[i][0] = dirUnit[i].todouble();
        for (size_t j = 1; j < n; ++j)
        {
            M[i][j] = (i == j)? 1.0:0.0;
        }
    }

    auto ** Q = new double*[n];
    auto ** R = new double*[n];
    for (size_t i = 0; i < n; ++i)
    {
        Q[i] = new double [n];
        R[i] = new double [n];
    }

    std::string error_msg;
    bool success = NOMAD::qr_factorization (error_msg,M,Q,R,static_cast<int>(n),static_cast<int>(n));

    if ( !success || !error_msg.empty())
    {
        std::cerr << "QR decomposition for QR 2N poll method has failed" << std::endl;
        return false;
    }

    // Centre du frame
    Vec center(n);
    for (size_t j = 0; j < n; ++j) center[j] = (*frameCenter)[j].todouble();

    std::vector<Vec> accepted;   // points projetes deja generes (anti-doublons)

    NOMAD::Direction dir(n);
    for (size_t i = 0; i < n; ++i)
    {
        for (int sgn = 1; sgn >= -1; sgn -= 2)
        {
            for (size_t j = 0; j < n; ++j)
            {
                dir[j] = sgn * Q[j][i];
            }

            if (!USE_PROJECTION)
            {
                dirs.push_back(dir);
                continue;
            }

            // 1. Point d'essai non projete : centre + delta .* d
            Vec trial(n);
            for (size_t j = 0; j < n; ++j)
                trial[j] = center[j] + boxSize[j].todouble() * dir[j].todouble();

            // 2. Projection sur C
            Vec y = projectOnC(trial);

            // 3. Rejeter si egal au centre ou deja genere
            double dc = 0.0;
            for (size_t j = 0; j < n; ++j) dc += std::abs(y[j] - center[j]);
            if (dc < TOL_SAME) continue;

            bool dup = false;
            for (const auto& a : accepted)
            {
                double d = 0.0;
                for (size_t j = 0; j < n; ++j) d += std::abs(y[j] - a[j]);
                if (d < TOL_SAME) { dup = true; break; }
            }
            if (dup) continue;
            accepted.push_back(y);

            // 4. Direction equivalente : NOMAD refera centre + delta .* d'
            //    donc d' = (y - centre) / delta
            NOMAD::Direction dproj(n);
            for (size_t j = 0; j < n; ++j)
                dproj[j] = (y[j] - center[j]) / boxSize[j].todouble();
            dirs.push_back(dproj);
        }
    }

    for ( size_t i = 0 ; i < n ; ++i )
    {
        delete [] M[i];
        delete [] Q[i];
        delete [] R[i];
    }
    delete [] Q;
    delete [] R;
    delete [] M;

    return true;
}

/*------------------------------------------*/
/*            NOMAD main function           */
/*------------------------------------------*/
int main()
{
    NOMAD::MainStep TheMainStep;

    auto params = std::make_shared<NOMAD::AllParameters>();
    initAllParams(params);
    TheMainStep.setAllParameters(params);

    std::unique_ptr<My_Evaluator> ev(new My_Evaluator(params->getEvalParams()));
    TheMainStep.setEvaluator(std::move(ev));

    TheMainStep.addCallback<NOMAD::MadsCallbackType::USER_METHOD_POLL>(myProjectedPollCallback);

    TheMainStep.start();
    TheMainStep.run();
    TheMainStep.end();

    return 0;
}
