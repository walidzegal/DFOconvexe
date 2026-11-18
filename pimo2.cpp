#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

// Générateur de nombres aléatoires global
std::mt19937 gen(42);
std::uniform_real_distribution<double> unif(0.0, 1.0);
std::normal_distribution<double> norm_dist(0.0, 1.0);

// Fonction objectif d'exemple : Fonction Sphère (Minimum global = 0 à x = [0, 0, ...])
double objective_function(const std::vector<double>& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}

// Génération d'un pas de vol de Lévy (Algorithme de Mantegna)
std::vector<double> levy_flight(int dim, double beta = 1.5) {
    double num = std::tgamma(1.0 + beta) * std::sin(M_PI * beta / 2.0);
    double den = std::tgamma((1.0 + beta) / 2.0) * beta * std::pow(2.0, (beta - 1.0) / 2.0);
    double sigma_u = std::pow(num / den, 1.0 / beta);

    std::vector<double> step(dim);
    for (int j = 0; j < dim; ++j) {
        double u = norm_dist(gen) * sigma_u;
        double v = norm_dist(gen);
        step[j] = u / std::pow(std::abs(v), 1.0 / beta);
    }
    return step;
}

// Maintien des solutions dans les bornes [lb, ub]
void clip_bounds(std::vector<double>& x, double lb, double ub) {
    for (size_t j = 0; j < x.size(); ++j) {
        if (x[j] < lb) x[j] = lb;
        if (x[j] > ub) x[j] = ub;
    }
}

// Algorithme PIMO
std::vector<double> pimo_optimize(int pop_size, int dim, int max_iter, double lb, double ub) {
    // 1. Initialisation de la population
    std::vector<std::vector<double>> pop(pop_size, std::vector<double>(dim));
    std::vector<double> fitness(pop_size);
    
    std::vector<double> best_sol(dim);
    double best_fit = std::numeric_limits<double>::infinity();

    for (int i = 0; i < pop_size; ++i) {
        for (int j = 0; j < dim; ++j) {
            pop[i][j] = lb + unif(gen) * (ub - lb);
        }
        fitness[i] = objective_function(pop[i]);
        if (fitness[i] < best_fit) {
            best_fit = fitness[i];
            best_sol = pop[i];
        }
    }

    // 2. Boucle principale d'optimisation
    for (int t = 0; t < max_iter; ++t) {
        // Facteur de décroissance adaptatif selon l'itération
        double alpha = 1.0 - (static_cast<double>(t) / max_iter);

        for (int i = 0; i < pop_size; ++i) {
            std::vector<double> new_sol = pop[i];

            // Sélection de deux individus aléatoires distincts
            int r1 = std::uniform_int_distribution<int>(0, pop_size - 1)(gen);
            int r2 = std::uniform_int_distribution<int>(0, pop_size - 1)(gen);

            double p = unif(gen);

            if (p < 0.25) {
                // Opérateur 1 : Residual-Guided Projection (RGP)
                for (int j = 0; j < dim; ++j) {
                    double residual = best_sol[j] - pop[i][j];
                    new_sol[j] = pop[i][j] + alpha * unif(gen) * residual;
                }
            } 
            else if (p < 0.50) {
                // Opérateur 2 : Dual Random Projection (DRP)
                for (int j = 0; j < dim; ++j) {
                    double diff = pop[r1][j] - pop[r2][j];
                    new_sol[j] = pop[i][j] + unif(gen) * diff;
                }
            } 
            else if (p < 0.75) {
                // Opérateur 3 : Weighted Random Projection Update (WRPU)
                double w = unif(gen);
                for (int j = 0; j < dim; ++j) {
                    new_sol[j] = w * best_sol[j] + (1.0 - w) * pop[r1][j] + alpha * norm_dist(gen) * 0.1;
                }
            } 
            else {
                // Opérateur 4 : Lévy Flight-Guided Projection (LFGP)
                std::vector<double> levy = levy_flight(dim);
                for (int j = 0; j < dim; ++j) {
                    new_sol[j] = pop[i][j] + 0.01 * levy[j] * (pop[i][j] - best_sol[j]);
                }
            }

            // Vérification des bornes et évaluation gloutonne
            clip_bounds(new_sol, lb, ub);
            double new_fit = objective_function(new_sol);

            if (new_fit < fitness[i]) {
                pop[i] = new_sol;
                fitness[i] = new_fit;

                if (new_fit < best_fit) {
                    best_fit = new_fit;
                    best_sol = new_sol;
                }
            }
        }
    }

    std::cout << "Meilleure fitness trouvee : " << best_fit << std::endl;
    return best_sol;
}

int main() {
    int pop_size = 30;      // Taille de la population
    int dim = 10;           // Nombre de dimensions (variables)
    int max_iter = 500;     // Nombre d'itérations
    double lb = -100.0;     // Borne inférieure
    double ub = 100.0;      // Borne supérieure

    std::vector<double> solution = pimo_optimize(pop_size, dim, max_iter, lb, ub);

    std::cout << "Meilleure solution : [ ";
    for (double val : solution) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;

    return 0;
}
