"""
PIMO - Projection-Iterative-Methods-based Optimizer
=====================================================

Implementation of the metaheuristic algorithm from:
Yu, D., Ji, Y., & Xia, Y. (2025). "Projection-Iterative-Methods-based
Optimizer: A novel metaheuristic algorithm for continuous optimization
problems and feature selection." Knowledge-Based Systems, 326, 113978.
https://doi.org/10.1016/j.knosys.2025.113978

PIMO combines swarm intelligence with geometric projection principles
through four operators:
    - RGP  : Residual Guided Projection
    - DRP  : Dual Random Projection
    - WRPU : Weighted Random Projection Update