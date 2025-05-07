# FOMPrototypes.jl

Prototype first-order method (FOM) for convex quadratic programs with cone constraints. The initial code works only for quadratic programs, but will be extended to work for more general cones. A key concern is for the solver to operate without a need to solve linear systems, whether indirectly or by caching a matrix factorisation.