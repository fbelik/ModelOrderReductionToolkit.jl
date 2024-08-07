# Reduced Basis Method Tutorial

This tutorial follows closely to the book Reduced Basis Methods for Partial Differential Equations by Quateroni, Alfie, Manzoni, and Negri. For more information, see their text, source 2.

### Problem formulation and motivation

In this tutorial, we consider scalar, linear, elliptic, parametrized PDEs of the form
```math
\mathcal{L}(u(x,p),p) = f(x,p)
```
where ``p`` is some parameter (vector), and the solution ``u`` depends on a spatial variable ``x`` and the parameter. We are interested in such problems specifically as upon discretization, say with finite elements, the discrete problem can be written in the form
```math
A(p) u(p) = b(p)
```
where ``A(p)\in\mathbb{R}^{N\times N}``, ``u(p)\in\mathbb{R}^N``, and ``b(p)\in\mathbb{R}^N``. Additionally, we will assume **affine** parameter dependence, i.e., we can write ``A(p)`` as 
```math
A(p) = \sum_{i=1}^{QA} \theta_i^A(p) A_i,
```
and ``b(p)`` as
```math
b(p) = \sum_{i=1}^{Qb} \theta_i^b(p) b_i.
```
Note that if a problem does not match this form, there exist algorithms ((D)EIM) to convert the problem to this form.

Upon sufficient discretization, we expect ``N`` to be large, and thus the problem of inverting ``A(p)`` several times for different parameter values can be expensive. A model order reduction technique is to build a reduced basis (RB) approximation to the solution. To do this, we wish to build an appropriate ``r`` dimensional RB space, with ``r \ll N``, on wish to use Galerkin projection. 

Specifically, given linearly independent (assumed orthogonal) basis vectors to this space, ``\{v_i\}_{i=1}^r``, we construct the RB space matrix
```math
V = \begin{bmatrix} | & | &  & | \\ v_1 & v_2 & \cdots & v_r \\  | & | &  & |  \end{bmatrix} \in \mathbb{R}^{N \times r}
```
such that the problem can be approximated by
```math
V^T A(p) V u_r(p) = V^T b,\quad u(p) \approx V u_r(p).
```
where now the task is to invert the much smaller, ``r\times r`` matrix, ``V^T A(p) V`` to form ``u_r(p)``, and then the solution is approximated by ``V u_r(p)``. Additionally, due to the affine parameter dependence of ``A(p)``, we need not store any terms that depend on ``N``, rather we only need to store the matrices ``V^T A_i V \in \mathbb{R}^{r\times r}`` for ``i=1,\ldots,QA``.

Now, suppose we wished to solve an inverse problem, such as finding the parameter vector ``p^*`` that yields some some 'optimal' solution ``u^*(p)``. Or suppose that we wish to perform a sensitivity analysis of ``u(p)`` on several different parameter values ``p``. These tasks would typically require us to solve the full-order problem a large number of times which may be computationally expensive. 

If we are willing to spend **offline** time to generate an RB space, ``V``, with dimension ``r\ll N``, then we can much more efficiently spend time **online** computing the Galerkin projected solution, ``V u_r(p)``, at a fraction of a cost of computing the full-order solution.

We will consider a steady state heat equation for this tutorial with affine-parameter dependent spacial diffusion coefficient and forcing terms. Once discretized, it can be written in the form ``A(p) u = b(p)`` with ``A(p)`` and ``b(p)`` each with affine parameter dependence. This model can be instantiated in `ModelOrderReductionToolkit.jl` by calling `PoissonModel()`.

```@example 1
using ModelOrderReductionToolkit
model = PoissonModel()
```

We can then form a snapshot matrix over a set of ``P=125`` parameter vectors.
```@example 1
params = [[i,j,k] for i in range(0,1,5) for j in range(0,1,5) for k in range(0,1,5)]
P = length(params)

S = zeros(output_length(model), P)
for i in 1:P
    p = params[i]
    u = model(p)
    S[:,i] .= u
end

S
```
Let's visualize the solutions.
```@example 1
using Plots
plt = plot()
for i in 1:P
    plot!(S[:,i],label=false,alpha=0.25)
end
title!("Solution Set")
savefig(plt, "rbm_tut1.svg"); nothing # hide
```
![](rbm_tut1.svg)

### Proper Orthogonal Decomposition/Principal Component Analysis

We know from the Schmidt-Eckart-Young theorem that the $r$-dimensional linear subspace that captures the most "energy" from the solutions in ``S`` (per the Frobenius norm) is the one spanned by the first ``r`` left singular vectors of ``S``. More specifically, if we denote ``V\in\mathbb{R}^{N\times r}`` to be the matrix whose ``r`` columns are the first ``r`` left singular vectors of ``S``, and let ``\sigma_1\geq\sigma_2\geq\ldots\geq\sigma_N`` be the singular values of ``S``, then we can write that
```math
||S - VV^TS||_F = \min_{\text{rank}(B)\leq r} ||A - B||_F = \sqrt{\sum_{i=r+1}^N \sigma_i^2}.
```
where ``VV^TS`` is the projection of ``S`` onto the columns of ``V``.

We can explicitly compute the SVD and pull the first ``r`` columns as 
```@example 1
using LinearAlgebra
r = 5
U,s,_ = svd(S)
V = U[:,1:r]
nothing; # hide
```

We can also plot the exponential singular value decay, suggesting to us that such an RBM will perform well.
```@example 1
plt = plot(s, yaxis=:log, label=false)
yaxis!("Singular Values")
xaxis!("Dimension")
savefig(plt, "rbm_tut2.svg"); nothing # hide
```
![](rbm_tut2.svg)

Now, the Schmidt-Eckart-Young theorem tells us that this basis is optimal in the sense that it minimizes ``l^2`` error in directly projecting our solutions, i.e., performing ``u(p) \approx VV^Tu(p)``. Let's visualize the accuracy of these projections.
```@example 1
plt = plot()
colors = palette(:tab10)
idxs = [rand(1:P) for i in 1:6]
for i in 1:6
    idx = idxs[i]
    p = params[idx]
    plot!(S[:,idx], c=colors[i], label=false)
    u_approx = V * V' * S[:,idx]
    plot!(u_approx, c=colors[i], label=false, ls=:dash)
end
title!("Truth and projected POD solutions")
savefig(plt, "rbm_tut3.svg"); nothing # hide
```
![](rbm_tut3.svg)

However, we wished to create a reduced order model (ROM) such that given any new parameter value, we can quickly reproduce a new solution. As was noted before, we do this through a Galerkin projection
```math
V^T A(p) V u_r(p) = V^T b \implies u(p) \approx V u_r(p) = u_\text{approx}(p)
```
from which we require only inverting an ``r\times r`` matrix. Although this is no longer guaranteed "optimal" by the Schmidt-Eckart-Young theorem, let's see how this performs on the same snapshots. To perform this task in `ModelOrderReductionToolkit.jl`, we pass the snapshot matrix into a `PODReductor` object and form a ROM from the reductor.
```@example 1
pod_reductor = PODReductor(model)
add_to_rb!(pod_reductor, S)
pod_rom = form_rom(pod_reductor, r)
plt = plot()
colors = palette(:tab10)
for i in 1:6
    idx = idxs[i]
    p = params[idx]
    plot!(S[:,idx], c=colors[i], label=false)
    u_r = pod_rom(p)
    u_approx = lift(pod_reductor, u_r)
    plot!(u_approx, c=colors[i], label=false, ls=:dash)
end
title!("Truth and projected Galerkin POD solutions")
savefig(plt, "rbm_tut4.svg"); nothing # hide
```
![](rbm_tut4.svg)

As we can see from these plots, a ``5``-dimensional approximation is quite accurate here! Even though after discretization, these solutions lie in ``\mathbb{R}^{999}``, we have shown that the solution manifold lies approximately on a ``5``-dimensional space. Additionally, even though we were only guaranteed "optimality" from direct projection of solutions, we still have very good accuracy when we use a Galerkin projection on the problem.

This process of projection onto left singular values is typically called **Proper Orthogonal Decomposition** (POD). Note that forming a `PODReductor` will call `svd` on the snapshot matrix. We can access the singular values from `pod_reductor.S`, and the left-singular vectors from `pod_reductor.V` (note that left-singular vectors are usually denoted with ``U``, but we use ``V`` to stick with RB notation).

### Strong Greedy Algorithm

An alternative way to generate this reduced basis is through a process called the **strong greedy algorithm**. This algorithm is called greedy, because we iteratively choose basis elements in a greedy way. We begin by choosing ``v_1`` to be the column of our solution matrix, ``S`` with the largest norm, and then normalized it by its length
```math
||s_1^*|| = \max_i ||s_i||,\quad v_1 = \frac{s_1^*}{||s_1^*||}.
```
Now, we use a Gram-Schmidt procedure to orthogonalize all other columns of ``S`` with respect to ``v_1``:
 ```math
s_i^{(1)} = s_i - (v_1^T s_i) v_1,\quad i=1,\ldots,P.
```
After the ``j-1``'st element ``v_{j-1}`` is chosen and all of the orthogonalization is performed, we then choose ``v_{j}`` to be the column of ``S^{(j-1)}`` which has the largest norm, i.e., has the worst projection error:
```math
||s_j^*|| = \max_i ||s_i^{(j-1)}||,\quad v_{j} = \frac{s_j^*}{||s_j^*||},
```
and again orthogonalize
 ```math
s_i^{(j)} = s_i^{(j-1)} - (v_j^T s_i^{(j-1)}) v_j,\quad i=1,\ldots,P.
```

Note that this procedure is exactly like performing a pivoted QR factorization on the matrix ``S``. Let's form this reduced basis of the same dimension:
```@example 1
Q,_,_ = qr(S, LinearAlgebra.ColumnNorm())
V = Q[:,1:r]
nothing; # hide
```

Now, we will play the same game. First, we directly project the solutions onto this space
```@example 1
plt = plot()
for i in 1:6
    idx = idxs[i]
    p = params[idx]
    plot!(S[:,idx], c=colors[i], label=false)
    u_approx = V * V' * S[:,idx]
    plot!(u_approx, c=colors[i], label=false, ls=:dash)
end
title!("Truth and projected SG solutions")
savefig(plt, "rbm_tut5.svg"); nothing # hide
```
![](rbm_tut5.svg)

Now, we will use an `SGReductor` object to form a Galerkin-projected reduced order model.
```@example 1
sg_reductor = SGReductor(model)
add_to_rb!(sg_reductor, S)
sg_rom = form_rom(sg_reductor, r)
plt = plot()
for i in 1:6
    idx = idxs[i]
    p = params[idx]
    plot!(S[:,idx], c=colors[i], label=false)
    u_r = sg_rom(p)
    u_approx = lift(sg_reductor, u_r)
    plot!(u_approx, c=colors[i], label=false, ls=:dash)
end
title!("Truth and projected Galerkin SG solutions")
savefig(plt, "rbm_tut6.svg"); nothing # hide
```
![](rbm_tut6.svg)

This procedure also performs quite well. We may expect the POD algorithm to be a bit more accurate/general as it can choose basis elements that are not "in the columns" of ``S``. Similar to the `PODReductor` object, we can access the reduced basis from `sg_reductor.V`.

### Weak Greedy Algorithm

Now, one downside of the above procedures was that we needed the matrix of full-order solutions ahead of time to perform either the SVD or QR factorizations. If our model was very computationally expensive, we would not want to have to do this. This is where the **weak greedy algorithm** is useful. It is again a greedy algorithm as we will be choosing "columns" greedily, but we wish to not have to construct all columns directly.

Suppose now, instead of having access to the columns ``s_i`` which correspond to the full-order solutions ``u(p_i)``, we only have access to the parameter values ``p_i``. Generally, we would wish to have an a priori error estimator for a Galerkin-projected ROM such that we could loop over our parameter vectors and choose the one with the estimated maximum error. In this tutorial, we use a stability-residual approach

One can show that there exists an upper-bound on projection error, given by
```math
||u(p) - V u_r(p)|| = ||A(p)^{-1} b(p) - V u_r(p)|| \leq \frac{||b(p) - A(p) V u_r(p)||}{\sigma_{min}(A(p))}
```
where ``\sigma_{min}(A(p))`` is the minimum singular value of ``A(p)``. Note that this upperbound on the error does not depend on the full order solution, ``u(p)``. So, we loop through each parameter vector ``p_i``, and select the one, ``p^*`` that yields the highest upper-bound error. We then form the full-order solution ``u(p_i)``, normalize it, and append it as a column of ``V``. Note that unlike in the strong algorithm, since we are not using true error, we are not guaranteed to choose the next "best" column of ``V``. However, if we are computing a reduced basis of size ``r``, then we only need to call the full-order model ``r`` times.

We now need a method to approximate (a lowerbound of ) ``\sigma_{min}(A(p))``, and then the numerator of the above can be computed explicitly. One way of doing this is through the **successive constraint method** (SCM). This method takes advantage of the affine parameter dependence of ``A(p)``, see source 1. We will form an SCM object and initialize an object to compute the norm of the residual through a `StabilityResidualErrorEstimator`.
```@example 1
error_estimator = StabilityResidualErrorEstimator(model, params);
```
Note that this method assumes that the model is coercive, i.e., the matrix `A(p)` is symmetric positive definite for each parameter. For this model, we know that this is the case if the parameter vectors have entries between 0 and 1. For a noncoercive model, add the keyword argument `coercive=false`. With this in place, we have enough to construct the weak greedy reductor.
```@example 1
wg_reductor = WGReductor(model, error_estimator)
```
Note that we have to build the reduced basis by looping over a parameter set, we will do this by calling `add_to_rb!`. Afterwards, since the reductor must store the ROM at each step to make error approximations, we can simply pull it from the reductor object.
```@example 1
add_to_rb!(wg_reductor, params, r)
println(" ") # hide
```
```@example 1
wg_rom = wg_reductor.rom
```


We can access the greedily chosen reduced basis by calling (note that for computational purposes, ``V`` is stored as a `VectorOfVectors` object).
```@example 1
wg_reductor.V
```
We can now visualize these solutions by calling `wg_rom(p)` on a paramater vector `p`.
```@example 1
plt = plot()
for i in 1:6
    idx = idxs[i]
    p = params[idx]
    plot!(S[:,idx], c=colors[i], label=false)
    u_r = wg_rom(p)
    u_approx = lift(wg_reductor, u_r)
    plot!(u_approx, c=colors[i], label=false, ls=:dash)
end
title!("Truth and WG solutions")
savefig(plt, "rbm_tut7.svg"); nothing # hide
```
![](rbm_tut7.svg)

### Comparison of the methods

Let's compare the above algorithms by comparing their average and worst case accuracy over the parameter set.
```@example 1
errors = [Float64[] for _ in 1:3]
for (i,p) in enumerate(params)
    pod_error = norm(lift(pod_reductor, pod_rom(p)) .- S[:,i])
    push!(errors[1], pod_error)
    sg_error = norm(lift(sg_reductor, sg_rom(p)) .- S[:,i])
    push!(errors[2], sg_error)
    wg_error = norm(lift(wg_reductor, wg_rom(p)) .- S[:,i])
    push!(errors[3], wg_error)
end
println("Errors for RB dimension r=$r")
println("POD mean error: $(sum(errors[1]) / length(errors[1]))")
println("POD worst error: $(maximum(errors[1]))")
println("SG mean error: $(sum(errors[2]) / length(errors[2]))")
println("SG worst error: $(maximum(errors[2]))")
println("WG mean error: $(sum(errors[3]) / length(errors[3]))")
println("WG worst error: $(maximum(errors[3]))")
nothing # hide
```

Let's repeat the process for a reduced basis of dimension ``r=15``
Let's compare the above algorithms by comparing their average and worst case accuracy over the parameter set.
```@example 1
oldr = r
r = 15
pod_rom = form_rom(pod_reductor, r)
sg_rom = form_rom(sg_reductor, r)
add_to_rb!(wg_reductor, params, r - oldr)
wg_rom = wg_reductor.rom
errors = [Float64[] for _ in 1:3]
for (i,p) in enumerate(params)
    pod_error = norm(lift(pod_reductor, pod_rom(p)) .- S[:,i])
    push!(errors[1], pod_error)
    sg_error = norm(lift(sg_reductor, sg_rom(p)) .- S[:,i])
    push!(errors[2], sg_error)
    wg_error = norm(lift(wg_reductor, wg_rom(p)) .- S[:,i])
    push!(errors[3], wg_error)
end
println("Errors for RB dimension r=$r")
println("POD mean error: $(sum(errors[1]) / length(errors[1]))")
println("POD worst error: $(maximum(errors[1]))")
println("SG mean error: $(sum(errors[2]) / length(errors[2]))")
println("SG worst error: $(maximum(errors[2]))")
println("WG mean error: $(sum(errors[3]) / length(errors[3]))")
println("WG worst error: $(maximum(errors[3]))")
nothing # hide
```

In conclusion, the weak greedy algorithm takes advantage of the affine parameter dependence of ``A(p)`` and ``b(p)``, and uses an upper-bound error approximator to produce a reduced basis that approximates solutions with comparable error compared to the strong greedy algorithm and the POD algorithm without needing to compute all solutions ahead of time.

### References:
1. D.B.P. Huynh, G. Rozza, S. Sen, A.T. Patera. A successive constraint linear optimization method for lower bounds of parametric coercivity and inf–sup stability constants. Comptes Rendus Mathematique. Volume 345, Issue 8. 2007. Pages 473-478. https://doi.org/10.1016/j.crma.2007.09.019.
2. Quarteroni, Alfio, Andrea Manzoni, and Federico Negri. Reduced Basis Methods for Partial Differential Equations. Vol. 92. UNITEXT. Cham: Springer International Publishing, 2016. http://link.springer.com/10.1007/978-3-319-15431-2.