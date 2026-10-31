using OrdinaryDiffEqRosenbrock, OrdinaryDiffEqNonlinearSolve, NonlinearSolve, ADTypes,
    ForwardDiff, LinearAlgebra, SciMLBase, Test

# du₁ = -u₁ + u₂, 0 = u₁ + u₂ - p. The consistent algebraic value is u₂ = p - u₁; then
# u₁' = p - 2u₁, so u₁(t) = p/2 + (u₁(0) - p/2) exp(-2t).
function brown_f!(du, u, p, t)
    du[1] = -u[1] + u[2]
    du[2] = u[1] + u[2] - p
    return nothing
end
const brown_fn = ODEFunction{true, SciMLBase.FullSpecialize}(
    brown_f!; mass_matrix = Diagonal([1.0, 0.0])
)
const brown_nlsolves = (
    nothing, NewtonRaphson(autodiff = AutoFiniteDiff()),
    NewtonRaphson(autodiff = AutoForwardDiff()),
)

@testset "In-place BrownFullBasicInit, nlsolve = $nl" for nl in brown_nlsolves
    ia = BrownFullBasicInit(nlsolve = nl)

    prob = ODEProblem(brown_fn, [1.0, 0.5], (0.0, 1.0), 1.0)
    integ = init(prob, Rodas5P(); initializealg = ia)
    @test integ.u ≈ [1.0, 0.0] atol = 1.0e-10
    @test prob.u0 == [1.0, 0.5]

    prob = ODEProblem(brown_fn, [1.0, 0.5], (0.0, 1.0), 1.0)
    integ = init(
        prob, Rodas5P(); initializealg = ia,
        alias = SciMLBase.ODEAliasSpecifier(alias_u0 = true)
    )
    @test integ.u === prob.u0
    @test prob.u0 ≈ [1.0, 0.0] atol = 1.0e-10

    prob = ODEProblem(brown_fn, [1.0, 0.5], (0.0, 1.0), 1.0)
    sol = solve(prob, Rodas5P(); initializealg = ia, abstol = 1.0e-10, reltol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    u1 = 0.5 + 0.5exp(-2.0)
    @test sol.u[end] ≈ [u1, 1 - u1] atol = 1.0e-7

    # Reinitialization after a callback that sets p = 2 at t = 0.5 changes only the
    # algebraic value, to 2 - u₁
    cb = DiscreteCallback(
        (u, t, integ) -> t == 0.5, integ -> (integ.p = 2.0); initializealg = ia
    )
    sol = solve(
        prob, Rodas5P(); initializealg = ia, callback = cb, tstops = [0.5],
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    @test SciMLBase.successful_retcode(sol)
    i = findall(==(0.5), sol.t)
    @test length(i) == 2
    @test sol.u[i[2]][1] == sol.u[i[1]][1]
    @test sol.u[i[2]][2] ≈ 2.0 - sol.u[i[1]][1] atol = 1.0e-10
    @test sol.u[i[1]][2] ≈ 1.0 - sol.u[i[1]][1] atol = 1.0e-8
end

@testset "In-place BrownFullBasicInit under ForwardDiff" begin
    # The initialized algebraic value is p - u₁(0) = p - 1
    function initialized_algebraic(p)
        prob = ODEProblem(brown_fn, [one(p), one(p) / 2], (0.0, 1.0), p)
        ia = BrownFullBasicInit(nlsolve = NewtonRaphson(autodiff = AutoForwardDiff()))
        return init(prob, Rodas5P(); initializealg = ia).u[2]
    end
    @test initialized_algebraic(1.5) ≈ 0.5 atol = 1.0e-10
    @test ForwardDiff.derivative(initialized_algebraic, 1.5) ≈ 1.0 atol = 1.0e-8
end

# A residual that throws once the algebraic value moves away from its starting value 0.5
# by more than a finite-difference perturbation: the first Newton step goes to 0, so the
# solve throws after the solver has updated its iterate.
function throwing_brown_f!(du, u, p, t)
    abs(u[2] - 0.5) > 0.1 && throw(DomainError(u[2], "algebraic value left [0.4, 0.6]"))
    return brown_f!(du, u, p, t)
end

@testset "In-place BrownFullBasicInit that throws, nlsolve = $nl" for nl in brown_nlsolves
    fn = ODEFunction{true, SciMLBase.FullSpecialize}(
        throwing_brown_f!; mass_matrix = Diagonal([1.0, 0.0])
    )
    # [0.5, 0.5] is consistent for p = 1, so `init` doesn't solve
    prob = ODEProblem(fn, [0.5, 0.5], (0.0, 1.0), 1.0)
    integ = init(prob, Rodas5P(); initializealg = BrownFullBasicInit())
    integ.u[1] = 1.0
    saved_u = copy(integ.u)
    saved_uprev = copy(integ.uprev)
    @test_throws DomainError OrdinaryDiffEqNonlinearSolve._initialize_dae!(
        integ, integ.sol.prob, BrownFullBasicInit(nlsolve = nl), Val(true)
    )
    @test integ.u == saved_u
    @test integ.uprev == saved_uprev
end
