@kwdef struct SolverConfig
    ref_solver::Symbol = :SCS
    variant::Symbol = :ADMM
    problem_set::String = "socp"
    problem_name::String = ""

    res_norm::Float64 = Inf
    rel_kkt_tol::Float64 = 1e-6

    accel_memory::Int = 20
    acceleration::Symbol = :none # in {:none, :krylov, :anderson}
    safeguard_norm::Symbol = :char
    safeguard_factor::Float64 = 0.99

    krylov_tries_per_mem::Int = 3
    krylov_operator::Symbol = :tilde_A

    anderson_interval::Int = 10
    anderson_broyden_type::Symbol = :normal2
    anderson_mem_type::Symbol = :rolling
    anderson_reg::Symbol = :none

    rho::Float64 = 1.0
    theta::Float64 = 1.0

    restart_period::Real = Inf
    linesearch_period::Real = Inf
    linesearch_eps::Float64 = 0.001

    max_iter::Int = 1000
    max_k_operator::Int = 1000
    print_mod::Int = 50
    print_res_rel::Bool = true
    show_vlines::Bool = false
    run_fast::Bool = true
    global_timeout::Float64 = 60.0
    loop_timeout::Float64 = 30.0
end

# allow string/symbol indexing with hyphens or underscores
_normalise_key(key::AbstractString) = replace(key, '-' => '_')
_normalise_key(key::Symbol) = _normalise_key(String(key))

function Base.getindex(config::SolverConfig, key::AbstractString)
    field = Symbol(_normalise_key(key))
    isdefined(config, field) || throw(KeyError(key))
    return getfield(config, field)
end
Base.getindex(config::SolverConfig, key::Symbol) = config[String(key)]

function Base.haskey(config::SolverConfig, key)
    field = Symbol(_normalise_key(key))
    return field in fieldnames(SolverConfig)
end

_config_kwargs(d::AbstractDict) = Dict{Symbol, Any}(
    Symbol(_normalise_key(k)) => v for (k, v) in d if Symbol(_normalise_key(k)) in fieldnames(SolverConfig)
)

SolverConfig(d::AbstractDict) = SolverConfig(; _config_kwargs(d)...)

function SolverConfig(config::SolverConfig; kwargs...)
    merged = merge(Dict{Symbol, Any}(), Dict(Symbol(k) => getfield(config, k) for k in fieldnames(SolverConfig)), Dict(kwargs))
    return SolverConfig(; merged...)
end

function to_dict(config::SolverConfig)
    d = Dict{String, Any}()
    for name in fieldnames(SolverConfig)
        d[replace(String(name), '_' => '-')] = getfield(config, name)
    end
    return d
end
