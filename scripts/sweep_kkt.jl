#
# Sweep through QP problems (mpc and sslsq sets) running the prototype solver
# with full diagnostics, then scan the captured stdout for local-QP KKT
# classification events. Records non-unique / inconsistent events alongside
# the corresponding pinv-FP and safeguard signals to a findings file, plus
# a cross-tabulation focused on the inconsistent-KKT case.
#
# Run with:  julia --project=. scripts/sweep_kkt.jl
#
import FOMPrototypes
using ClarabelBenchmarks
using Dates
using Printf

const ITER_COUNT          = 300
const PER_PROBLEM_TIMEOUT = 120.0      # seconds, hard cap inside solver
# Skip problems whose dimensions would force a dense (m+n)×(m+n) tilde_A
# bigger than ~1.2 GB; the full_diagnostics path otherwise OOMs or hangs
# during setup. This single cap removes the need for a separate tail run.
const MAX_MN              = 12_000
const SAFEGUARD_FACTOR    = 0.9
const FINDINGS_PATH       = joinpath(@__DIR__, "..", "kkt_sweep_findings.txt")

const PROBLEM_SETS = ["mpc", "sslsq"]

const BASE_ARGS = Dict(
    "ref-solver"   => :Clarabel,
    "variant"      => :ADMM,

    "res-norm"     => Inf,
    "rel-kkt-tol"  => 1e-9,

    "accel-memory" => 15,
    "acceleration" => :krylov,
    "safeguard-norm" => :char,
    "safeguard-factor" => SAFEGUARD_FACTOR,

    "krylov-tries-per-mem"  => 1,
    "krylov-operator"       => :B,
    "krylov-zero-init"      => false,

    "anderson-interval"     => 10,
    "anderson-broyden-type" => :QR2,
    "anderson-mem-type"     => :restarted,
    "anderson-reg"          => :none,

    "randomized-regularization" => 1e-8,
    "randomized-operator" => :tilde_A,

    "rho"   => 0.1,
    "rho-update-period" => 100,
    "theta" => 1.0,

    "max-iter"           => ITER_COUNT,
    "max-k-operator"     => ITER_COUNT,
    "print-mod"          => 10_000,
    "print-res-rel"      => true,
    "plot-set"           => :none,
    "show-vlines"        => true,
    "run-fast"           => true,
    "global-timeout"     => PER_PROBLEM_TIMEOUT,
    "loop-timeout"       => PER_PROBLEM_TIMEOUT,
)

# ---- per-attempt accumulator and parser --------------------------------

struct AttemptResult
    iter::Int
    givens::Int
    kkt_class::Symbol            # :unique, :inconsistent, :nonunique_{primal,dual,mixed}, :unknown
    kkt_line::String
    pinv_ratio::Union{Float64, Nothing}        # fp_true / fp_van for pinv candidate
    safeguard_ratio::Union{Float64, Nothing}   # fp_acc / fp_van for GMRES candidate
    safeguard_accepted::Bool
end

mutable struct AttemptBuilder
    has_iter::Bool
    iter::Int
    givens::Int
    kkt_class::Symbol
    kkt_line::String
    pinv_ratio::Union{Float64, Nothing}
    safeguard_ratio::Union{Float64, Nothing}
    safeguard_accepted::Bool
end
AttemptBuilder() = AttemptBuilder(false, -1, -1, :unknown, "", nothing, nothing, false)

function reset!(b::AttemptBuilder)
    b.has_iter = false
    b.iter = -1; b.givens = -1
    b.kkt_class = :unknown; b.kkt_line = ""
    b.pinv_ratio = nothing; b.safeguard_ratio = nothing
    b.safeguard_accepted = false
end

function commit!(b::AttemptBuilder, out::Vector{AttemptResult})
    if b.has_iter
        push!(out, AttemptResult(b.iter, b.givens, b.kkt_class, b.kkt_line,
            b.pinv_ratio, b.safeguard_ratio, b.safeguard_accepted))
    end
    reset!(b)
end

function classify_kkt(line::AbstractString)
    occursin("✓ unique", line)            && return :unique
    occursin("✗ INCONSISTENT", line)      && return :inconsistent
    occursin("⚠ nonunique_primal", line)  && return :nonunique_primal
    occursin("⚠ nonunique_dual", line)    && return :nonunique_dual
    occursin("⚠ nonunique_mixed", line)   && return :nonunique_mixed
    return :unknown
end

function parse_attempts(output::AbstractString)
    out = AttemptResult[]
    b = AttemptBuilder()
    for line in eachline(IOBuffer(output))
        m_hdr = match(r"accel attempt: iter (\d+), givens (\d+)", line)
        if m_hdr !== nothing
            commit!(b, out)
            b.iter    = parse(Int, m_hdr.captures[1])
            b.givens  = parse(Int, m_hdr.captures[2])
            b.has_iter = true
            continue
        end
        if occursin("local QP KKT", line)
            b.kkt_class = classify_kkt(line)
            b.kkt_line  = strip(line)
            continue
        end
        m_pinv = match(r"T\(pinv sol\): fp_true/fp_van = ([\d.eE+\-]+)", line)
        if m_pinv !== nothing
            b.pinv_ratio = parse(Float64, m_pinv.captures[1])
            continue
        end
        m_sg = match(r"safeguard: fp_acc/fp_van = ([\d.eE+\-]+).*?(ACCEPTED|REJECTED)", line)
        if m_sg !== nothing
            b.safeguard_ratio    = parse(Float64, m_sg.captures[1])
            b.safeguard_accepted = m_sg.captures[2] == "ACCEPTED"
            continue
        end
    end
    commit!(b, out)
    return out
end

# ---- formatting --------------------------------------------------------

mark(b::Bool) = b ? "✓" : "✗"

function format_signals(a::AttemptResult)
    pinv_str = a.pinv_ratio === nothing ? "pinv N/A" :
        @sprintf("pinv %.2e (red %s, acc-%.2f %s)",
            a.pinv_ratio,
            mark(a.pinv_ratio < 1.0),
            SAFEGUARD_FACTOR, mark(a.pinv_ratio < SAFEGUARD_FACTOR))
    sg_str = a.safeguard_ratio === nothing ? "sg N/A" :
        @sprintf("sg %.2e (red %s) %s",
            a.safeguard_ratio,
            mark(a.safeguard_ratio < 1.0),
            a.safeguard_accepted ? "ACCEPTED" : "REJECTED")
    return pinv_str * "  |  " * sg_str
end

# ---- per-problem runner ------------------------------------------------

function run_one(problem_set, problem_name)
    args = copy(BASE_ARGS)
    args["problem-set"]  = problem_set
    args["problem-name"] = problem_name

    config  = FOMPrototypes.SolverConfig(args)
    problem = FOMPrototypes.fetch_data(problem_set, problem_name)

    if problem.m + problem.n > MAX_MN
        return (:skipped_by_size, problem.m, problem.n, AttemptResult[], "")
    end

    io_capture = IOBuffer()
    run_ok = true
    run_err = ""
    original_stdout = stdout
    original_stderr = stderr
    try
        rd_out, wr_out = redirect_stdout()
        rd_err, wr_err = redirect_stderr()
        reader_out = @async read(rd_out, String)
        reader_err = @async read(rd_err, String)
        try
            FOMPrototypes.run_prototype(problem, problem_set, problem_name, config;
                full_diagnostics = true, spec_plot_period = Inf)
        catch e
            run_ok = false
            run_err = sprint(showerror, e)
        finally
            redirect_stdout(original_stdout)
            redirect_stderr(original_stderr)
            close(wr_out); close(wr_err)
        end
        print(io_capture, fetch(reader_out), fetch(reader_err))
    catch e
        redirect_stdout(original_stdout)
        redirect_stderr(original_stderr)
        run_ok = false
        run_err = sprint(showerror, e)
    end

    attempts = parse_attempts(String(take!(io_capture)))

    !run_ok && isempty(attempts) && return (:fail, problem.m, problem.n, attempts, run_err)
    isempty(attempts)            && return (:no_attempts, problem.m, problem.n, attempts, "")
    return (:done, problem.m, problem.n, attempts, run_ok ? "" : "WARN $run_err")
end

# ---- cross-tab accumulators -------------------------------------------

mutable struct InconsTally
    total::Int
    pinv_reduced::Int
    pinv_at_factor::Int
    pinv_missing::Int
    sg_reduced::Int
    sg_accepted::Int
end
InconsTally() = InconsTally(0, 0, 0, 0, 0, 0)

function tally!(t::InconsTally, a::AttemptResult)
    a.kkt_class === :inconsistent || return
    t.total += 1
    if a.pinv_ratio === nothing
        t.pinv_missing += 1
    else
        a.pinv_ratio < 1.0              && (t.pinv_reduced   += 1)
        a.pinv_ratio < SAFEGUARD_FACTOR && (t.pinv_at_factor += 1)
    end
    a.safeguard_ratio !== nothing && a.safeguard_ratio < 1.0 && (t.sg_reduced += 1)
    a.safeguard_accepted && (t.sg_accepted += 1)
end

# ---- file I/O ----------------------------------------------------------

function log_line(msg)
    open(FINDINGS_PATH, "a") do io
        println(io, msg)
    end
    println(stderr, msg)
end

open(FINDINGS_PATH, "w") do io
    println(io, "# KKT sweep findings — generated $(now())")
    println(io, "# settings: variant=:ADMM, acceleration=:krylov, accel-memory=15,")
    println(io, "#           max-k-operator=$ITER_COUNT, full_diagnostics=true,")
    println(io, "#           per-problem timeout = $(PER_PROBLEM_TIMEOUT)s,")
    println(io, "#           safeguard factor = $SAFEGUARD_FACTOR,")
    println(io, "#           size filter MAX_MN (m+n) = $MAX_MN")
    println(io, "#")
    println(io, "# Tags: UNIFORM-UNIQUE   — every accel attempt classified as unique")
    println(io, "#       INTERESTING      — at least one inconsistent / non-unique event")
    println(io, "#       NO-ATTEMPTS      — no Krylov accel attempt within iteration budget")
    println(io, "#       SKIPPED-BY-SIZE  — m+n exceeds MAX_MN (dense tilde_A too large)")
    println(io, "#       SKIP / FAIL      — setup or run error")
    println(io, "#")
    println(io, "# For INTERESTING problems, each non-unique attempt is detailed with:")
    println(io, "#   - the local QP KKT classification line")
    println(io, "#   - signals: pinv fp_true/fp_van  (red ⇔ <1; acc-$(SAFEGUARD_FACTOR) ⇔ <factor)")
    println(io, "#              sg   fp_acc /fp_van  (red ⇔ <1; ACCEPTED/REJECTED is the safeguard's actual decision)")
    println(io)
end

# ---- main loop ---------------------------------------------------------

tally = InconsTally()

for problem_set in PROBLEM_SETS
    names = sort(collect(keys(ClarabelBenchmarks.PROBLEMS[problem_set])))
    log_line("\n## Set '$problem_set' — $(length(names)) candidate problems")

    for problem_name in names
        status, mm, nn, attempts, errmsg = try
            run_one(problem_set, problem_name)
        catch e
            (:fail, -1, -1, AttemptResult[], sprint(showerror, e))
        end

        dim_str = mm < 0 ? "(?, ?)" : "(m=$mm, n=$nn)"

        if status === :skipped_by_size
            log_line("SKIPPED-BY-SIZE $problem_set/$problem_name $dim_str (m+n=$(mm+nn) > MAX_MN=$MAX_MN)")
            continue
        elseif status === :fail
            log_line("FAIL $problem_set/$problem_name $dim_str :: $errmsg")
            continue
        elseif status === :no_attempts
            log_line("NO-ATTEMPTS $problem_set/$problem_name $dim_str")
            continue
        end

        n_total    = length(attempts)
        n_unique   = count(a -> a.kkt_class == :unique,           attempts)
        n_incons   = count(a -> a.kkt_class == :inconsistent,     attempts)
        n_nup      = count(a -> a.kkt_class == :nonunique_primal, attempts)
        n_nud      = count(a -> a.kkt_class == :nonunique_dual,   attempts)
        n_num      = count(a -> a.kkt_class == :nonunique_mixed,  attempts)
        n_interest = n_incons + n_nup + n_nud + n_num

        for a in attempts
            tally!(tally, a)
        end

        warn_str = isempty(errmsg) ? "" : " :: $errmsg"

        if n_interest > 0
            log_line("INTERESTING $problem_set/$problem_name $dim_str " *
                "attempts=$n_total unique=$n_unique inconsistent=$n_incons " *
                "nu-primal=$n_nup nu-dual=$n_nud nu-mixed=$n_num$warn_str")
            open(FINDINGS_PATH, "a") do io
                for a in attempts
                    a.kkt_class === :unique && continue
                    println(io, "    [iter $(a.iter) | $(a.kkt_class)] $(a.kkt_line)")
                    println(io, "        " * format_signals(a))
                end
            end
        else
            log_line("UNIFORM-UNIQUE $problem_set/$problem_name $dim_str attempts=$n_total$warn_str")
        end
        flush(stdout)
    end
end

# ---- inconsistent-KKT cross-tab ---------------------------------------

T = tally.total
pct(x) = T > 0 ? @sprintf("%5.1f%%", 100 * x / T) : "  N/A"

open(FINDINGS_PATH, "a") do io
    println(io)
    println(io, "# ===================================================================")
    println(io, "# INCONSISTENT-KKT FP-REDUCTION CROSS-TAB")
    println(io, "# ===================================================================")
    println(io, "#")
    println(io, "# Total inconsistent-KKT attempts across all problems:  $T")
    println(io, "#")
    println(io, "# pinv 'true fixed-point' candidate, on these attempts:")
    println(io, @sprintf("#   reduced fp metric (fp_true/fp_van < 1):       %5d   (%s)",
        tally.pinv_reduced, pct(tally.pinv_reduced)))
    println(io, @sprintf("#   would pass safeguard (< %.2f):                %5d   (%s)",
        SAFEGUARD_FACTOR, tally.pinv_at_factor, pct(tally.pinv_at_factor)))
    println(io, @sprintf("#   pinv diagnostics missing/failed:              %5d   (%s)",
        tally.pinv_missing, pct(tally.pinv_missing)))
    println(io, "#")
    println(io, "# GMRES acceleration candidate, on these attempts:")
    println(io, @sprintf("#   reduced fp metric (fp_acc/fp_van < 1):        %5d   (%s)",
        tally.sg_reduced, pct(tally.sg_reduced)))
    println(io, @sprintf("#   accepted by safeguard (< %.2f):               %5d   (%s)",
        SAFEGUARD_FACTOR, tally.sg_accepted, pct(tally.sg_accepted)))
end

log_line("\n# Sweep complete at $(now())")
println("Done. Findings → $FINDINGS_PATH")
