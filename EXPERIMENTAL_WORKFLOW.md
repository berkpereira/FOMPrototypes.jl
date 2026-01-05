# Experimental Workflow Guide

## Philosophy: Lower the Barrier to Trying Ideas

**Core principle:** Make validating ideas cheap. Most experiments fail—that's fine. The goal is to try many ideas quickly, not to write perfect code for each one.

**Key insight:** Iteration count accuracy matters first. Code quality, performance optimization, and integration come later—only for ideas that work.

---

## Three-Tier Experimentation Framework

### Tier 1: Quick Validation Scripts
**Purpose:** Validate basic concept, check iteration counts
**Location:** `scripts/experimental/`
**Code quality:** Throwaway—messy is fine
**Time investment:** 2 hours to 1 day

**Rules:**
- ✅ Copy-paste code liberally (don't reuse abstractions)
- ✅ Allocate freely (no scratch buffer reuse)
- ✅ Hardcode parameters
- ✅ Skip safeguarding, recycling, complex features
- ✅ Use whatever packages help (CUDA.jl, etc.)
- ❌ Don't worry about performance
- ❌ Don't integrate with workspace dispatch
- ❌ Don't commit to main branch

**Output:** Does the idea work mathematically? What are iteration counts vs baseline?

**Graduation criteria:** If iteration counts beat baseline on 3+ test problems → move to Tier 2

---

### Tier 2: Experimental Workspace Types
**Purpose:** Proper comparison with existing methods
**Location:** `src/alg/experimental_*.jl` on feature branch
**Code quality:** Readable but hacky
**Time investment:** 1-2 weeks

**Rules:**
- ✅ Create new workspace type (e.g., `GPUVanillaWorkspace`)
- ✅ Integrate minimally (hook into `run_prototype()` via config)
- ✅ Duplicate code rather than forcing reuse (especially method operators)
- ✅ Allocate own scratch buffers (no sharing)
- ✅ Reuse problem loading and diagnostics
- ⚠️ Can skip safeguarding initially, add if method works
- ❌ Don't refactor existing code to accommodate
- ❌ Don't try to unify with two-column Krylov operator

**Output:** Benchmark results comparing to vanilla/Anderson/Krylov on problem sets

**Graduation criteria:** If beats baseline meaningfully + drafting paper → move to Tier 3

---

### Tier 3: Publication-Ready Integration
**Purpose:** Clean code for paper submission
**Location:** Merged to main branch, tagged
**Code quality:** Production-grade
**Time investment:** 2-3 weeks

**Rules:**
- ✅ Proper abstractions (if beneficial)
- ✅ Optimize scratch buffer usage
- ✅ Add comprehensive tests
- ✅ Full diagnostics integration
- ✅ Documentation and comments
- ✅ Performance optimization
- ✅ Safeguarding and recycling

**Output:** Tagged release, ready for paper reproducibility

---

## Git Workflow for Experiments

### Feature Branch Structure
```
main (ECC 2026 baseline - tagged and stable)
│
├── feature/gpu-experiments
├── feature/adaptive-stepsizes
├── feature/rnla (current)
└── feature/other-idea
```

### Typical Experiment Lifecycle

1. **Start new idea:**
   ```bash
   git checkout -b feature/idea-name
   ```

2. **Tier 1 - Script phase:**
   - Work in `scripts/experimental/idea_name.jl`
   - Commit to feature branch (optional—can stay uncommitted)
   - **70% of ideas die here** → document learnings, abandon branch

3. **Tier 2 - Integration phase:**
   - Add `src/alg/idea_name.jl` + workspace type
   - Modify `src/core/config.jl` for new acceleration option
   - Commit regularly to feature branch
   - Run benchmarks on fenway
   - **50% of remaining ideas die here** → merge notes to main, delete code

4. **Tier 3 - Publication phase:**
   - Refactor and optimize
   - Add tests
   - Merge to main
   - Tag release (e.g., `v0.5-idea-name-paper`)

### What to Commit Where

**Feature branches (safe to commit):**
- Tier 1 scripts (optional)
- Tier 2 experimental code
- Benchmark results, plots
- Personal notes in `scripts/experimental/README.md`

**Main branch (only commit):**
- Tier 3 production code
- Working baselines
- Clean tagged releases for papers

**Never commit:**
- Large data files
- Fenway-specific paths
- Personal scratch files

---

## Specific Guidance for Common Experiments

### GPU Work
- **Tier 1:** Start with vanilla ADMM, convert to CuArrays
- **Simplification:** Keep Cholesky solve on CPU (hybrid approach)
- **Focus:** Validate GPU setup before attempting acceleration
- **Don't:** Try to port two-column Krylov operator initially

### New Acceleration Methods
- **Tier 1:** Copy `vanilla_step!()`, add your acceleration logic
- **Simplification:** Use `onecol_method_operator!()`, ignore two-column
- **Don't:** Try to create unified operator interface with Krylov

### Adaptive Parameters
- **Tier 1:** Hardcode adaptive logic in loop
- **Tier 2:** Add fields to workspace, integrate with config
- **Reuse:** Existing residual computation infrastructure

### Problem-Specific Methods
- **Tier 1:** Hardcode problem structure (e.g., for OPF problems)
- **Value:** Understand limits of generic approach
- **Don't:** Generalize until Tier 3

---

## Decision Guides

### When to move Tier 1 → Tier 2?
✅ Move if:
- Iteration counts improve on 3+ diverse problems
- Improvement is substantial (>20% reduction)
- You can explain why it works

❌ Stay in Tier 1 if:
- Only works on one problem type
- Improvement is marginal
- You're not sure if it's a bug or a feature

### When to move Tier 2 → Tier 3?
✅ Move if:
- Beats baseline on problem set
- You're drafting a paper
- You need reproducible benchmarks

❌ Stay in Tier 2 if:
- Still exploring variations
- Not sure which version is best
- Paper is far away

### When to abandon?
✅ Abandon if:
- Doesn't beat baseline after tuning
- Works only on trivial problems
- Too complex for the benefit
- Better idea comes along

**Remember:** Abandoning dead ends quickly is success, not failure.

---

## Anti-Patterns to Avoid

### ❌ Premature Abstraction
"I should create a unified operator interface before trying this"
→ Just duplicate the code and see if the idea works first

### ❌ Premature Optimization
"I should use scratch buffers efficiently in my experimental code"
→ Allocate freely in Tier 1, optimize only in Tier 3

### ❌ Perfectionism
"This experimental code is messy, I should clean it up"
→ Only clean up code that's going into a paper

### ❌ Over-Integration
"I should make this work with the Krylov operator too"
→ Only integrate what you need for your specific experiment

### ❌ Feature Creep
"While I'm here, I should also add..."
→ Focus on validating one idea at a time

---

## Mantras

- **"Messy code that answers the question > clean code that doesn't exist"**
- **"Duplicate now, deduplicate later (maybe never)"**
- **"Most ideas will fail—make failing cheap"**
- **"Iteration counts first, performance later"**
- **"When in doubt, Tier 1 script it out"**

---

## Quick Reference: Starting a New Experiment

```julia
# 1. Create Tier 1 script
# scripts/experimental/my_idea.jl

import FOMPrototypes

# Copy-paste problem loading
problem = FOMPrototypes.fetch_data("sslsq", "NYPA_Maragal_3_huber")

# Copy-paste minimal solver loop from src/alg/vanilla.jl
# Hack in your idea
# Run it, check iteration counts

# 2. Compare to baseline
# Run scripts/main_repl.jl with acceleration = :none
# Did you beat it? Yes → Tier 2. No → try different idea.
```

---

## Expected Statistics

- **Tier 1 experiments:** 10-20 per year (quick!)
- **Tier 2 implementations:** 3-5 per year (promising ideas)
- **Tier 3 integrations:** 1-2 per year (paper-worthy ideas)

**Target:** More experiments, not more perfect code.

---

*Remember: Your PhD constraint is time, not compute. Optimize for trying more ideas, not for perfect code.*
