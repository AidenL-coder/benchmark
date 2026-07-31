"""cbs -- Capability-Boundary Study of self-improving coding agents.

This package is the *measurement instrument* described in
`docs/CLAUDE_CODE_PROJECT_BRIEF.md` section 5.2. It is deliberately kept separate from
any forked self-improvement loop (`S_evo`): the loop is borrowed, the instrument
is the contribution.

Layout
------
cbs.config      configuration loading/validation (all experiments are config-driven)
cbs.budget      matched-compute accountant (brief section 4: equal budgets across systems)
cbs.models      provider-agnostic frozen-model clients (mock / OpenAI-compatible)
cbs.sandbox     pluggable sandboxed execution (Docker-preferred)
cbs.tasks       task schema, verifiers, frozen hashed splits
cbs.scaffolds   support-tagged scaffolds (brief section 3.1), incl. the minimal S0
cbs.frontier    reachable-solution frontier estimation (brief section 3.2)
"""

__version__ = "0.1.0"
