# Stochastic Dynamic Pricing (SDP) Implementation Context

## Project context

This repository implements DTU assignment 46750, *Optimization in Modern
Power Systems* (Fall 2026). It contains deterministic community models
and scenario-generation utilities. The next task is to extend
deterministic dynamic pricing (DP) to stochastic dynamic pricing (SDP)
for Q3.

This is an implementation handoff for a coding agent. Preserve existing
R, I, C, and DP behaviour. Treat deterministic DP as the baseline to
reuse and validate against.

## Relevant architecture

### `model.py`

-   `Results` stores question, status, objective, hourly output, duals,
    and metadata. Its save method exports results.
-   `FlexibleConsumerModel`, `DisutilityConsumer`, and
    `BatteryDisutilityConsumer` implement consumer models.
-   `Community` currently supports contract types `R`, `I`, `C`, and
    `DP`.
-   `Community.build()` builds independent member models for R/I and a
    joint Gurobi model for C/DP.
-   DP adds internal price variables, member-level KKT conditions, dual
    variables, and complementarity constraints. With
    `price_discrimination=False`, the internal price is common across
    members.
-   `Community.solve()` returns member-level results for R/I and a
    community-level `Results` object for C/DP.
-   `CommunityOptimizationModel` wraps the community model.
-   `_add_complementarity()` implements complementarity using big-M
    binaries or SOS1.

Do not rewrite the deterministic DP path in a way that changes its
current behaviour.

### `scenario_generation.py`

Relevant functions include `BetaRule`, `draw_scaling_scenarios(...)`,
`apply_scaling(load, theta)`, `merge_scenarios(...)`, and
`draw_preference_scenarios(...)`.

A joint scenario has this general structure:

``` python
{
    "id": scenario_id,
    "probability": scenario_probability,
    "index": ...,
    "theta": ...,
    "members": {
        consumer_id: {
            appliance_id: perturbed_appliance_dict
        }
    }
}
```

Important: `scenario["members"]` contains appliance-level dictionaries,
not `CommunityData` or `InputData`. A conversion adapter is required.

## Q3 modelling requirements

1.  Each member's reference profile is scaled by a random parameter.
2.  The daily minimum energy requirement follows the perturbed reference
    profile and must be recalculated for each scenario.
3.  Uncertainty in disutility slope `cL` may also be included.
4.  A joint scenario specifies uncertain preference data for every
    member.
5.  The manager chooses one common internal price vector shared across
    scenarios and members.
6.  Member dispatch/responses, grid-interface variables, and payments
    are scenario-dependent.
7.  The base formulation is risk-neutral, using expected costs and
    expected versions of price-property constraints.
8.  At least two additional risk representations are required (for
    example, CVaR and chance constraints).
9.  Use the same in-sample scenario set when comparing formulations and
    evaluate policies out-of-sample.

Do not optimize a separate price vector in each scenario. That does not
represent the required common first-stage price decision.

## Recommended implementation

### 1. Build a scenario-to-data adapter

Implement a function such as:

``` python
def apply_scenario_to_community(base_community_data, scenario):
    ...
```

It should create a fresh scenario-specific community data object by
mapping scenario appliance dictionaries back to baseline
members/appliances.

Requirements: - Match members and appliances using their identifiers. -
Copy data; never mutate baseline objects. - Rebuild each member's
scenario-specific reference profile/load. - Recalculate
scenario-specific daily minimum energy. - Recalculate hourly lower/upper
load bounds consistently with the perturbed reference and assignment
definitions. - Apply scenario-specific `linear_disutility`/`cL` if that
uncertainty is enabled. - Preserve unrelated
technical/network/interface/battery parameters unless explicitly changed
by the scenario.

**Do not guess the data mapping.** Inspect `data_loader.py` (or the
module defining `CommunityData`, `InputData`, and appliance aggregation)
before implementing this adapter. That file had not yet been inspected
when this handoff was prepared. Add checks for valid bounds and correct
scenario daily energy.

### 2. Extend `Community` without breaking DP

Add an SDP route, preferably a contract type `"SDP"`, and
backward-compatible constructor arguments such as `scenarios=None` and
`stochastic_constraints=...`.

Route SDP to a separate method such as `_build_stochastic_dp()`. Keep
R/I/C/DP paths unchanged.

### 3. Reuse follower KKT construction

Refactor deterministic DP's follower KKT construction into a helper if
practical, for example:

``` python
_add_follower_kkt(scenario_id, member, member_variables, internal_price)
```

For SDP: - Internal price variables are common across scenarios (and
across members when price discrimination is disabled). - Member primal
variables are scenario-indexed. - Follower dual variables and
complementarity constraints are scenario-indexed. - Grid
import/export/excess, payments, and scenario costs are scenario-indexed.

A stationarity equation may schematically resemble:

\[ x_t + `\mu`{=tex}*{`\delta`{=tex},i,t}\^{s} -
`\mu`{=tex}*{i,t}\^{-,s} + `\mu`{=tex}\_{i,t}\^{+,s} -
`\mu`{=tex}\_i\^{E,s} = 0, \]

but copy the exact signs and terms from the existing DP derivation. Do
not rely on the schematic equation without checking the code.

**Big-M caution:** bounds must be valid across every scenario, not just
baseline data. Revisit existing big-M values or use SOS1 where
appropriate.

### 4. Formulate the risk-neutral extensive form

Use one joint extensive-form optimization model:

-   First-stage/common decision: internal price vector.
-   Scenario recourse: member responses, grid-interface variables,
    payments, and scenario-specific KKT/dual/complementarity variables.

General risk-neutral objective:

\[ `\min`{=tex}*{x,{y\^s}} `\sum`{=tex}*{s`\in`{=tex}`\mathcal `{=tex}S}
p_s `\left`{=tex}(C\_{`\mathrm{meter}`{=tex}}\^s +
C\_{`\mathrm{private}`{=tex}}\^s`\right`{=tex}), \]

where `x` is the common price vector, `y^s` is scenario recourse, and
`p_s` is scenario probability.

Use the exact objective terms and sign conventions from the assignment
and existing DP model. Inspect the meaning of the existing meter-cost
and private-cost expressions rather than assuming their interpretation.
Validate that probabilities are nonnegative and sum to one (or
explicitly normalize and document).

### 5. Implement stochastic price-property constraints

Expected budget balance may take the form:

\[ `\sum`{=tex}\_s
p_s(`\mathrm{payments}`{=tex}^s-`\mathrm{meter\ cost}`{=tex}^s)=0. \]

Expected individual rationality may take the form:

\[ `\sum`{=tex}\_s p_s(C_i^{DP,s}-C_i^{bench,s})`\leq 0`{=tex}. \]

These are structural examples: match exact definitions and signs to the
assignment and current code. The IR benchmark should be
scenario-specific when uncertain preferences change the member's outside
option. Do not automatically compare every scenario to one deterministic
baseline benchmark. Determine the assignment's benchmark definition and
implement it consistently for each member/scenario.

Keep stochastic-constraint treatment configurable.

### 6. Add risk alternatives

After the risk-neutral model works, implement at least two additional
risk representations, for example:

-   **CVaR:** add threshold and excess-loss variables for the relevant
    cost or constraint violation; document confidence level and
    risk-aversion parameter.
-   **Chance constraints:** enforce a specified probability of
    satisfying a constraint using a mathematically valid scenario
    formulation.

An expected-value constraint is not a chance constraint. State exactly
which quantity is protected and document all risk parameters. Use the
same in-sample scenarios across formulations.

### 7. Add a separate stochastic result type

Prefer a separate `StochasticResults` dataclass rather than breaking the
existing `Results` API. Include: - Solver status and objective. - Common
optimized price vector. - Scenario-indexed dispatch/member responses. -
Scenario-indexed grid-interface quantities and payments. - Scenario
probabilities. - Expected cost and relevant risk metrics. - Risk
formulation/settings and reproducibility metadata.

Preserve deterministic result exports and reporting conventions.

## Implementation and validation sequence

1.  Inspect `data_loader.py` (or equivalent) and establish exact
    `CommunityData`, `InputData`, and appliance mapping.
2.  Implement the scenario adapter; test that it creates independent
    data and does not mutate baseline objects.
3.  Validate scenario profiles, daily minimum energy, hourly bounds, and
    optional `cL` changes.
4.  Refactor/reuse KKT code and confirm deterministic DP results remain
    unchanged.
5.  Run a one-scenario equivalence test: a baseline-identical SDP
    scenario should reproduce DP results within solver tolerances and
    with matching settings.
6.  Build the multi-scenario extensive form with common prices and
    scenario-indexed recourse.
7.  Validate expected objective, scenario probabilities, BB, and IR.
8.  Add and test CVaR and chance-constraint variants.
9.  Evaluate the fixed policy on a separate out-of-sample scenario set;
    do not use test scenarios to optimize prices.
10. Run regression tests for R/I/C/DP.

## Design cautions

-   Never mutate baseline data while constructing scenarios.
-   Never create scenario-specific prices for the common-price SDP
    formulation.
-   Do not solve scenarios independently, then average their optimized
    prices; that is not equivalent to joint stochastic optimization.
-   A single extensive-form solve is the natural formulation: common
    prices are shared, while recourse and follower KKT systems are
    scenario-indexed.
-   Separate scenario solves can be useful for scenario-specific
    outside-option benchmarks or evaluation of a fixed policy, but they
    do not replace the joint stochastic optimization.
-   Check KKT signs, dual bounds, complementarity logic, and big-M
    validity against the existing implementation.
-   Treat assignment definitions as authoritative; do not invent
    interpretations for ambiguous objective or constraint terms.
-   In scenario generation, `aligned` mode pairs the same sample index
    across independently drawn streams. Understand the selected mode and
    its implications for cross-member scenario dependence.

## Open dependency

Before implementing the scenario adapter, inspect the repository's
data-loading module (previously identified as `data_loader.py`) to
establish the exact structure and conversion rules for `CommunityData`,
`InputData`, and appliance fields. This handoff intentionally does not
invent those mappings.
