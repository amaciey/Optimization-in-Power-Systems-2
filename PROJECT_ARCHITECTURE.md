# Community Optimization Architecture Context

## Purpose and scope

This document describes the intended conceptual architecture for the community optimization assignment. It provides context for implementation decisions; the repository, starter code, and assignment specification remain authoritative for exact APIs, equations, constraints, and required outputs.

The design goal is to keep community/member data separate from optimization formulations while reusing existing code wherever practical.

## Existing data path and abstractions

The assignment uses a community loader such as `load_community("base", day=...)`. The loaded community is described as containing member data represented by `InputData` objects, with shared contract/interface information available through the community data structure. The community's members may be accessible through `com.members`.

The existing `InputData` representation and Assignment 1 consumer-level optimization code should be reused when they already provide the necessary member data and behaviour. A separate `Consumer` class is not inherently required. Likewise, a separate `Community` wrapper is optional if the loader already returns a suitable community object.

The actual return type, available fields, and existing project APIs must be established from the repository before implementation.

## Separation of responsibilities

The architecture distinguishes three concerns:

1. **Member data:** individual consumers' input data, preferences, and constraints, represented using existing project types.
2. **Community data:** the collection of members and shared contract or grid-interface information.
3. **Optimization formulations:** solver models that use the same underlying data to implement the required market or coordination cases.

A community data container, if needed, should hold or reference members and shared data, and may provide small aggregation helpers. It should not contain solver-specific variables, constraints, objectives, or mutable solver state.

Each optimization run should use a fresh solver model so that variables and constraints do not leak between formulations. Shared input data should not be unintentionally mutated by one run in a way that affects another.

## Community connection and balance

The assignment describes the community as connected through a community connection point without an internal network. The hourly net consumption of the members is balanced against the community's net import/export at that interface.

Schematically, the centralized formulation includes a balance of the form:

```text
sum(member_net_consumption[i, t] for i in members)
    = community_import[t] - community_export[t]
```

This expression is illustrative. The exact equation, variable definitions, units, time indexing, and signs must follow the assignment and existing code.

## Optimization formulations

### R — Reference case

The reference formulation should follow the assignment's definition exactly and reuse existing consumer-level optimization logic where appropriate. Its exact relationship to the other cases must be taken from the assignment specification.

### I — Individual contracts

Members are optimized individually according to the assignment's individual-contract rules. Member-level decisions and results remain distinct. Centralized coordination constraints should not be added unless required by the specification.

### C — Centralized community dispatch

The centralized formulation is a joint optimization over all members. Consistent with the assignment description, the community manager knows the members' preferences and constraints and dispatches the members centrally.

The community-level hourly balance is part of the same optimization problem as the member decisions. Formulation C must not be implemented by optimizing each member independently and summing the resulting decisions, because that would omit joint coordination through the shared community interface.

If the assignment defines an internal uniform hourly price as the dual variable of the community balance, the implementation must preserve an appropriate balance constraint and verify that the chosen solver and formulation support retrieving its dual value.

## Model organization

A single parameterized optimization class may be suitable for R, I, and C, provided the formulation-specific logic remains understandable and the assignment's interfaces are respected. A conceptual API could look like:

```python
class CommunityOptimizationModel:
    def __init__(self, community, formulation, ...):
        ...

    def build_model(self):
        ...

    def solve(self):
        ...

    def extract_results(self):
        ...
```

This is a conceptual example, not an established repository API. Depending on the existing code, the implementation may instead use separate formulation classes or shared helper functions. The important design goal is to reuse common data handling and model components without obscuring the differences between formulations.

## Potential later extensions: DP and SDP

Later assignment questions may require dynamic pricing (DP) and stochastic dynamic pricing (SDP). These formulations may need additional structures for price decisions or responses, uncertainty, scenarios, probabilities, and any required non-anticipativity constraints.

Their exact mathematical requirements and class boundaries must be derived from the assignment specification. No particular scenario structure, pricing rule, or inheritance hierarchy is assumed here.

## Expected implementation workflow

The implementation flow is conceptually:

1. Load the community with the existing loader.
2. Inspect and validate member data, shared contract/interface data, time horizon, dimensions, and units.
3. Pass the same underlying community/member data to the selected formulation.
4. Construct a fresh optimization model.
5. Solve and verify solver status before reading solution values or duals.
6. Extract results in a consistent, documented structure.
7. Validate relevant constraints and compare results according to the assignment's requirements.

A conceptual usage example is:

```python
community = load_community("base", day=day)

model = CommunityOptimizationModel(
    community=community,
    formulation="C",
    # other parameters required by the actual project
)

model.build_model()
model.solve()
results = model.extract_results()
```

The exact call signature depends on the actual repository and should not be assumed without inspection.

## Validation expectations

Relevant checks include:

- Solver status is valid before solution values are accessed.
- Member and time dimensions match the loaded data.
- The centralized community balance is satisfied within a documented numerical tolerance.
- Import/export, generation, and consumption use consistent units and sign conventions.
- Objective terms and contract limits match the assignment.
- Dual values are extracted only when supported by the solver and model formulation.
- Shared input data are not unintentionally changed between runs.
- Result fields are comparable across formulations where the assignment requires comparison.

## Repository-specific details that remain to be established

The following are intentionally unspecified because they depend on the actual assignment and starter code:

- The exact return type and fields of `load_community(...)`.
- Whether a community wrapper already exists or is needed.
- The representation of the shared contract and community connection point.
- The precise sign convention for net consumption, import, and export.
- The solver and modelling framework, including support for dual values.
- The exact distinction between R and I.
- Required result fields and comparison metrics.
- The detailed mathematical requirements for DP and SDP.

These details should be resolved from the assignment and repository rather than inferred from this architecture overview.
