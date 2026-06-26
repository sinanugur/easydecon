# Marker roles and phase-specific routing

Marker roles let one marker table express different purposes for Phase 1 and
Phase 2. The supported roles are:

`positive`
: A normal positive marker. Missing or blank roles become `positive`.

`negative`
: An anti-marker. Only UCell-like Phase 2 scoring interprets negative markers.

`presence`
: A Phase 1 marker used to detect whether a group is plausible.

`identity`
: A Phase 2 marker used to distinguish a group from others.

Unknown role values raise an error. Negative direction is determined by
`marker_role`, not by the sign of `logfoldchanges`. Signed values remain signed
when they come from signed Scanpy inference.

## Routing modes

`marker_roles="shared"`
: The default. Without a role column, the same marker table is used for Phase 1
  and Phase 2. With a role column, role-aware routing is applied.

`marker_roles="phase_specific"`
: Phase 1 and Phase 2 use role-selected subsets. This is generated
  automatically only by reference-profile markers, unless you provide a manual
  role table.

## Shared mode routing

| Destination | Roles used |
| --- | --- |
| Phase 1 | `positive`, `presence`, `identity` |
| UCell-like Phase 2 | `positive`, `identity`, `negative` |
| Non-UCell Phase 2 | `positive`, `presence`, `identity` |

## Phase-specific mode routing

| Destination | Roles used |
| --- | --- |
| Phase 1 | `presence` |
| UCell-like Phase 2 | `positive`, `identity`, `negative` |
| Non-UCell Phase 2 | `positive`, `identity` |

For `marker_roles="phase_specific"`, easydecon raises if no Phase 1 presence
markers remain when Phase 1 is required, or if no Phase 2 identity-compatible
markers remain.

## Role generation

Reference-profile marker generation can produce `presence`, `identity`, and
`negative` roles with `marker_roles="phase_specific"`.

Scanpy and PyDESeq2 do not automatically generate phase-specific
presence/identity roles. Provide a manual `marker_role` table if you need those
roles with Scanpy or PyDESeq2 markers.

`marker_role_inference="scanpy_signed"` is a separate shared-mode feature for
Scanpy-style signed rows. It creates only `positive` and `negative` roles and
is useful with `method="ucell"`. It does not create phase-specific
`presence`/`identity` roles.

## Top-N with roles

When roles are present, `top_n_genes` is applied per group and role during
standardization or phase routing. This means positive and negative markers can
both retain their own top rows.
