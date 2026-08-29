"""Splice-to-fixpoint reduction of the Mondo label DAG (exp 0109).

WHY THIS EXISTS — the 763 degenerate nodes of exp 0104
------------------------------------------------------
The whole-Mondo readout banner reads ``3057 fittable nodes, 763 degenerate
(constant fallback)`` at C=3,820, and the degenerate set is exactly
``{root} ∪ {class nodes with exactly one kept child}``. The mechanism is
structural, not statistical:

  1. `anchor_hierarchy.reduce_to_anchor_hierarchy` gives every kept node its
     *nearest* superset cover as its single parent. Mondo is a multi-axis DAG, so
     two class nodes routinely have OVERLAPPING (not nested) terminal covers —
     say A={t1,t2,t3}, B={t1,t2}, C={t2,t3}. Each terminal is then STOLEN by
     whichever cover is smallest-and-first, and C is left holding just {t3}. The
     flattening turns the DAG into a tree and leaves behind class nodes with a
     single kept child.
  2. Under ``label_mask_mode: closure`` a parent is OBSERVED only on rows inside
     its own closure. A class node with one kept child is observed on exactly the
     rows that child (and its subtree) covers — so within its observed train set
     it is constant-1, its per-node readout cell is single-class, and the head
     falls back to a constant column.
  3. Those constant columns then poison the case-vs-background detection metric
     (a per-doc max over a column that never varies pins detection AUC at
     0.5000 — visible in every 0104 readout).

A class node that never discriminates anything its child does not already
discriminate is not a label; it is a rung in a ladder. This module removes the
rungs: repeatedly SPLICE OUT every class node with exactly one kept child
(connecting its parents straight to that child) and DROP every class node left
with none, iterating to fixpoint so a whole chain of single-child classes
collapses in one go. Terminals — the powered OMOP anchors, the things patients
actually attest — are never touched, so the corpus's label content is unchanged;
only the abstract scaffolding between them shrinks. The root stays even though it
remains degenerate: it is one node, and it is what makes the forest connected.

WHY IT IS A SEPARATE MODULE (and not a few lines inside `mondo_dag`)
-------------------------------------------------------------------
`_case_finding_cache.compute_bundle_cache_key` folds
``_module_source_hash(mondo_dag)`` into every Mondo bundle key — the
auto-invalidation discipline that guarantees a hierarchy edit can never be served
from a stale cache. That is exactly right, and it also means ANY edit to
`mondo_dag.py` — a comment, a docstring — moves every Mondo key and orphans every
cached bundle in every bucket, including exp 0104's record run (~20 min of
BigQuery per rebuild, mid-flight as this lands). The reduction therefore lives
here, `mondo_dag.py` is untouched, and the key folds this module ONLY when the
collapse is switched on. Collapse OFF ⇒ byte-identical keys to today.

Everything here is pure integer-id graph work over the id space `mondo_dag`
documents: terminals are POSITIVE OMOP standard-condition concept ids, class
nodes are SYNTHETIC NEGATIVES, and the forest root is ``-1``. That sign
convention IS the terminal test (overridable for tests / other id spaces).
"""
from __future__ import annotations

# Bumped when the reduction's OUTPUT would change for the same inputs. It is
# folded into the bundle cache key alongside this module's source hash: the hash
# is the automatic guard (no one has to remember), the version string is the
# human-readable record of WHICH reduction a cached bundle was built under, which
# is what an experiment doc can cite.
DAG_COLLAPSE_VERSION = "splice-fixpoint-v1"


def _default_is_terminal(cid) -> bool:
    """Terminals are the powered OMOP anchors: positive concept ids (`mondo_dag`'s
    id-space contract). Class nodes and the root are synthetic negatives."""
    return cid > 0


def _children_map(parents, nodes) -> dict:
    """``{node: sorted kept children}`` over `nodes`, from a ``{child: [parents]}``
    map. Parents outside `nodes` are ignored, so this is safe to call mid-collapse."""
    children = {n: set() for n in nodes}
    for child, ps in parents.items():
        if child not in nodes:
            continue
        for p in ps:
            if p in children:
                children[p].add(child)
    return {n: sorted(cs) for n, cs in children.items()}


def _nearest_surviving_ancestors(parents, node, keep, root) -> set:
    """The kept nodes `node` rewires up to, walking transitively past dropped ones.

    Deliberately the same walk `condition_dag.prune_by_attestation` uses for its
    own drops, so a spliced chain reattaches by exactly the rule the rest of the
    pipeline already reattaches by. A node whose every ancestor was dropped lands
    on the root, which keeps the forest connected."""
    surviving, seen, stack = set(), set(), list(parents.get(node, []))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        if p in keep:
            surviving.add(p)
        else:
            stack.extend(parents.get(p, []))
    return surviving or {root}


def collapse_only_child_classes(parent_of, root, *, is_terminal=None):
    """Splice-to-fixpoint over a ``{child: [parents]}`` map. Returns
    ``(new_parent_of, stats)``.

    One PASS collects every class node that is currently an only-child parent
    (exactly one kept child ⇒ SPLICE) or a childless class (zero kept children ⇒
    DROP), then removes them all at once by rewiring every survivor to its nearest
    surviving ancestors. Batching is what makes a chain A→B→C→terminal collapse in
    a single pass: B and C both qualify, and the terminal's nearest surviving
    ancestor — walking up past C and B — is A.

    A pass can still CREATE new only-children (two sibling chains that collapse
    onto the same terminal leave their shared parent with one child; dropping a
    childless class can leave its parent with one), so passes repeat until a pass
    finds nothing. That is the fixpoint, and it is why the predicted residual
    only-child count below is zero rather than "smaller".

    Never removed: the root (structural — it is what roots the forest, and it
    stays degenerate by design) and terminals (the powered anchors ARE the labels;
    a terminal with one child is a real, attestable disease node).
    """
    is_terminal = is_terminal or _default_is_terminal
    parents = {c: sorted(set(ps)) for c, ps in parent_of.items() if c != root}
    nodes = {root} | set(parents)

    def _is_class(n):
        return n != root and not is_terminal(n)

    n_before = len(nodes)
    n_classes_before = sum(1 for n in nodes if _is_class(n))
    spliced = dropped = passes = 0
    while True:
        children = _children_map(parents, nodes)
        # Sorted so the counts (and any future per-node logging) are deterministic
        # across runs; the removal itself is set-based and order-independent.
        only_child = [n for n in sorted(nodes)
                      if _is_class(n) and len(children.get(n, ())) == 1]
        childless = [n for n in sorted(nodes)
                     if _is_class(n) and not children.get(n)]
        doomed = set(only_child) | set(childless)
        if not doomed:
            break
        passes += 1
        spliced += len(only_child)
        dropped += len(childless)
        keep = nodes - doomed
        parents = {c: sorted(_nearest_surviving_ancestors(parents, c, keep, root))
                   for c in keep if c != root}
        nodes = keep

    # The prediction the diagnostic line publishes, computed rather than asserted:
    # after the fixpoint the only degenerate label cell left should be the root.
    children = _children_map(parents, nodes)
    residual = [n for n in sorted(nodes)
                if _is_class(n) and len(children.get(n, ())) == 1]
    stats = {
        "version": DAG_COLLAPSE_VERSION,
        "passes": passes,
        "spliced": spliced,
        "dropped_childless": dropped,
        "n_nodes_before": n_before,
        "n_nodes_after": len(nodes),
        "n_classes_before": n_classes_before,
        "n_classes_after": sum(1 for n in nodes if _is_class(n)),
        "n_terminals": sum(1 for n in nodes if is_terminal(n)),
        "residual_only_children": len(residual),
        # {root} ∪ {remaining only-children}: the exact degenerate set exp 0104
        # measured, re-predicted for the collapsed DAG. Should be 1.
        "predicted_degenerate": 1 + len(residual),
    }
    return parents, stats


def collapse_engine_dag(dag, *, is_terminal=None):
    """`collapse_only_child_classes` applied to a `ConditionDag`. Returns
    ``(collapsed_dag, stats)``.

    Names are carried through for the survivors (they are what the manifest's
    `name_by_id` and every per-node report read). `orphans` is deliberately NOT
    carried: post-collapse, "attached to the root" no longer means "its parent was
    filtered out upstream", and `prune_by_attestation` drops the attribute for the
    same reason."""
    from charmpheno.omop.condition_dag import ConditionDag

    parents, stats = collapse_only_child_classes(
        dag.parents, dag.anchor, is_terminal=is_terminal)
    nodes = {dag.anchor} | set(parents)
    names = {c: n for c, n in dag.names.items() if c in nodes}
    return ConditionDag(parents, dag.anchor, names), stats


def format_collapse_report(stats) -> str:
    """The one-line DAG-build diagnostic (exp 0109 spec item 4): what the reduction
    removed and what it predicts about the residual degenerate count, so the number
    is on the record BEFORE the readout's own banner confirms or refutes it."""
    return (
        f"[mondo]   dag-collapse ({stats['version']}): spliced {stats['spliced']} "
        f"only-child class node(s), dropped {stats['dropped_childless']} childless, "
        f"in {stats['passes']} pass(es); nodes {stats['n_nodes_before']} -> "
        f"{stats['n_nodes_after']} (classes {stats['n_classes_before']} -> "
        f"{stats['n_classes_after']}, terminals {stats['n_terminals']} unchanged); "
        f"predicted residual degenerate = {stats['predicted_degenerate']} "
        f"(root + {stats['residual_only_children']} remaining only-child class)")
