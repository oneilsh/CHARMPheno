# 0103 — Which-member skill is mostly the diagnosis in other words

**Setting:** 0137 fit; framing-1 survey over 243 distinct Mondo groups (2–25 members),
records with the members' mapped codes, exact-name-matched concepts and `[measured]`
removed; label-year single-member test truth; E3/E5/E6 engines (exp 0150).

## Observation

The groups where "which member?" works best are carried, per the code head's reasons,
by (1) synonym or parent diagnosis codes the code map does not attest to the member
("Viral hepatitis C", "Opioid abuse", "Secondary polycythemia", "Monoclonal
gammopathy"), (2) confirmatory test results (HBsAg positive), and (3) treatments
(entecavir, buprenorphine, lenalidomide). Framing-1 skill correlates with framing-2
gain (Spearman +0.54) but hardly with new-diagnosis AUC (+0.11 θ / +0.21 raw).

## Why it matters

- With label-year truth most scored patients are prevalent (insight 0100), so their
  pre-index record already documents the diagnosis. Removing the mapped codes and exact
  names is not enough; the diagnosis leaks through synonyms, tests and treatments.
- The honest framing-1/2 evaluation scores INCIDENT patients only (no member or group
  attestation anywhere before the index, E4 sidecar): their pre-index record is a
  pre-diagnosis record. Expect skill to fall; what remains is the early signal a
  clinician would actually have.
- The code-map gap is broader than the HCM / ancestor-group cases (0141): concept-level
  synonyms of the members are common and should be audited (SNOMED descendants and
  name variants of each member's mapped codes).
