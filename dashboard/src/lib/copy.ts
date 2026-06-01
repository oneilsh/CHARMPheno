/**
 * Centralized user-facing copy for the dashboard.
 *
 * Single source of truth for explanatory prose: tab intros, the "what is
 * this?" popovers, conceptual tooltips, section subheads, empty states, and
 * chart axis labels. Edit a string here and it updates live everywhere that
 * string is used — no need to hunt through components.
 *
 * Conventions
 *  - Plain strings for static copy.
 *  - Functions for copy that interpolates a live value (the τ threshold, a
 *    sample count, an entity name). The formatting lives inside the function
 *    so this file owns the whole sentence, percent signs and all.
 *  - `string[]` for multi-paragraph popovers. Those paragraphs may contain
 *    inline <em>/<strong> emphasis and are rendered with {@html}; the content
 *    is author-controlled (never user input), so that is safe here.
 *
 * Deliberately NOT centralized (still inline in their components): one-to-
 * three-word section labels / eyebrows, button glyphs, input placeholders,
 * aria-labels, transient status text ("computing layout"), and the d3 hover
 * read-outs that merely assemble metric values (the bubble/dot tooltips).
 * Ask if you'd like any of those pulled in here too.
 */

// Shared formatter: τ (a fraction in [0,1]) rendered as a whole percent.
const pct = (tau: number): string => (tau * 100).toFixed(0)

export const copy = {
  // ── Masthead ──────────────────────────────────────────────────────────
  masthead: {
    subtitle: `exploring latent phenotypes`,
    // Model-size readout (K/V/n), shown in the masthead beside the cohort
    // selector in advanced mode.
    meta: {
      k: `K: the number of phenotypes (topics) the model was asked to learn from the dataset.`,
      v: `V: distinct conditions displayed in the dashboard, over total distinct conditions in the source dataset. Low-count conditions are suppressed for patient privacy.`,
      n: `n: number of patient records the model was fit on (in thousands).`,
    },
  },

  // ── Phenotype Atlas tab ───────────────────────────────────────────────
  atlas: {
    title: `Phenotype Atlas`,
    kicker: `Each marker is a learned phenotype. Bubbles that sit closer together share more of their leading conditions; bubble size shows how widely the phenotype shows up across patients.`,
    whatIsSummary: `What's a phenotype?`,
    // kLabel: the phenotype count K (or a "~80" fallback while loading).
    whatIs: (kLabel: string | number): string[] => [
      `A <em>phenotype</em> here is a recurring pattern of clinical conditions that tends to appear together across patients. For example, "Type 2 diabetes care" concentrates on diabetes, retinopathy, neuropathy, and related conditions.`,
      `These phenotypes were learned automatically from de-identified patient records using a topic model (Latent Dirichlet Allocation). The model didn't know about diseases ahead of time; it just looked for groups of conditions that tend to co-occur, and produced ${kLabel} phenotypes.`,
      `A patient is a mix of phenotypes, not a single one. A phenotype is not a diagnosis; it's a pattern. Some patterns name a single disease, others name a family of related conditions, and some describe broad health backgrounds (e.g. chronic comorbidity follow-up).`,
    ],
    legend: {
      coherence: `Coherence: how reliably the phenotype's leading conditions actually co-occur in the same patients. Higher means the conditions really do show up together; lower means the pattern is weaker or more diffuse. (Bubble color encodes this.)`,
      prevalence: (tau: number): string =>
        `Prevalence: estimated share of patients for whom this phenotype makes up at least ${pct(tau)}% (τ) of their coded activity. Bubble size scales with this share.`,
      topicMass: `Topic mass: mean topic mixture share across patients (doc-mean of θ). Sums to 100% across phenotypes; not a patient count.`,
    },
  },

  // ── Phenotype detail (CodePanel) ──────────────────────────────────────
  phenotypeDetail: {
    empty: `Select a phenotype on the map to read its top conditions.`,
    findInPatientsTip: `Switch to the Patient Atlas with patients carrying this phenotype highlighted`,
    prevalence: {
      labelBasic: `Prevalence`,
      labelAdvanced: `Prevalence (patients)`,
      tipAdvanced: (tau: number): string =>
        `Fraction of patients with θ > τ = ${tau.toFixed(2)} — at least ${pct(tau)}% of their coded activity is attributed to this phenotype. A threshold-based approximation of clinical prevalence in the cohort.`,
      tipBasic: (tau: number): string =>
        `Fraction of patients for whom this phenotype makes up at least ${pct(tau)}% of their coded activity.`,
      tipNoHistogram: `Topic mass (no patient distribution available for this bundle).`,
    },
    topicMassTip: `Mean topic mixture share across patients (doc-mean of θ). Sums to 100% across phenotypes; not a patient count.`,
    coherenceTip: `Coherence: how reliably this phenotype's leading conditions co-occur in real patients (NPMI: normalized pointwise mutual information). Higher means the conditions really do show up together.`,
    pairCoverageTip: `Pair coverage: fraction of the leading-condition pairs that had enough joint observations to actually contribute to the coherence number. Low coverage means the coherence value was computed on only a few pairs and is less trustworthy.`,
    sourceTip: `Source #: the raw topic index from the LDA fit before sorting. Useful for cross-referencing the underlying model.`,
    quality: {
      phenotype: `Quality: phenotype. A clinically coherent pattern of conditions; the cluster names a recognisable disease or related family.`,
      background: `Quality: background. Broad, non-specific health-care activity (general checkups, common comorbidities). Real signal but not disease-specific.`,
      anchor: `Quality: anchor. Dominated by one or two very common conditions; useful as a reference point but lower information content than a fuller phenotype.`,
      mixed: `Quality: mixed. The leading conditions span multiple unrelated clinical areas; the topic merged what should probably be separate phenotypes.`,
      dead: `Quality: dead. Minimal usage by the model and very small divergence from the dataset average; this topic slot was effectively unused. (Hidden in basic mode.)`,
    },
    histogram: {
      title: `Phenotype Prominence`,
      tip: `How prominently this phenotype features in each patient's mixture, across the cohort. The x-axis is the share of a patient's coded activity attributed to this phenotype (shown from τ upward); the y-axis is the share of patients at each level. Patients below τ are summarised by the '< τ' figure rather than drawn. Bins with fewer than 20 patients are suppressed for privacy.`,
      belowTauTip: `Share of patients for whom this phenotype is below the τ threshold — i.e. they are not counted as having it. Not drawn on the chart (the x-axis starts at τ).`,
    },
    relevance: {
      weightingTip: `Relevance term weighting. The slider blends two views of 'top conditions': raw frequency (how much of the phenotype's mass falls on this condition) and lift (how much more this condition shows up here than in the overall dataset). Slide left for surprise/lift, right for sheer frequency.`,
      liftEndTip: `Lift: how much more this condition appears in this phenotype than across all patients overall. Surfaces rare-but-concentrated conditions.`,
      freqEndTip: `Frequency: the condition's raw probability under this phenotype.`,
      colTipAdvanced: `Relevance: λ·log p(w|k) + (1−λ)·log lift. The slider tunes how much frequency vs lift the ranking favors. Bar shows raw frequency p(w|k); number is its percentage.`,
      colTipBasic: `Relevance: the leading conditions for this phenotype, ranked by a balance of how often they appear here AND how distinctive they are to this phenotype. The bar shows the condition's share of this phenotype.`,
    },
  },

  // ── Prevalence histogram (chart-internal labels) ──────────────────────
  histogram: {
    ariaLabel: `Phenotype prominence distribution histogram`,
    axisX: `% of a patient's coded activity`,
    axisY: `% of patients`,
    suppressedTip: `< 20 patients (suppressed for privacy)`,
  },

  // ── Phenotype browser (table) ─────────────────────────────────────────
  phenotypeBrowser: {
    topicMassTip: `Mean topic mixture share (doc-mean of θ). Sums to 100% across phenotypes.`,
    prevTipAdvanced: (tau: number): string =>
      `Fraction of patients with mixture weight above τ = ${tau.toFixed(2)} (at least ${pct(tau)}% of their coded activity).`,
    prevTipBasic: (tau: number): string =>
      `Fraction of patients for whom this phenotype makes up at least ${pct(tau)}% of their coded activity.`,
  },

  // ── Condition search ──────────────────────────────────────────────────
  conditionSearch: {
    // entityLabel is "phenotypes" on the atlas, "patients" on the patient tab.
    activeChipTip: (entityLabel: string): string =>
      `${entityLabel.charAt(0).toUpperCase() + entityLabel.slice(1)} containing this condition are highlighted`,
  },

  // ── Patient Atlas tab ─────────────────────────────────────────────────
  patient: {
    title: `Patient Atlas`,
    kicker: `Each dot is a synthetic patient drawn from the model. Patients near each other share a similar mix of phenotypes.`,
    whatIsSummary: `What's a synthetic patient?`,
    whatIs: [
      `<strong>These are not real people.</strong> Each "patient" here is generated by sampling from the phenotype model: first a mix of phenotypes (the <em>profile</em>), then a bag of conditions consistent with that mix.`,
      `The conditions a patient is shown with are <em>unordered</em> — they're the set of things that happened in this patient's record over a time period, not a timeline. Counts are how often a code appeared in that record.`,
      `Patients sit in the same 2D space as the phenotype atlas. A patient concentrated on one phenotype lands near that phenotype's bubble; a patient mixing several lands between them. Click a dot to inspect one.`,
    ],
    profileSub: `Phenotype profile · click a band below to see which of this patient's codes contributed to it.`,
    findPhenotypeChipTip: `Patients carrying this phenotype are ringed in amber on the atlas and have an amber right-band in the table`,
    empty: `Click a patient on the atlas to see their details.`,
  },

  // ── Patient atlas map (legend) ────────────────────────────────────────
  patientMap: {
    legendTip: `Each dot is one synthetic patient. Position comes from a 2D UMAP of patient phenotype mixes using cosine distance. Patients near each other have similar phenotype profiles. Dot color matches the dominant phenotype's color in this patient's profile bar.`,
    legendNote: `Patients with mixed or unclear phenotypes are excluded in basic view, toggle advanced mode to see all patients.`,
  },

  // ── Contributing-codes panel ──────────────────────────────────────────
  contributingCodes: {
    heading: `Top contributing codes`,
    openInAtlasTip: `Switch to the Phenotype Atlas with this phenotype selected`,
    otherLabel: `Other / tail phenotypes`,
    subOther: `Codes from this patient's record that the model attributes to phenotypes outside this patient's dominant mix, ordered by tail-responsibility.`,
    subMatch: `Codes from this patient's record that match this phenotype, ordered by occurrence count.`,
    hintNoSelection: `Click a phenotype band above to see which codes from this patient's record drove the assignment.`,
    hintNoCodes: (label: string): string =>
      `No codes from this patient's record contribute to ${label}.`,
  },

  // ── Simulator tab ─────────────────────────────────────────────────────
  simulator: {
    title: `Simulator`,
    kicker: `Pick some starting conditions and the model will tell you what kind of patient this looks like and what else would round out their year.`,
    whatIsSummary: `What is this?`,
    whatIs: [
      `The simulator asks the model: <em>given these conditions, what kind of patient could this be?</em> It answers by drawing many possible complete year-of-life records from the model's distribution.`,
      `Each draw is one plausible patient. The <strong>profile bar</strong> shows the average phenotype mix across those draws. The <strong>expected codes</strong> table shows what the model thinks fills in the rest of the year. The <strong>atlas</strong> shows where these patients land relative to the synthetic cohort - tight cluster means the conditions you gave nail one kind of patient, smeared cloud means they're consistent with several.`,
      `Start by clicking conditions on the left, or just hit Simulate to draw new patients from scratch.`,
    ],
    runSub: `Draw a batch of plausible patients from the conditions above.`,
    autoregressiveTip: `When on, the model re-evaluates the phenotype mix after every drawn code so each token shifts the next one's distribution.`,
    phenotypeMixHeading: `This patient is a mix of…`,
    phenotypeMixSub: (n: number): string => `Average across ${n} simulated draws.`,
    emptyFromScratch: `Add some starting conditions on the left (or just hit Simulate to draw patients from scratch), then click <strong>simulate →</strong> to see what kind of patient this looks like.`,
    emptyReady: (n: number): string =>
      `${n} starting condition${n === 1 ? '' : 's'} ready. Click <strong>simulate →</strong> to see what kind of patient this looks like.`,
  },

  // ── Simulator: per-sample structure plot ──────────────────────────────
  structurePlot: {
    heading: `How confident is the model?`,
    sub: `Each column is one simulated patient. A solid color block means the model agrees with itself across draws; a rainbow means the starting conditions are consistent with several phenotype mixes.`,
    hint: `Run the simulator to see the per-sample distribution.`,
  },

  // ── Simulator: predicted-record panel ─────────────────────────────────
  predictedRecord: {
    heading: `Model expects to also see`,
    sub: (n: number): string =>
      `Across ${n} simulated years, the model most often fills these in. The bar shows how often this condition shows up (P10 → P90, tick at median).`,
    hintEmpty: `Run the simulator to see what the model predicts.`,
    hintNone: `The model did not predict any additional conditions in this run.`,
  },

  // ── Simulator: mini atlas ─────────────────────────────────────────────
  simMiniMap: {
    sub: `Each bright dot is one simulated patient on the same atlas as the patient cohort.`,
  },

  // ── Simulator: conditions editor ──────────────────────────────────────
  conditionsEditor: {
    sub: `Conditions this patient already has. The simulator fills in the rest of their year.`,
    hint: `Each click draws one random condition from that phenotype's profile.`,
  },

  // ── Neighbor ribbon ───────────────────────────────────────────────────
  neighborRibbon: {
    heading: `Patients with similar profiles`,
  },

  // ── Guided tours (Shepherd) ───────────────────────────────────────────
  // Two short, skippable walkthroughs launched from the "Take the tour" link
  // beside the basic/advanced toggle. The link is context-sensitive: in basic
  // mode it runs `basic`, in advanced mode it runs `advanced`.
  //
  //  - basic: drives across all three tabs to narrate the conceptual arc
  //    (phenotypes → patients → simulate). Ends pointing at the view toggle,
  //    inviting the user to switch to advanced (which they do themselves —
  //    the tour never flips it) where the same link offers the advanced tour.
  //  - advanced: explains the model internals that only appear in advanced
  //    mode (the K/V/n metrics, the prominence histogram, the relevance
  //    slider, phenotype quality grades, the simulator's sampling controls).
  //
  // Anchoring (which element each step points at, which tab it lives on)
  // lives in tour.ts; the words live here, keyed by the same step id. `body`
  // is rendered as HTML by Shepherd, so inline <em>/<strong> is live markup.
  tour: {
    startLabel: `Take the tour`,
    nextLabel: `Next`,
    backLabel: `Back`,
    doneLabel: `Done`,
    basic: {
      welcome: {
        title: `Welcome to CHARMPheno`,
        body: `This dashboard explores <em>phenotypes</em> — recurring patterns of clinical conditions a model learned from de-identified patient records. The tour takes about a minute and visits all three tabs. You can leave any time with the × or the Esc key.`,
      },
      cohort: {
        title: `Pick a cohort`,
        body: `Each cohort is a separate model fit on a different slice of patients (different inclusion criteria or time window). Switch cohorts here at any time — everything below reloads for the model you choose.`,
      },
      atlasMap: {
        title: `The phenotype atlas`,
        body: `Every bubble is one learned phenotype. Bubbles that sit close together share their leading conditions; bigger bubbles show up in more patients. Click any bubble to inspect it.`,
      },
      findCondition: {
        title: `Find a condition`,
        body: `Search for a specific condition and the atlas highlights the phenotypes it features in — a quick way to ask "which patterns involve diabetes?" without hunting through bubbles.`,
      },
      atlasDetail: {
        title: `Inside a phenotype`,
        body: `The selected phenotype's leading conditions appear here, along with how prevalent it is and how reliably its conditions co-occur. Hover any label with a dotted underline or a <strong>?</strong> for a plain-language explanation.`,
      },
      findInPatients: {
        title: `Jump to the patients`,
        body: `<strong>Find in patients</strong> carries this phenotype over to the Patient Atlas and rings every synthetic patient who carries it — connecting "what is this pattern" to "who has it."`,
      },
      patientMap: {
        title: `Synthetic patients`,
        body: `Now the patients. <strong>These are not real people</strong> — each dot is a synthetic record the model generated. Patients near each other carry a similar mix of phenotypes, in the same 2D space as the atlas you just saw.`,
      },
      patientProfile: {
        title: `A patient is a mix`,
        body: `No patient is a single phenotype. This profile bar shows the selected patient's blend; click a band to see which of their conditions drove that part of the mix.`,
      },
      openInAtlas: {
        title: `…and back again`,
        body: `The link runs both ways: <strong>open in atlas</strong> takes whichever phenotype band you're inspecting back to the Phenotype Atlas, selected and ready. The two atlases are two views of the same model.`,
      },
      simulator: {
        title: `Ask the model "what if?"`,
        body: `Give the model some starting conditions here and hit <strong>simulate</strong>. It draws many plausible complete records and tells you what kind of patient this looks like and what else would tend to round out their year.`,
      },
      viewToggle: {
        title: `Want to go deeper?`,
        body: `That's the basics. Flip this to <strong>advanced</strong> whenever you're ready — it reveals the model's internals, and a second tour (same link) walks you through them. Explore away.`,
      },
    },
    advanced: {
      welcome: {
        title: `Advanced mode`,
        body: `You're now seeing the model's internals — extra metrics, every phenotype (including weak ones), and finer controls. This short tour explains what the added machinery means.`,
      },
      metrics: {
        title: `Model at a glance`,
        body: `<strong>K</strong> is how many phenotypes the model learned, <strong>V</strong> is conditions shown over total in the source data (rare ones are suppressed for privacy), and <strong>n</strong> is how many patient records it was fit on.`,
      },
      detailStats: {
        title: `Phenotype diagnostics`,
        body: `Advanced mode adds quality diagnostics for the selected phenotype: <em>topic mass</em> (its average share across patients), <em>coherence</em> (how tightly its leading conditions co-occur), and <em>pair coverage</em> (how much evidence that coherence rests on). Hover each for detail.`,
      },
      histogram: {
        title: `Phenotype prominence`,
        body: `This shows how prominently the selected phenotype features across patients: the x-axis is the share of a patient's coded activity it accounts for (from the τ threshold up), the y-axis the share of patients. Low-count bins are suppressed for privacy.`,
      },
      relevance: {
        title: `Re-rank the conditions`,
        body: `The leading-conditions list can be re-ranked between raw <em>frequency</em> (how much of the phenotype's mass falls here) and <em>lift</em> (how much more this condition appears here than across all patients). Slide left for surprising/distinctive conditions, right for sheer frequency.`,
      },
      quality: {
        title: `Every phenotype, graded`,
        body: `Advanced mode also shows the weaker bubbles basic mode hides. Each phenotype carries a <em>quality</em> grade — from a clean disease pattern down to <em>mixed</em> (merged areas) or <em>dead</em> (effectively unused) — so you can tell signal from noise.`,
      },
      simulator: {
        title: `Tune the simulation`,
        body: `Advanced mode exposes the sampling controls: more <strong>samples</strong> tighten the estimate, and <strong>autoregressive</strong> re-fits the phenotype mix after every drawn code. The structure plot below reads out how confident the model is across draws.`,
      },
    },
  },
}
