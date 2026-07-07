// Guided tours (Shepherd).
//
// Two short walkthroughs launched from the "Take the tour" link beside the
// basic/advanced toggle (see App.svelte). The link is context-sensitive:
// basic mode runs the `basic` tour, advanced mode runs the `advanced` one.
//
// Separation of concerns:
//  - The WORDS live in copy.ts (copy.tour.basic / copy.tour.advanced), keyed
//    by the same step ids used here.
//  - The ANCHORING lives here: which tab a step belongs to, which
//    `data-tour="…"` element it points at, and where the popover sits.
// Adding or reordering a stop is therefore a one-line change in each file.
//
// Cross-tab: a step's `beforeShowPromise` navigates to its tab via the router
// and waits for the target element to mount before Shepherd positions the
// popover. If the element never appears (timeout), Shepherd falls back to a
// centered popover rather than hanging.

import Shepherd from 'shepherd.js'
import type { StepOptionsButton, Tour } from 'shepherd.js'
import 'shepherd.js/dist/css/shepherd.css'
import { copy } from './copy'
import { go, type TopId } from './router'

type Placement =
  | 'top' | 'bottom' | 'left' | 'right'
  | 'top-start' | 'top-end' | 'bottom-start' | 'bottom-end'
  | 'left-start' | 'left-end' | 'right-start' | 'right-end'

interface StepDef {
  /** Step id; must match a key under copy.tour.basic / .advanced. */
  id: string
  /** Top-level tab to navigate to before showing this step. Omit to stay put. */
  tab?: TopId
  /**
   * Subtab to select within `tab` before showing this step (e.g. the old
   * standalone Patient/Simulator tabs are now the Simulator top tab's two
   * subtabs). Omit to leave the subtab as-is / default.
   */
  sub?: string
  /** Element the popover attaches to. Omit for a centered, unattached step. */
  selector?: string
  /** Which side of the target the popover sits on. */
  on?: Placement
  /**
   * Optional side-effect fired once the step's tab has mounted, before the
   * anchor is polled — e.g. click "simulate" so the sample-mix / predicted
   * panels exist to point at. Best-effort: failures are swallowed so a step
   * still degrades to a centered popover.
   */
  before?: () => void
}

// Trigger a generation on the Simulate subtab so the post-generation panels
// (sample-mix, posterior-predictive) are present for their tour stops. The
// Simulate button lives in the run panel; clicking it runs the same code path
// as a user click. No-op (and harmless) if a result already exists.
function runSimulation(): void {
  const btn = document.querySelector<HTMLButtonElement>('[data-tour="sim-controls"] .run-btn')
  if (btn && !btn.disabled) btn.click()
}

// ── Anchoring tables ────────────────────────────────────────────────────
// Each entry pairs a copy id with where it points. The conceptual arc follows
// the app's tab order: Atlas (Explore → Compare) → Simulator (Simulate →
// Explore), ending on the view toggle.
const BASIC_STEPS: StepDef[] = [
  { id: 'welcome' /* centered */ },
  // Model selector lives in the masthead (present on every tab), no nav.
  { id: 'model', selector: '[data-tour="cohort"]', on: 'bottom' },
  // ── Phenotype Atlas · Explore ──
  { id: 'atlasMap', tab: 'atlas', sub: 'explore', selector: '[data-tour="atlas-map"]', on: 'right' },
  { id: 'atlasCovariates', tab: 'atlas', sub: 'explore', selector: '[data-tour="atlas-covariates"]', on: 'right' },
  { id: 'browse', tab: 'atlas', sub: 'explore', selector: '[data-tour="phenotype-browser"]', on: 'top' },
  { id: 'findCondition', tab: 'atlas', sub: 'explore', selector: '[data-tour="find-condition"]', on: 'bottom' },
  { id: 'atlasDetail', tab: 'atlas', sub: 'explore', selector: '[data-tour="phenotype-detail"]', on: 'left' },
  // ── Phenotype Atlas · Compare ──
  { id: 'compareHeatmap', tab: 'atlas', sub: 'compare', selector: '[data-tour="correlation-heatmap"]', on: 'right' },
  { id: 'compareDiff', tab: 'atlas', sub: 'compare', selector: '[data-tour="phenotype-difference"]', on: 'left' },
  // ── Simulator · Simulate ──
  { id: 'simRun', tab: 'sim', sub: 'simulate', selector: '[data-tour="sim-controls"]', on: 'right' },
  { id: 'simConditions', tab: 'sim', sub: 'simulate', selector: '[data-tour="sim-conditions"]', on: 'right' },
  { id: 'simConditioning', tab: 'sim', sub: 'simulate', selector: '[data-tour="sim-conditioning"]', on: 'right' },
  // Fire a generation so the post-run panels exist to point at.
  { id: 'sampleMix', tab: 'sim', sub: 'simulate', selector: '[data-tour="sample-mix"]', on: 'left', before: runSimulation },
  { id: 'predicted', tab: 'sim', sub: 'simulate', selector: '[data-tour="posterior-predictive"]', on: 'left' },
  // ── Simulator · Explore ──
  { id: 'patientMap', tab: 'sim', sub: 'explore', selector: '[data-tour="patient-map"]', on: 'right' },
  { id: 'patientProfile', tab: 'sim', sub: 'explore', selector: '[data-tour="patient-profile"]', on: 'left' },
  { id: 'contributingCodes', tab: 'sim', sub: 'explore', selector: '[data-tour="contributing-codes"]', on: 'left' },
  // Toggle lives in the masthead (present on every tab), so no navigation.
  { id: 'viewToggle', selector: '[data-tour="view-toggle"]', on: 'bottom' },
]

// Advanced mode reveals the model internals; this tour explains the diagnostics
// that only exist there. Runs on the Atlas Explore tab except the last stop.
const ADVANCED_STEPS: StepDef[] = [
  { id: 'welcome' /* centered */ },
  { id: 'metrics', tab: 'atlas', sub: 'explore', selector: '[data-tour="metrics"]', on: 'bottom' },
  { id: 'detailStats', tab: 'atlas', sub: 'explore', selector: '[data-tour="detail-stats"]', on: 'left' },
  { id: 'histogram', tab: 'atlas', sub: 'explore', selector: '[data-tour="histogram"]', on: 'left' },
  { id: 'relevance', tab: 'atlas', sub: 'explore', selector: '[data-tour="relevance"]', on: 'left' },
  // Quality grades live per-bubble and aren't guaranteed on screen, so point
  // at the atlas itself (top-right) and explain the grades in the copy.
  { id: 'quality', tab: 'atlas', sub: 'explore', selector: '[data-tour="atlas-map"]', on: 'right-start' },
  { id: 'simulator', tab: 'sim', sub: 'simulate', selector: '[data-tour="sim-controls"]', on: 'right' },
]

// Resolve the target element after (optionally) switching tabs. Polls a few
// frames so a freshly-mounted tab's DOM has time to appear; resolves anyway
// after `timeout` so a missing anchor degrades to a centered popover.
function ready(def: StepDef, timeout = 3000): Promise<void> {
  if (def.tab) go(def.tab, def.sub)
  const fireBefore = () => {
    if (def.before) {
      try { def.before() } catch { /* best-effort; degrade to centered popover */ }
    }
  }
  return new Promise((resolve) => {
    if (!def.selector) {
      // No anchor: wait one frame for any tab swap to paint, fire, then show.
      requestAnimationFrame(() => { fireBefore(); resolve() })
      return
    }
    const start = performance.now()
    let fired = false
    const tick = () => {
      // Fire the side-effect once, on the first tick — the step's tab has
      // navigated and (for same-tab steps) already painted, so the button the
      // side-effect clicks is present. Then poll for the anchor it produces.
      if (!fired) { fired = true; fireBefore() }
      if (document.querySelector(def.selector!) || performance.now() - start > timeout) {
        resolve()
        return
      }
      requestAnimationFrame(tick)
    }
    tick()
  })
}

function buildTour(mode: 'basic' | 'advanced'): Tour {
  const defs = mode === 'basic' ? BASIC_STEPS : ADVANCED_STEPS
  const words = mode === 'basic' ? copy.tour.basic : copy.tour.advanced

  const tour = new Shepherd.Tour({
    useModalOverlay: true,
    defaultStepOptions: {
      scrollTo: { behavior: 'smooth', block: 'center' },
      cancelIcon: { enabled: true },
      classes: 'charm-tour-step',
    },
  })

  defs.forEach((def, i) => {
    const isFirst = i === 0
    const isLast = i === defs.length - 1
    const buttons: StepOptionsButton[] = []
    if (!isFirst) {
      buttons.push({ text: copy.tour.backLabel, action: () => tour.back(), secondary: true })
    }
    buttons.push(
      isLast
        ? { text: copy.tour.doneLabel, action: () => tour.complete() }
        : { text: copy.tour.nextLabel, action: () => tour.next() },
    )

    // Does this step move to a different tab or subtab than the one before
    // it? If so it's the step that establishes a new screen, and we scroll
    // the page to the top instead of centering the feature — otherwise
    // centering a tall map scrolls the nav (and its fresh tab highlight) off
    // the top, hiding the very cue we just lit up. The tabs sit directly
    // above the features, so scroll-to-top fits the tab, the feature, and
    // the popover together. Subtab changes count too: the old standalone
    // Patient/Simulator tabs are now subtabs of the same "Simulator" top
    // tab, but switching between them is still a full screen swap.
    const prev = defs[i - 1]
    const entersNewTab = !!def.tab && (def.tab !== prev?.tab || def.sub !== prev?.sub)

    const stepCopy = (words as Record<string, { title: string; body: string }>)[def.id]
    tour.addStep({
      id: def.id,
      title: stepCopy.title,
      text: stepCopy.body,
      buttons,
      attachTo: def.selector ? { element: def.selector, on: def.on ?? 'bottom' } : undefined,
      // Also light up the nav tab this step lives on, so the overlay cuts a
      // hole around both the feature and its tab. When the tour jumps tabs
      // the highlight visibly leaps to the new tab — making it obvious the
      // feature lives on a different screen.
      extraHighlights: def.tab ? [`[data-tour="tab-${def.tab}"]`] : undefined,
      scrollTo: { behavior: 'smooth', block: 'center' },
      // Instant (not smooth): the destination tab may run heavy synchronous
      // work on mount (the Patient atlas fits a UMAP), which stalls a
      // main-thread smooth-scroll animation part-way. A jump is immune.
      scrollToHandler: entersNewTab
        ? () => window.scrollTo({ top: 0, behavior: 'auto' })
        : undefined,
      beforeShowPromise: () => ready(def),
    })
  })

  return tour
}

/** Launch a tour, cancelling any that's already running. */
export function startTour(mode: 'basic' | 'advanced' = 'basic'): void {
  Shepherd.activeTour?.cancel()
  buildTour(mode).start()
}
