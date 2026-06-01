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
import { go, type Route } from './router'

type Placement =
  | 'top' | 'bottom' | 'left' | 'right'
  | 'top-start' | 'top-end' | 'bottom-start' | 'bottom-end'
  | 'left-start' | 'left-end' | 'right-start' | 'right-end'

interface StepDef {
  /** Step id; must match a key under copy.tour.basic / .advanced. */
  id: string
  /** Tab to navigate to before showing this step. Omit to stay put. */
  tab?: Route
  /** Element the popover attaches to. Omit for a centered, unattached step. */
  selector?: string
  /** Which side of the target the popover sits on. */
  on?: Placement
}

// ── Anchoring tables ────────────────────────────────────────────────────
// Each entry pairs a copy id with where it points. The conceptual arc:
// phenotypes → patients → simulate, ending on the view toggle.
const BASIC_STEPS: StepDef[] = [
  { id: 'welcome' /* centered */ },
  // Cohort selector lives in the masthead (present on every tab), no nav.
  { id: 'cohort', selector: '[data-tour="cohort"]', on: 'bottom' },
  { id: 'atlasMap', tab: 'atlas', selector: '[data-tour="atlas-map"]', on: 'right' },
  { id: 'findCondition', tab: 'atlas', selector: '[data-tour="find-condition"]', on: 'bottom' },
  { id: 'atlasDetail', tab: 'atlas', selector: '[data-tour="phenotype-detail"]', on: 'left' },
  // The find-in-patients / open-in-atlas pair bookends the patient section,
  // showing the two atlases are linked views of the same model.
  { id: 'findInPatients', tab: 'atlas', selector: '[data-tour="find-in-patients"]', on: 'left' },
  { id: 'patientMap', tab: 'patient', selector: '[data-tour="patient-map"]', on: 'right' },
  { id: 'patientProfile', tab: 'patient', selector: '[data-tour="patient-profile"]', on: 'left' },
  { id: 'openInAtlas', tab: 'patient', selector: '[data-tour="open-in-atlas"]', on: 'left' },
  { id: 'simulator', tab: 'simulator', selector: '[data-tour="simulator-input"]', on: 'right' },
  // Toggle lives in the masthead (present on every tab), so no navigation.
  { id: 'viewToggle', selector: '[data-tour="view-toggle"]', on: 'bottom' },
]

// Advanced mode reveals the model internals; this tour explains the things
// that only exist there. Runs entirely on the Atlas tab except the last stop.
const ADVANCED_STEPS: StepDef[] = [
  { id: 'welcome' /* centered */ },
  { id: 'metrics', tab: 'atlas', selector: '[data-tour="metrics"]', on: 'bottom' },
  { id: 'detailStats', tab: 'atlas', selector: '[data-tour="detail-stats"]', on: 'left' },
  { id: 'histogram', tab: 'atlas', selector: '[data-tour="histogram"]', on: 'left' },
  { id: 'relevance', tab: 'atlas', selector: '[data-tour="relevance"]', on: 'left' },
  // Quality grades live per-bubble and aren't guaranteed on screen, so point
  // at the atlas itself (top-right) and explain the grades in the copy.
  { id: 'quality', tab: 'atlas', selector: '[data-tour="atlas-map"]', on: 'right-start' },
  { id: 'simulator', tab: 'simulator', selector: '[data-tour="sim-controls"]', on: 'right' },
]

// Resolve the target element after (optionally) switching tabs. Polls a few
// frames so a freshly-mounted tab's DOM has time to appear; resolves anyway
// after `timeout` so a missing anchor degrades to a centered popover.
function ready(def: StepDef, timeout = 3000): Promise<void> {
  if (def.tab) go(def.tab)
  return new Promise((resolve) => {
    if (!def.selector) {
      // No anchor: just wait one frame for any tab swap to paint.
      requestAnimationFrame(() => resolve())
      return
    }
    const start = performance.now()
    const tick = () => {
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

    // Does this step move to a different tab than the one before it? If so
    // it's the step that establishes a new screen, and we scroll the page to
    // the top instead of centering the feature — otherwise centering a tall
    // map scrolls the nav (and its fresh tab highlight) off the top, hiding
    // the very cue we just lit up. The tabs sit directly above the features,
    // so scroll-to-top fits the tab, the feature, and the popover together.
    const entersNewTab = !!def.tab && def.tab !== defs[i - 1]?.tab

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
