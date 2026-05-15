// Zero-delay tooltip overlay.
//
// The browser-native tooltip from `title=` (and the SVG `<title>` child
// element) has a ~700ms hover delay that can't be tuned. This module
// hijacks `title` attributes and surfaces them through our own overlay
// element, with no delay.
//
// Install once at app start (App.svelte onMount). The handler walks up the
// DOM from the hover target looking for a `title` or `data-tip` attribute,
// or (for SVG) a child `<title>` element. The first time a `title` is
// encountered, it is moved to `data-tip` so the native tooltip won't also
// fire on top of ours.

let tipEl: HTMLDivElement | null = null
let currentTarget: Element | null = null

function ensureTipEl(): HTMLDivElement {
  if (tipEl) return tipEl
  const el = document.createElement('div')
  el.className = 'app-tip'
  el.setAttribute('role', 'tooltip')
  document.body.appendChild(el)
  tipEl = el
  return el
}

interface TipHit { el: Element; text: string }

function getTipText(target: Element | null): TipHit | null {
  let node: Element | null = target
  while (node) {
    const fromData = node.getAttribute('data-tip')
    if (fromData) return { el: node, text: fromData }
    const t = node.getAttribute('title')
    if (t) {
      node.setAttribute('data-tip', t)
      node.removeAttribute('title')
      return { el: node, text: t }
    }
    // SVG tooltip — a child <title> element. Promote it to data-tip on the
    // owning element so the browser-native tooltip also stops firing.
    if (node.namespaceURI === 'http://www.w3.org/2000/svg') {
      const childTitle = Array.from(node.children).find(
        (c) => c.tagName.toLowerCase() === 'title',
      )
      if (childTitle && childTitle.textContent) {
        node.setAttribute('data-tip', childTitle.textContent)
        childTitle.remove()
        return { el: node, text: node.getAttribute('data-tip')! }
      }
    }
    node = node.parentElement
  }
  return null
}

function show(target: Element, text: string) {
  const el = ensureTipEl()
  el.textContent = text
  el.classList.add('visible')
  const r = target.getBoundingClientRect()
  const tw = el.offsetWidth
  const th = el.offsetHeight
  let x = r.left + r.width / 2 - tw / 2
  let y = r.top - th - 8
  if (y < 4) y = r.bottom + 8
  x = Math.max(4, Math.min(window.innerWidth - tw - 4, x))
  el.style.left = x + 'px'
  el.style.top = y + 'px'
}

function hide() {
  if (!tipEl) return
  tipEl.classList.remove('visible')
  currentTarget = null
}

let installed = false

export function installTooltips(): void {
  if (installed) return
  installed = true
  document.addEventListener('mouseover', (e) => {
    const hit = getTipText(e.target as Element)
    if (!hit) {
      if (currentTarget) hide()
      return
    }
    if (hit.el !== currentTarget) {
      currentTarget = hit.el
      show(hit.el, hit.text)
    }
  })
  window.addEventListener('scroll', hide, true)
  window.addEventListener('blur', hide)
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') hide() })
}
