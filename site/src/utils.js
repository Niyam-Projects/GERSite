/**
 * Discretize confidence to N steps for style cache efficiency.
 */
export function discretizeConf(conf) {
  if (conf == null || isNaN(conf)) return 'null'
  return Math.round(Math.max(0, Math.min(1, conf)) * 20)
}

// Confidence gradient stops: red → yellow → green
const CONF_STOPS = [
  { t: 0.0, r: 215, g: 48,  b: 39  },  // #d73027 red
  { t: 0.5, r: 254, g: 224, b: 139 },  // #fee08b yellow
  { t: 1.0, r: 26,  g: 152, b: 80  },  // #1a9850 green
]

function lerpChannel(a, b, t) {
  return Math.round(a + (b - a) * t)
}

/**
 * Map a confidence value [0,1] to a hex color via red→yellow→green gradient.
 */
export function confidenceColor(value) {
  if (value == null || isNaN(value)) return '#999999'
  const v = Math.max(0, Math.min(1, value))

  // Find the two surrounding stops
  let lo = CONF_STOPS[0]
  let hi = CONF_STOPS[CONF_STOPS.length - 1]
  for (let i = 0; i < CONF_STOPS.length - 1; i++) {
    if (v <= CONF_STOPS[i + 1].t) {
      lo = CONF_STOPS[i]
      hi = CONF_STOPS[i + 1]
      break
    }
  }

  const span = hi.t - lo.t
  const t = span === 0 ? 0 : (v - lo.t) / span
  const r = lerpChannel(lo.r, hi.r, t)
  const g = lerpChannel(lo.g, hi.g, t)
  const b = lerpChannel(lo.b, hi.b, t)
  return `rgb(${r},${g},${b})`
}
