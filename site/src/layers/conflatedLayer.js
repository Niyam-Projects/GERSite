import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { Style, Circle, Fill, Stroke } from 'ol/style'
import { confidenceColor, discretizeConf } from '../utils.js'
import { CONFLATED_PMTILES_URL } from '../constants.js'

let layer = null
let enabledLabels = null  // null = all on; otherwise Set<string>

const styleCache = {}

export function getConflatedLayer() {
  if (layer) return layer

  layer = new VectorTileLayer({
    source: new PMTilesVectorSource({ url: CONFLATED_PMTILES_URL }),
    style: conflatedTileStyle,
    zIndex: 10,
  })
  return layer
}

/**
 * Update the set of enabled shared_label values and redraw the layer in place.
 * Called from MapContainer.vue whenever the conflated filter checkboxes change.
 *
 * filtersObj is {shared_label: boolean}. If every value is true we leave the
 * filter as null (fast-path, no per-feature set lookup).
 */
export function updateConflatedFilters(filtersObj) {
  const entries = Object.entries(filtersObj)
  const enabled = entries.filter(([, v]) => v).map(([k]) => k)
  if (enabled.length === entries.length) {
    enabledLabels = null
  } else if (enabled.length === 0) {
    enabledLabels = new Set()  // hide all
  } else {
    enabledLabels = new Set(enabled)
  }
  if (layer) layer.changed()
}

function conflatedTileStyle(feature) {
  if (enabledLabels !== null) {
    const label = feature.get('shared_label')
    if (!enabledLabels.has(label)) return null
  }

  const conf = feature.get('conf_mean')
  const bucket = discretizeConf(conf)
  if (!styleCache[bucket]) {
    const color = confidenceColor(conf == null || isNaN(conf) ? null : conf)
    styleCache[bucket] = new Style({
      image: new Circle({
        radius: 5,
        fill: new Fill({ color }),
        stroke: new Stroke({ color: '#fff', width: 1 }),
      }),
    })
  }
  return styleCache[bucket]
}

/**
 * Wrap an immutable VectorTile RenderFeature in a plain object that exposes
 * the OL-Feature-ish API used by PoiPopup.vue.
 */
export function wrapConflatedFeature(rf) {
  const props = {
    _source: 'conflated',
    unified_id: rf.get('unified_id'),
    source: rf.get('source'),
    shared_label: rf.get('shared_label'),
    name: rf.get('name'),
    brand: rf.get('brand'),
    conf_mean: rf.get('conf_mean'),
    match_score: rf.get('match_score'),
    match_distance_m: rf.get('match_distance_m'),
    source_dataset: 'Conflated (OSM + Overture)',
  }
  return {
    get: (k) => props[k],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}
