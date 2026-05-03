import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { Style, Fill, Stroke } from 'ol/style'
import { FEMA_BUILDINGS_PMTILES_URL, OCCUPANCY_COLORS } from '../constants.js'

let layer = null
let enabledOccupancies = null  // null = all on; Set<string> for filtered

const styleCache = {}

export function getFemaBuildingsLayer() {
  if (layer) return layer

  layer = new VectorTileLayer({
    source: new PMTilesVectorSource({ url: FEMA_BUILDINGS_PMTILES_URL }),
    style: femaBuildingStyle,
    zIndex: 11,
    visible: false,
  })
  return layer
}

export function updateFemaFilters(filtersObj) {
  const entries = Object.entries(filtersObj)
  const enabled = entries.filter(([, v]) => v).map(([k]) => k)
  if (enabled.length === entries.length) {
    enabledOccupancies = null
  } else if (enabled.length === 0) {
    enabledOccupancies = new Set()
  } else {
    enabledOccupancies = new Set(enabled)
  }
  if (layer) layer.changed()
}

function femaBuildingStyle(feature) {
  const occ = feature.get('OCC_CLS')
  if (enabledOccupancies !== null && !enabledOccupancies.has(occ)) return null

  const key = occ ?? '_unknown'
  if (!styleCache[key]) {
    const hex = OCCUPANCY_COLORS[key] ?? OCCUPANCY_COLORS._unknown
    styleCache[key] = new Style({
      fill:   new Fill({ color: hex + 'bb' }),
      stroke: new Stroke({ color: hex, width: 0.8 }),
    })
  }
  return styleCache[key]
}

export function wrapFemaFeature(rf) {
  const props = {
    _source: 'fema',
    build_id:       rf.get('BUILD_ID') ?? rf.get('build_id'),
    occ_cls:        rf.get('OCC_CLS')  ?? rf.get('occ_cls'),
    bldg_type:      rf.get('BLDGTYPE') ?? rf.get('bldgtype'),
    sqfeet:         rf.get('SQFEET')   ?? rf.get('sqfeet'),
    stories:        rf.get('STORIES')  ?? rf.get('stories'),
    source_dataset: 'FEMA USA Structures',
  }
  return {
    get: (k) => props[k],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}
