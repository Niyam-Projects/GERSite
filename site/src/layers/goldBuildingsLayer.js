import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { Style, Fill, Stroke } from 'ol/style'
import { GOLD_BUILDINGS_PMTILES_URL, OCCUPANCY_COLORS } from '../constants.js'

let layer = null
let enabledOccupancies = null  // null = all on; Set<string> for filtered

const styleCache = {}

export function getGoldBuildingsLayer() {
  if (layer) return layer

  layer = new VectorTileLayer({
    source: new PMTilesVectorSource({ url: GOLD_BUILDINGS_PMTILES_URL }),
    style: goldBuildingStyle,
    zIndex: 12,
  })
  return layer
}

export function updateGoldFilters(filtersObj) {
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

function goldBuildingStyle(feature) {
  const occ = feature.get('general_occupancy')
  if (enabledOccupancies !== null && !enabledOccupancies.has(occ)) return null

  const key = occ ?? '_unknown'
  if (!styleCache[key]) {
    const hex = OCCUPANCY_COLORS[key] ?? OCCUPANCY_COLORS._unknown
    styleCache[key] = new Style({
      fill:   new Fill({ color: hex + 'bb' }),
      stroke: new Stroke({ color: '#ffffff88', width: 0.5 }),
    })
  }
  return styleCache[key]
}

export function wrapGoldFeature(rf) {
  const props = {
    _source: 'gold',
    building_id:            rf.get('building_id'),
    source:                 rf.get('source'),
    overture_id:            rf.get('overture_id'),
    fema_id:                rf.get('fema_id'),
    fema_iou:               rf.get('fema_iou'),
    height:                 rf.get('height'),
    num_floors:             rf.get('num_floors'),
    overture_class:         rf.get('overture_class'),
    fema_occ_cls:           rf.get('fema_occ_cls'),
    nsi_occtype:            rf.get('nsi_occtype'),
    nsi_val_struct:         rf.get('nsi_val_struct'),
    general_occupancy:      rf.get('general_occupancy'),
    occupancy_confidence:   rf.get('occupancy_confidence'),
    conflation_confidence:  rf.get('conflation_confidence'),
    source_dataset:         'Gold (Conflated)',
  }
  return {
    get: (k) => props[k],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}
