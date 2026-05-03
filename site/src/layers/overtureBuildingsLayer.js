import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { Style, Fill, Stroke } from 'ol/style'
import { OVERTURE_BUILDINGS_PMTILES_URL, COLORS } from '../constants.js'

let layer = null

// Fixed style — Overture buildings, no confidence coloring
const OVERTURE_STYLE = new Style({
  fill: new Fill({ color: COLORS.overture + 'bb' }),
  stroke: new Stroke({ color: COLORS.overture, width: 0.8 }),
})

export function getOvertureBuildingsLayer() {
  if (layer) return layer

  layer = new VectorTileLayer({
    source: new PMTilesVectorSource({ url: OVERTURE_BUILDINGS_PMTILES_URL }),
    style: OVERTURE_STYLE,
    zIndex: 10,
    visible: false,
  })
  return layer
}

/** No-op: no occupancy filter for Overture buildings. */
export function updateOvertureBuildingFilters(_filtersObj) {}

/**
 * Wrap a VectorTile RenderFeature for BuildingPopup.vue.
 */
export function wrapOvertureBuildingFeature(rf) {
  const names = tryParse(rf.get('names'))
  const props = {
    _source:        'overture_buildings',
    overture_id:    rf.get('id'),
    name:           names?.primary ?? null,
    height:         rf.get('height'),
    num_floors:     rf.get('num_floors'),
    overture_class: rf.get('class'),
    source_dataset: 'Overture Maps (Buildings)',
  }
  return {
    get: (k) => props[k],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}

function tryParse(str) {
  if (!str) return null
  try { return JSON.parse(str) } catch { return null }
}
