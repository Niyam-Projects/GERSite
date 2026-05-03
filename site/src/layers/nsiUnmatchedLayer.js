import VectorTileLayer from 'ol/layer/VectorTile'
import { PMTilesVectorSource } from 'ol-pmtiles'
import { Style, Circle, Fill, Stroke } from 'ol/style'
import { NSI_UNMATCHED_PMTILES_URL, COLORS } from '../constants.js'

let layer = null

// Distinct orange point style to contrast with building polygon layers
const NSI_STYLE = new Style({
  image: new Circle({
    radius: 5,
    fill: new Fill({ color: COLORS.nsi }),
    stroke: new Stroke({ color: '#ffffff', width: 1 }),
  }),
})

export function getNsiUnmatchedLayer() {
  if (layer) return layer

  layer = new VectorTileLayer({
    source: new PMTilesVectorSource({ url: NSI_UNMATCHED_PMTILES_URL }),
    style: NSI_STYLE,
    zIndex: 20,    // on top of polygon layers so points are always clickable
    visible: false,
  })
  return layer
}

/**
 * Wrap a VectorTile RenderFeature for BuildingPopup.vue.
 */
export function wrapNsiUnmatchedFeature(rf) {
  const props = {
    _source:        'nsi_unmatched',
    nsi_id:         rf.get('nsi_id'),
    nsi_occtype:    rf.get('nsi_occtype'),
    nsi_val_struct: rf.get('nsi_val_struct'),
    nsi_val_cont:   rf.get('nsi_val_cont'),
    source_dataset: 'NSI (unmatched — no building footprint)',
  }
  return {
    get: (k) => props[k],
    getKeys: () => Object.keys(props),
    getGeometry: () => rf.getGeometry(),
  }
}
