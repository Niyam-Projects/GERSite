// PMTiles URLs — served locally by the Vite dev server middleware (vite.config.js).
// For production deployment, replace with public HTTP URLs (e.g., Source Cooperative).
export const GOLD_BUILDINGS_PMTILES_URL = '/tiles/gold_buildings/miami_dade/buildings.pmtiles'
export const FEMA_BUILDINGS_PMTILES_URL = '/tiles/fema_buildings/miami_dade/fema.pmtiles'
export const NSI_UNMATCHED_PMTILES_URL  = '/tiles/nsi_unmatched/miami_dade/nsi_unmatched.pmtiles'

// Overture buildings — hosted monthly release (update URL each release cycle)
export const OVERTURE_BUILDINGS_PMTILES_URL =
  'https://tiles.overturemaps.org/2026-04-15.0/buildings.pmtiles'

// Canonical FEMA OCC_CLS occupancy categories (used as filter options for
// both the Gold layer general_occupancy field and the FEMA OCC_CLS field).
export const OCCUPANCY_TYPES = [
  'Residential',
  'Commercial',
  'Industrial',
  'Agriculture',
  'Religion',
  'Government',
  'Education',
]

// Occupancy type color palette (Tableau 10 — colorblind-distinguishable)
export const OCCUPANCY_COLORS = {
  Residential: '#4e79a7',
  Commercial:  '#f28e2b',
  Industrial:  '#e15759',
  Agriculture: '#76b7b2',
  Religion:    '#59a14f',
  Government:  '#edc948',
  Education:   '#b07aa1',
  _unknown:    '#aaaaaa',  // fallback for unclassified features
}

// Keep non-occupancy colors used by other layers
export const COLORS = {
  nsi:      '#f28e2b',  // orange — NSI unmatched points
  overture: '#76b7b2',  // teal   — Overture buildings
}

export const CONFIDENCE_THRESHOLDS = { low: 0.3, high: 0.7 }

// OpenFreeMap base map styles
export const BASE_MAP_STYLES = [
  {
    key: 'positron',
    label: 'Positron',
    url: 'https://tiles.openfreemap.org/styles/positron',
  },
  {
    key: 'liberty',
    label: 'Liberty',
    url: 'https://tiles.openfreemap.org/styles/liberty',
  },
  {
    key: 'dark',
    label: 'Dark Matter',
    url: 'https://tiles.openfreemap.org/styles/dark',
  },
]

// Zoom thresholds
export const MIN_ZOOM = 10
export const INITIAL_CENTER = [-80.1918, 25.7617]  // Miami, FL
export const INITIAL_ZOOM = 14

// Stadia Maps Geocoding
export const STADIA_GEOCODING_URL = 'https://api.stadiamaps.com/geocoding/v1/search'

