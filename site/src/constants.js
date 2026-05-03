// Overture buildings — hosted monthly release (update URL each release cycle)
export const OVERTURE_BUILDINGS_PMTILES_URL =
  'https://tiles.overturemaps.org/2026-04-15.0/buildings.pmtiles'

// PMTile URL helpers — served locally by the Vite dev server middleware (vite.config.js).
// For production deployment, replace the base path with a public HTTP URL.
export function goldBuildingsTilesUrl(aoiId)  { return `/tiles/gold_buildings/${aoiId}/buildings.pmtiles` }
export function femaBuildingsTilesUrl(aoiId)  { return `/tiles/fema_buildings/${aoiId}/fema.pmtiles` }
export function nsiUnmatchedTilesUrl(aoiId)   { return `/tiles/nsi_unmatched/${aoiId}/nsi_unmatched.pmtiles` }

// Areas of Interest (AOIs) — all PMTile layers for every AOI are loaded simultaneously.
// The AOI selector in the header is purely for navigation (fly-to), not visibility toggling.
// bbox: [west, south, east, north]
export const AOIS = [
  {
    id:     'miami_dade',
    label:  'Miami-Dade County, FL',
    center: [-80.1918, 25.7617],
    bbox:   [-80.880, 25.130, -80.100, 25.980],
  },
  {
    id:     'saipan',
    label:  'Saipan, CNMI',
    center: [145.765, 15.19],
    bbox:   [145.650, 15.060, 145.880, 15.320],
  },
  {
    id:     'puerto_rico',
    label:  'Puerto Rico',
    center: [-66.26, 18.22],
    bbox:   [-67.300, 17.870, -65.220, 18.520],
  },
  {
    id:     'guam',
    label:  'Guam',
    center: [144.794, 13.444],
    bbox:   [144.618, 13.234, 144.956, 13.654],
  },
]

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
export const INITIAL_CENTER = [-80.1918, 25.7617]  // Miami-Dade (first AOI)
export const INITIAL_ZOOM = 14

// Stadia Maps Geocoding
export const STADIA_GEOCODING_URL = 'https://api.stadiamaps.com/geocoding/v1/search'

