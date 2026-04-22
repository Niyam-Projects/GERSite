// Taxonomy arrays are generated from the conflation CSVs by
// scripts/build_taxonomy.py. Only the display-label maps below are
// hand-maintained. Run `python scripts/check_taxonomy_sync.py` to detect drift.
import {
  SHARED_LABELS,
  OSM_KEYS,
  OVERTURE_L0S,
} from './taxonomy.generated.js'

// S3 URLs
export const OSM_S3_BASE =
  'https://openpois-public.s3.us-west-2.amazonaws.com/snapshots/osm/20260417/osm_snapshot_partitioned'

export const FSQ_S3_BASE =
  'https://openpois-public.s3.us-west-2.amazonaws.com/snapshots/foursquare/20260313/foursquare_snapshot_partitioned'

export const CONFLATED_S3_BASE =
  'https://openpois-public.s3.us-west-2.amazonaws.com/snapshots/conflated/20260422/conflated_partitioned'

// Overture PMTiles (latest release — update URL on each Overture monthly release)
export const OVERTURE_PMTILES_URL =
  'https://tiles.overturemaps.org/2026-04-15.0/places.pmtiles'

// Confidence color ramp (conf_mean 0-1, 1 = stable)
export const COLORS = {
  low: '#d73027',      // red, conf < 0.3
  medium: '#fee08b',   // yellow, conf 0.3-0.7
  high: '#1a9850',     // green, conf > 0.7
  foursquare: '#3b82f6', // blue (deferred)
  cluster: '#6366f1',  // indigo for clusters
  geolocation: '#60a5fa', // light blue dot
}

export const CONFIDENCE_THRESHOLDS = { low: 0.3, high: 0.7 }

const OSM_KEY_LABELS = {
  amenity: 'Amenity',
  shop: 'Shop',
  leisure: 'Leisure',
  healthcare: 'Healthcare',
  craft: 'Craft',
  historic: 'Historic',
  landuse: 'Landuse',
  office: 'Office',
  tourism: 'Tourism',
}

const OVERTURE_L0_LABELS = {
  food_and_drink: 'Food & Drink',
  shopping: 'Shopping',
  arts_and_entertainment: 'Arts & Entertainment',
  sports_and_recreation: 'Sports & Recreation',
  health_care: 'Health Care',
  services_and_business: 'Services & Business',
  lifestyle_services: 'Lifestyle Services',
  community_and_government: 'Community & Government',
  cultural_and_historic: 'Cultural & Historic',
  education: 'Education',
  travel_and_transportation: 'Travel & Transportation',
  lodging: 'Lodging',
}

export const OSM_FILTER_KEYS = OSM_KEYS.map(key => ({
  key,
  label: OSM_KEY_LABELS[key] ?? key,
}))

export const OVERTURE_CATEGORIES = OVERTURE_L0S.map(key => ({
  key,
  label: OVERTURE_L0_LABELS[key] ?? key,
}))

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

// Conflated shared_label categories — generated from match_radii.csv.
// "Other *" entries are sorted last; App.vue uses that convention to leave
// them unchecked by default.
export const CONFLATED_LABELS = SHARED_LABELS

// Foursquare L1 categories
export const FSQ_CATEGORIES = [
  { key: 'Dining and Drinking', label: 'Dining & Drinking' },
  { key: 'Retail', label: 'Retail' },
  { key: 'Arts and Entertainment', label: 'Arts & Entertainment' },
  { key: 'Sports and Recreation', label: 'Sports & Recreation' },
  { key: 'Health and Medicine', label: 'Health & Medicine' },
]

// Geohash config
export const GEOHASH_PRECISION = 4
export const MAX_GEOHASH_CELLS = 50

// Zoom thresholds
export const MIN_ZOOM_FOR_DATA = 14
export const CLUSTER_MAX_ZOOM = 12

// Stadia Maps Geocoding
export const STADIA_GEOCODING_URL =
  'https://api.stadiamaps.com/geocoding/v1/search'

// Initial map view — Times Square (fallback if geolocation is denied)
export const INITIAL_CENTER = [-73.9855, 40.758]
export const INITIAL_ZOOM = 18
