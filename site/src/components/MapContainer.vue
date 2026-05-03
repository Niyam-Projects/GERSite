<template>
  <div class="map-container">
    <div ref="mapEl" style="width: 100%; height: 100%"></div>

    <button
      class="geolocation-btn"
      title="My location"
      @click="handleGeolocate"
    >
      <span class="material-symbols-outlined">my_location</span>
    </button>

    <!-- Desktop: native select -->
    <div class="basemap-switcher basemap-switcher-desktop">
      <select v-model="selectedStyle" @change="switchBaseMap">
        <option v-for="s in baseMapStyles" :key="s.key" :value="s.key">
          {{ s.label }}
        </option>
      </select>
    </div>

    <!-- Mobile: button + centered modal -->
    <button class="basemap-switcher basemap-mobile-btn" @click="basemapModalOpen = true">
      {{ baseMapStyles.find(s => s.key === selectedStyle)?.label }} ▾
    </button>
    <Teleport to="body">
      <div v-if="basemapModalOpen" class="basemap-modal-overlay" @click.self="basemapModalOpen = false">
        <div class="basemap-modal">
          <div class="basemap-modal-title">Base Map</div>
          <button
            v-for="s in baseMapStyles"
            :key="s.key"
            :class="['basemap-modal-option', { active: selectedStyle === s.key }]"
            @click="selectBasemap(s.key)"
          >
            {{ s.label }}
          </button>
        </div>
      </div>
    </Teleport>

    <!-- Popup overlay anchor (managed by OL Overlay, not Vue v-if) -->
    <div ref="popupEl">
      <BuildingPopup :feature="selectedFeature" @close="closePopup" />
    </div>

    <div class="map-attribution">
      &copy;
      <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap<span class="attr-long"> contributors</span></a>,
      <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener noreferrer">OpenMapTiles</a>,
      <a href="https://overturemaps.org/" target="_blank" rel="noopener noreferrer">Overture Maps<span class="attr-long"> Foundation</span></a>,
      <a href="https://www.fema.gov/flood-maps/national-flood-hazard-layer" target="_blank" rel="noopener noreferrer">FEMA USA Structures</a>,
      <a href="https://www.nsi.usace.army.mil/" target="_blank" rel="noopener noreferrer">USACE NSI</a>
    </div>
  </div>
</template>

<script setup>
import {
  ref, shallowRef, watch, onMounted, onBeforeUnmount, nextTick,
} from 'vue'
import Map from 'ol/Map'
import View from 'ol/View'
import Overlay from 'ol/Overlay'
import { fromLonLat, transformExtent } from 'ol/proj'
import { apply } from 'ol-mapbox-style'
import BuildingPopup from './BuildingPopup.vue'
import { useGeolocation } from '../composables/useGeolocation.js'
import { getGoldBuildingsLayer, updateGoldFilters, wrapGoldFeature }           from '../layers/goldBuildingsLayer.js'
import { getFemaBuildingsLayer, updateFemaFilters, wrapFemaFeature }            from '../layers/femaBuildingsLayer.js'
import { getOvertureBuildingsLayer, wrapOvertureBuildingFeature }               from '../layers/overtureBuildingsLayer.js'
import { getNsiUnmatchedLayer, wrapNsiUnmatchedFeature }                        from '../layers/nsiUnmatchedLayer.js'
import {
  BASE_MAP_STYLES,
  INITIAL_CENTER,
  INITIAL_ZOOM,
  MIN_ZOOM,
} from '../constants.js'

const props = defineProps({
  activeSource:        { type: String,  required: true },  // 'gold' | 'fema' | 'overture'
  goldFilters:         { type: Object,  required: true },
  femaFilters:         { type: Object,  required: true },
  nsiUnmatchedVisible: { type: Boolean, default: false },
})

const mapEl = ref(null)
const popupEl = ref(null)
const map = shallowRef(null)
const popupOverlay = shallowRef(null)
const selectedFeature = shallowRef(null)
const selectedStyle = ref('positron')
const basemapModalOpen = ref(false)
const baseMapStyles = BASE_MAP_STYLES

const { locate } = useGeolocation()

let geoOverlay = null
let geocodeMarker = null

// Helper: get all data layers
function getDataLayers() {
  return [getGoldBuildingsLayer(), getFemaBuildingsLayer(), getOvertureBuildingsLayer(), getNsiUnmatchedLayer()]
}

onMounted(async () => {
  const view = new View({
    center: fromLonLat(INITIAL_CENTER),
    zoom: INITIAL_ZOOM,
    minZoom: MIN_ZOOM,
  })

  const goldLyr     = getGoldBuildingsLayer()
  const femaLyr     = getFemaBuildingsLayer()
  const overtureLyr = getOvertureBuildingsLayer()
  const nsiLyr      = getNsiUnmatchedLayer()
  goldLyr.setVisible(props.activeSource === 'gold')
  femaLyr.setVisible(props.activeSource === 'fema')
  overtureLyr.setVisible(props.activeSource === 'overture')
  nsiLyr.setVisible(props.nsiUnmatchedVisible)

  const olMap = new Map({
    target: mapEl.value,
    view,
    layers: [goldLyr, femaLyr, overtureLyr, nsiLyr],
  })
  map.value = olMap

  document.getElementById('initial-loader')?.remove()

  // Try to apply vector tile style (replaces fallback raster)
  applyBaseStyle('positron')

  // Popup overlay
  await nextTick()
  popupOverlay.value = new Overlay({
    element: popupEl.value,
    positioning: 'bottom-center',
    offset: [0, -8],
  })
  olMap.addOverlay(popupOverlay.value)
  popupEl.value.addEventListener('pointerdown', (e) => e.stopPropagation())

  olMap.on('singleclick', handleClick)

  // Initialise PMTiles filters from props
  updateGoldFilters(props.goldFilters)
  updateFemaFilters(props.femaFilters)

  handleGeolocate()
})

onBeforeUnmount(() => {
  if (map.value) map.value.setTarget(null)
})

async function applyBaseStyle(styleKey) {
  const style = BASE_MAP_STYLES.find(s => s.key === styleKey)
  if (!style || !map.value) return

  const olMap = map.value
  const dataLayers = getDataLayers()

  // Save view state
  const center = olMap.getView().getCenter()
  const zoom = olMap.getView().getZoom()

  // Remove data layers BEFORE apply() to prevent "duplicate item" error.
  // apply() replaces/manages base layers but leaves others — removing them
  // first lets us cleanly re-add them on top afterward.
  const layerArr = olMap.getLayers().getArray()
  for (const lyr of dataLayers) {
    if (layerArr.includes(lyr)) olMap.removeLayer(lyr)
  }

  try {
    await apply(olMap, style.url)
    // Re-add data layers on top of the new base style
    for (const lyr of dataLayers) {
      olMap.addLayer(lyr)
    }
    olMap.getView().setCenter(center)
    olMap.getView().setZoom(zoom)
  } catch (err) {
    console.error('Failed to apply base style:', err)
    // Ensure data layers are present even on error
    const arr = olMap.getLayers().getArray()
    for (const lyr of dataLayers) {
      if (!arr.includes(lyr)) olMap.addLayer(lyr)
    }
  }
}

function switchBaseMap() {
  applyBaseStyle(selectedStyle.value)
}

function selectBasemap(key) {
  selectedStyle.value = key
  basemapModalOpen.value = false
  applyBaseStyle(key)
}

// ---- Interaction ----

function handleClick(evt) {
  closePopup()

  map.value.forEachFeatureAtPixel(evt.pixel, (feature, lyr) => {
    let wrapped = null
    if (lyr === getGoldBuildingsLayer())          wrapped = wrapGoldFeature(feature)
    else if (lyr === getFemaBuildingsLayer())      wrapped = wrapFemaFeature(feature)
    else if (lyr === getOvertureBuildingsLayer())  wrapped = wrapOvertureBuildingFeature(feature)
    else if (lyr === getNsiUnmatchedLayer())       wrapped = wrapNsiUnmatchedFeature(feature)
    if (!wrapped) return false

    selectedFeature.value = wrapped
    popupOverlay.value.setPosition(evt.coordinate)
    return true
  })
}

function closePopup() {
  selectedFeature.value = null
  if (popupOverlay.value) {
    popupOverlay.value.setPosition(undefined)
  }
}

// ---- Geolocation ----

async function handleGeolocate() {
  try {
    const pos = await locate()
    const coord = fromLonLat([pos.lon, pos.lat])

    map.value.getView().animate({
      center: coord,
      zoom: 15,
      duration: 500,
    })

    if (!geoOverlay) {
      const el = document.createElement('div')
      el.className = 'geolocation-dot'
      geoOverlay = new Overlay({
        element: el,
        positioning: 'center-center',
      })
      map.value.addOverlay(geoOverlay)
    }
    geoOverlay.setPosition(coord)
  } catch (err) {
    console.warn('Geolocation failed:', err)
  }
}

// ---- Public methods ----

function flyToBbox(bbox) {
  if (!map.value) return
  const extent = transformExtent(
    [bbox.west, bbox.south, bbox.east, bbox.north],
    'EPSG:4326',
    'EPSG:3857'
  )
  map.value.getView().fit(extent, {
    duration: 500,
    maxZoom: 17,
    padding: [50, 50, 50, 50],
  })

  if (bbox.lng != null && bbox.lat != null) {
    const coord = fromLonLat([bbox.lng, bbox.lat])
    if (!geocodeMarker) {
      const el = document.createElement('div')
      el.className = 'geocode-marker'
      geocodeMarker = new Overlay({
        element: el,
        positioning: 'center-center',
        stopEvent: false,
      })
      map.value.addOverlay(geocodeMarker)
    }
    geocodeMarker.setPosition(coord)
  }
}

defineExpose({ flyToBbox })

// ---- Watchers ----

// Source visibility watcher — NSI stays independent
watch(() => props.activeSource, (src) => {
  getGoldBuildingsLayer().setVisible(src === 'gold')
  getFemaBuildingsLayer().setVisible(src === 'fema')
  getOvertureBuildingsLayer().setVisible(src === 'overture')
  closePopup()
})

watch(() => props.nsiUnmatchedVisible, (v) => {
  getNsiUnmatchedLayer().setVisible(v)
})

watch(() => props.goldFilters, (f) => { updateGoldFilters(f) }, { deep: true })
watch(() => props.femaFilters, (f) => { updateFemaFilters(f) }, { deep: true })

</script>
