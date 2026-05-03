<template>
  <div class="top-bar">
    <SourceToggle
      :active-source="activeSource"
      :nsi-unmatched-visible="nsiUnmatchedVisible"
      @update:source="setSource"
      @update:nsi-visible="nsiUnmatchedVisible = $event"
    />
    <SearchBar @fly-to="handleFlyTo" />
    <AoiSelector @fly-to="handleFlyTo" />
    <div class="top-bar-right">
      <a href="/about.html" class="about-link">About</a>
      <a
        href="https://henryspatialanalysis.com/"
        target="_blank"
        rel="noopener noreferrer"
        class="brand-logo-link"
      >
        <img src="./assets/logo.png" alt="Henry Spatial Analysis" class="brand-logo" />
      </a>
    </div>
  </div>
  <MapContainer
    ref="mapRef"
    :active-source="activeSource"
    :gold-filters="goldFilters"
    :fema-filters="femaFilters"
    :nsi-unmatched-visible="nsiUnmatchedVisible"
  />
  <OccupancyFilter
    :active-source="activeSource"
    :gold-filters="goldFilters"
    :fema-filters="femaFilters"
    @update:gold-filters="goldFilters = $event"
    @update:fema-filters="femaFilters = $event"
  />
</template>

<script setup>
import { ref } from 'vue'
import SourceToggle from './components/SourceToggle.vue'
import SearchBar from './components/SearchBar.vue'
import AoiSelector from './components/AoiSelector.vue'
import MapContainer from './components/MapContainer.vue'
import OccupancyFilter from './components/OccupancyFilter.vue'
import { OCCUPANCY_TYPES } from './constants.js'

const activeSource = ref('gold')
const nsiUnmatchedVisible = ref(false)
const mapRef = ref(null)

const goldFilters = ref(OCCUPANCY_TYPES.reduce((acc, t) => ({ ...acc, [t]: true }), {}))
const femaFilters = ref(OCCUPANCY_TYPES.reduce((acc, t) => ({ ...acc, [t]: true }), {}))

function setSource(src) {
  activeSource.value = src
}

function handleFlyTo(bbox) {
  if (mapRef.value) {
    mapRef.value.flyToBbox(bbox)
  }
}
</script>
