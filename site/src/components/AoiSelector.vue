<template>
  <div class="aoi-selector">
    <select @change="onSelect" :value="''">
      <option value="" disabled>Jump to AOI…</option>
      <option
        v-for="aoi in aois"
        :key="aoi.id"
        :value="aoi.id"
      >
        {{ aoi.label }}
      </option>
    </select>
  </div>
</template>

<script setup>
import { AOIS } from '../constants.js'

const emit = defineEmits(['fly-to'])
const aois = AOIS

function onSelect(evt) {
  const aoi = aois.find(a => a.id === evt.target.value)
  if (!aoi) return
  const [west, south, east, north] = aoi.bbox
  const [lng, lat] = aoi.center
  emit('fly-to', { west, south, east, north, lng, lat })
  // Reset to placeholder so the same AOI can be re-selected
  evt.target.value = ''
}
</script>
