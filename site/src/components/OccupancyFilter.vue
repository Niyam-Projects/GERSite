<template>
  <div class="occupancy-filter">
    <div class="filter-header" @click="collapsed = !collapsed">
      <span>Filters</span>
      <span>{{ collapsed ? '+' : '−' }}</span>
    </div>

    <div v-if="!collapsed" class="filter-body">
      <!-- Gold / FEMA: show occupancy checkboxes with color swatches -->
      <template v-if="activeSource === 'gold' || activeSource === 'fema'">
        <div class="filter-actions">
          <button class="filter-action-btn" @click="selectAll">All</button>
          <button class="filter-action-btn" @click="selectNone">None</button>
        </div>
        <div class="filter-list">
          <label v-for="occ in OCCUPANCY_TYPES" :key="occ">
            <input
              type="checkbox"
              :checked="currentFilters[occ]"
              @change="toggle(occ)"
            />
            <span
              class="occ-swatch"
              :style="{ background: OCCUPANCY_COLORS[occ] ?? OCCUPANCY_COLORS._unknown }"
            />
            {{ occ }}
          </label>
        </div>
      </template>

      <!-- Overture: no occupancy data -->
      <template v-else-if="activeSource === 'overture'">
        <p class="filter-note">
          Overture buildings do not include occupancy classification.
        </p>
      </template>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { OCCUPANCY_TYPES, OCCUPANCY_COLORS } from '../constants.js'

const props = defineProps({
  activeSource:  { type: String, required: true },
  goldFilters:   { type: Object, required: true },
  femaFilters:   { type: Object, required: true },
})

const emit = defineEmits(['update:gold-filters', 'update:fema-filters'])

const collapsed = ref(false)

const currentFilters = computed(() =>
  props.activeSource === 'fema' ? props.femaFilters : props.goldFilters
)

function emitUpdate(updated) {
  if (props.activeSource === 'fema') {
    emit('update:fema-filters', updated)
  } else {
    emit('update:gold-filters', updated)
  }
}

function toggle(occ) {
  emitUpdate({ ...currentFilters.value, [occ]: !currentFilters.value[occ] })
}

function selectAll() {
  const all = {}
  for (const occ of OCCUPANCY_TYPES) all[occ] = true
  emitUpdate(all)
}

function selectNone() {
  const none = {}
  for (const occ of OCCUPANCY_TYPES) none[occ] = false
  emitUpdate(none)
}
</script>

<style scoped>
.occupancy-filter {
  position: absolute;
  bottom: 2.5rem;
  left: 0.75rem;
  background: #fff;
  border-radius: 6px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.2);
  min-width: 170px;
  max-width: 200px;
  font-size: 0.82rem;
  z-index: 1000;
  overflow: hidden;
}
.filter-header {
  display: flex;
  justify-content: space-between;
  padding: 6px 10px;
  cursor: pointer;
  background: #f4f4f4;
  font-weight: 600;
  user-select: none;
}
.filter-header:hover { background: #e8e8e8; }

.filter-body {
  padding: 8px 10px;
}
.filter-actions {
  display: flex;
  gap: 6px;
  margin-bottom: 6px;
}
.filter-action-btn {
  flex: 1;
  padding: 3px 0;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #f9f9f9;
  cursor: pointer;
  font-size: 0.78rem;
}
.filter-action-btn:hover { background: #e8e8e8; }

.filter-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.filter-list label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
}

.occ-swatch {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 2px;
  flex-shrink: 0;
}

.filter-note {
  color: #666;
  font-size: 0.78rem;
  margin: 0;
}
</style>
