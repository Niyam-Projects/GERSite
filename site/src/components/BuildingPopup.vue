<template>
  <div class="building-popup" v-if="feature">
    <button class="close-btn" @click="$emit('close')">&times;</button>

    <div class="source-badge" :class="sourceBadgeClass">{{ sourceLabel }}</div>

    <!-- Gold (conflated) building -->
    <template v-if="src === 'gold'">
      <h3>Building</h3>
      <div v-if="f.building_id" class="detail-row detail-row--muted">
        <span class="detail-label">ID</span>
        <span class="detail-value detail-monospace">{{ f.building_id }}</span>
      </div>
      <div v-if="f.source" class="detail-row">
        <span class="detail-label">Match type</span>
        <span class="detail-value">{{ formatMatchType(f.source) }}</span>
      </div>
      <div v-if="f.general_occupancy" class="detail-row">
        <span class="detail-label">Occupancy</span>
        <span class="detail-value">{{ f.general_occupancy }}</span>
      </div>
      <template v-if="validNum(f.occupancy_confidence)">
        <div class="detail-row">
          <span class="detail-label">Occupancy conf.</span>
          <span class="detail-value">{{ pct(f.occupancy_confidence) }}%</span>
        </div>
        <div class="confidence-bar">
          <div class="confidence-fill" :style="confStyle(f.occupancy_confidence)" />
        </div>
      </template>
      <template v-if="validNum(f.conflation_confidence)">
        <div class="detail-row">
          <span class="detail-label">Location conf.</span>
          <span class="detail-value">{{ pct(f.conflation_confidence) }}%</span>
        </div>
        <div class="confidence-bar">
          <div class="confidence-fill" :style="confStyle(f.conflation_confidence)" />
        </div>
      </template>
      <div v-if="validNum(f.height)" class="detail-row">
        <span class="detail-label">Height</span>
        <span class="detail-value">{{ f.height }} m</span>
      </div>
      <div v-if="validNum(f.num_floors)" class="detail-row">
        <span class="detail-label">Floors</span>
        <span class="detail-value">{{ f.num_floors }}</span>
      </div>
      <div v-if="f.fema_occ_cls" class="detail-row">
        <span class="detail-label">FEMA OCC_CLS</span>
        <span class="detail-value">{{ f.fema_occ_cls }}</span>
      </div>
      <div v-if="f.nsi_occtype" class="detail-row">
        <span class="detail-label">NSI occtype</span>
        <span class="detail-value">{{ f.nsi_occtype }}</span>
      </div>
      <div v-if="validNum(f.nsi_val_struct)" class="detail-row">
        <span class="detail-label">NSI struct. value</span>
        <span class="detail-value">${{ formatCurrency(f.nsi_val_struct) }}</span>
      </div>
      <div v-if="f.fema_iou != null" class="detail-row">
        <span class="detail-label">FEMA IoU</span>
        <span class="detail-value">{{ Number(f.fema_iou).toFixed(3) }}</span>
      </div>
      <div v-if="f.overture_id" class="detail-row detail-row--muted">
        <span class="detail-label">Overture ID</span>
        <span class="detail-value detail-monospace small-id">{{ f.overture_id }}</span>
      </div>
      <div v-if="f.fema_id" class="detail-row detail-row--muted">
        <span class="detail-label">FEMA ID</span>
        <span class="detail-value detail-monospace small-id">{{ f.fema_id }}</span>
      </div>
    </template>

    <!-- FEMA USA Structures building -->
    <template v-else-if="src === 'fema'">
      <h3>FEMA Building</h3>
      <div v-if="f.occ_cls" class="detail-row">
        <span class="detail-label">Occupancy class</span>
        <span class="detail-value">{{ f.occ_cls }}</span>
      </div>
      <div v-if="f.bldg_type" class="detail-row">
        <span class="detail-label">Building type</span>
        <span class="detail-value">{{ f.bldg_type }}</span>
      </div>
      <div v-if="validNum(f.sqfeet)" class="detail-row">
        <span class="detail-label">Sq. feet</span>
        <span class="detail-value">{{ Number(f.sqfeet).toLocaleString() }}</span>
      </div>
      <div v-if="validNum(f.stories)" class="detail-row">
        <span class="detail-label">Stories</span>
        <span class="detail-value">{{ f.stories }}</span>
      </div>
      <div v-if="f.build_id" class="detail-row detail-row--muted">
        <span class="detail-label">BUILD_ID</span>
        <span class="detail-value detail-monospace small-id">{{ f.build_id }}</span>
      </div>
    </template>

    <!-- Overture Maps building -->
    <template v-else-if="src === 'overture_buildings'">
      <h3>{{ f.name || 'Building' }}</h3>
      <div v-if="f.overture_class" class="detail-row">
        <span class="detail-label">Class</span>
        <span class="detail-value">{{ f.overture_class }}</span>
      </div>
      <div v-if="validNum(f.height)" class="detail-row">
        <span class="detail-label">Height</span>
        <span class="detail-value">{{ f.height }} m</span>
      </div>
      <div v-if="validNum(f.num_floors)" class="detail-row">
        <span class="detail-label">Floors</span>
        <span class="detail-value">{{ f.num_floors }}</span>
      </div>
      <div v-if="f.overture_id" class="detail-row detail-row--muted">
        <span class="detail-label">ID</span>
        <span class="detail-value detail-monospace small-id">{{ f.overture_id }}</span>
      </div>
    </template>

    <!-- NSI unmatched point -->
    <template v-else-if="src === 'nsi_unmatched'">
      <h3>NSI Point (unmatched)</h3>
      <div v-if="f.nsi_occtype" class="detail-row">
        <span class="detail-label">Occtype</span>
        <span class="detail-value">{{ f.nsi_occtype }}</span>
      </div>
      <div v-if="validNum(f.nsi_val_struct)" class="detail-row">
        <span class="detail-label">Struct. value</span>
        <span class="detail-value">${{ formatCurrency(f.nsi_val_struct) }}</span>
      </div>
      <div v-if="validNum(f.nsi_val_cont)" class="detail-row">
        <span class="detail-label">Content value</span>
        <span class="detail-value">${{ formatCurrency(f.nsi_val_cont) }}</span>
      </div>
      <div v-if="f.nsi_id" class="detail-row detail-row--muted">
        <span class="detail-label">NSI ID</span>
        <span class="detail-value detail-monospace small-id">{{ f.nsi_id }}</span>
      </div>
    </template>

    <div class="source-footer">{{ f.source_dataset }}</div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { confidenceColor } from '../utils.js'

const props = defineProps({
  feature: { type: Object, default: null },
})
defineEmits(['close'])

const src = computed(() => props.feature?.get('_source'))

// Flat property bag for templates above
const f = computed(() => {
  if (!props.feature) return {}
  const obj = {}
  props.feature.getKeys().forEach(k => { obj[k] = props.feature.get(k) })
  return obj
})

const sourceLabel = computed(() => {
  switch (src.value) {
    case 'gold':              return 'Gold — Conflated'
    case 'fema':              return 'FEMA'
    case 'overture_buildings': return 'Overture'
    case 'nsi_unmatched':    return 'NSI (unmatched)'
    default:                 return src.value ?? 'Unknown'
  }
})

const sourceBadgeClass = computed(() => ({
  'badge--gold': src.value === 'gold',
  'badge--fema': src.value === 'fema',
  'badge--overture': src.value === 'overture_buildings',
  'badge--nsi': src.value === 'nsi_unmatched',
}))

function formatMatchType(s) {
  if (s === 'both') return 'Matched (FEMA + Overture)'
  if (s === 'fema_only') return 'FEMA only'
  if (s === 'overture_only') return 'Overture only'
  return s
}

function confStyle(conf) {
  return {
    width: (conf * 100) + '%',
    backgroundColor: confidenceColor(conf),
  }
}

function validNum(v) {
  return v != null && !isNaN(Number(v))
}

function pct(v) {
  return (Number(v) * 100).toFixed(0)
}

function formatCurrency(v) {
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })
}
</script>

<style scoped>
.building-popup {
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  padding: 1rem;
  min-width: 220px;
  max-width: 300px;
  font-size: 0.82rem;
  position: relative;
}
.close-btn {
  position: absolute;
  top: 6px;
  right: 8px;
  background: none;
  border: none;
  font-size: 1.1rem;
  cursor: pointer;
  color: #666;
  line-height: 1;
}
.close-btn:hover { color: #000; }

.source-badge {
  display: inline-block;
  padding: 1px 7px;
  border-radius: 4px;
  font-size: 0.72rem;
  font-weight: 600;
  margin-bottom: 6px;
  color: #fff;
}
.badge--gold    { background: #1a9850; }
.badge--fema    { background: #4682b4; }
.badge--overture { background: #2a7f7f; }
.badge--nsi     { background: #d46a00; }

h3 { margin: 0 0 8px; font-size: 0.9rem; }

.detail-row {
  display: flex;
  gap: 6px;
  margin: 3px 0;
  align-items: baseline;
}
.detail-row--muted { opacity: 0.65; }
.detail-label {
  color: #666;
  font-size: 0.75rem;
  white-space: nowrap;
  flex-shrink: 0;
  min-width: 100px;
}
.detail-value { flex: 1; word-break: break-word; }
.detail-monospace { font-family: monospace; font-size: 0.78rem; }
.small-id { font-size: 0.68rem; }

.confidence-bar {
  height: 5px;
  background: #eee;
  border-radius: 3px;
  margin: 2px 0 5px;
  overflow: hidden;
}
.confidence-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.2s;
}

.source-footer {
  margin-top: 10px;
  font-size: 0.68rem;
  color: #999;
  border-top: 1px solid #eee;
  padding-top: 5px;
}
</style>
