<template>
  <!-- Dropdown for narrow screens -->
  <select
    class="source-select"
    :value="activeSource"
    @change="$emit('update:source', $event.target.value)"
  >
    <option value="gold">Gold Buildings</option>
    <option value="fema">FEMA</option>
    <option value="overture">Overture</option>
  </select>

  <!-- Buttons for wider screens -->
  <div class="source-toggle">
    <button
      :class="{ active: activeSource === 'gold' }"
      @click="$emit('update:source', 'gold')"
    >
      Gold Buildings
    </button>
    <button
      :class="{ active: activeSource === 'fema' }"
      @click="$emit('update:source', 'fema')"
    >
      FEMA
    </button>
    <button
      :class="{ active: activeSource === 'overture' }"
      @click="$emit('update:source', 'overture')"
    >
      Overture
    </button>
  </div>

  <!-- NSI overlay toggle (always visible) -->
  <label class="nsi-toggle">
    <input
      type="checkbox"
      :checked="nsiUnmatchedVisible"
      @change="$emit('update:nsi-visible', $event.target.checked)"
    />
    NSI Points
  </label>
</template>

<script setup>
defineProps({
  activeSource: { type: String, required: true },
  nsiUnmatchedVisible: { type: Boolean, required: true },
})
defineEmits(['update:source', 'update:nsi-visible'])
</script>
