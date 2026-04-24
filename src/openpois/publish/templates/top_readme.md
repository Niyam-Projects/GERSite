# OpenPOIs

A unified, open dataset for points of interest across the United States. Built from [OpenStreetMap](https://www.openstreetmap.org) and
[Overture Maps](https://overturemaps.org), with per-POI confidence scores
produced by the OpenPOIs turnover model.

- 🌐 **Interactive map:** <https://openpois.org/>
- 💻 **Source code:** <https://github.com/henryspatialanalysis/openpois>
- 📘 **Data license:** [Open Database License v.1.0](./LICENSE). For more details, see the [Open Data Commons](https://opendatacommons.org/licenses/odbl/1-0/).

## Repository layout

Each refresh writes a new versioned folder. Inside every version:

```
<YYYY-MM-DD-vN>/
├── README.md                     # version metadata (OSM date, Overture release, model)
├── osm-parquet/                  # OSM-only snapshot, geohash-partitioned
├── osm-pmtiles/osm.pmtiles       # OSM snapshot as a single PMTiles archive
├── conflated-parquet/            # OSM × Overture conflated snapshot, geohash-partitioned
└── conflated-pmtiles/conflated.pmtiles
```

Browse all versions at
<https://source.coop/henryspatialanalysis/openpois>.

## What's in the conflated dataset

One row per real-world POI after matching OpenStreetMap features against
Overture Maps places. Key columns:

| Column | Description |
|---|---|
| `unified_id` | Stable ID for the conflated POI |
| `source` | `osm`, `overture`, or `both` |
| `osm_id`, `osm_type` | Source OSM feature (when present) |
| `overture_id` | Source Overture ID (when present) |
| `name`, `brand` | Preferred display names |
| `shared_label` | Harmonised category across the two source taxonomies |
| `conf_mean` | Model-estimated probability the POI currently exists (1 = exists, 0 = incorrect or stale) |
| `conf_lower`, `conf_upper` | 90% uncertainty interval for confidence score |
| `match_score`, `match_distance_m` | Diagnostics for the OSM × Overture link |
| `geometry` | WKB point (EPSG:4326) |

The `osm-parquet/` files contain the same OSM rows before conflation. This data retains the original OSM tags.

## Quickstart

Read a specific version directly from Source Cooperative (no authentication):

### Python: pyarrow

```python
import pyarrow.dataset as ds

BASE = "https://data.source.coop/henryspatialanalysis/openpois"
VERSION = "2026-04-23-v0"   # replace with the latest version folder

pois = ds.dataset(
    f"{BASE}/{VERSION}/conflated-parquet/",
    format = "parquet",
    partitioning = "hive",
)
print(pois.schema)
print(f"{pois.count_rows():,} POIs")
```

### Python: DuckDB

```python
import duckdb

BASE = "https://data.source.coop/henryspatialanalysis/openpois"
VERSION = "2026-04-23-v0"

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
df = con.execute(f"""
    SELECT shared_label, COUNT(*) AS n
    FROM read_parquet('{BASE}/{VERSION}/conflated-parquet/**/*.parquet',
                      hive_partitioning = true)
    GROUP BY shared_label
    ORDER BY n DESC
    LIMIT 20
""").df()
print(df)
```

### Python: GeoPandas

```python
import geopandas as gpd

BASE = "https://data.source.coop/henryspatialanalysis/openpois"
VERSION = "2026-04-23-v0"

# geohash_prefix=9q is roughly the US West coast
gdf = gpd.read_parquet(
    f"{BASE}/{VERSION}/conflated-parquet/geohash_prefix=9q/part-0.parquet"
)
print(gdf.head())
```

### Browser / vector-tile map

The `*-pmtiles/*.pmtiles` archives can be loaded directly by any PMTiles
client (MapLibre + `pmtiles://`, OpenLayers + `ol-pmtiles`, etc.). See
`site/` in the GitHub repo for a working example.

The snippet below is a self-contained HTML page that renders the conflated
PMTiles over MapLibre, coloured by the model's `conf_mean`. Save it as
`openpois.html` and open it in a browser — no build step, no server needed.
PMTiles are authored at zoom 14, so zoom in past z14 to see points.

```html
<!doctype html>
<meta charset="utf-8" />
<title>OpenPOIs — conflated</title>
<link href="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css" rel="stylesheet" />
<style>html, body, #map { height: 100%; margin: 0; }</style>
<div id="map"></div>
<script src="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/pmtiles@3/dist/pmtiles.js"></script>
<script>
  const BASE = "https://data.source.coop/henryspatialanalysis/openpois";
  const VERSION = "2026-04-23-v0";

  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);

  const map = new maplibregl.Map({
    container: "map",
    style: "https://tiles.openfreemap.org/styles/positron",
    center: [-73.9855, 40.758],   // Times Square
    zoom: 16,
  });

  map.on("load", () => {
    map.addSource("openpois", {
      type: "vector",
      url: `pmtiles://${BASE}/${VERSION}/conflated-pmtiles/conflated.pmtiles`,
      minzoom: 14,
    });
    map.addLayer({
      id: "openpois-points",
      type: "circle",
      source: "openpois",
      "source-layer": "conflated_pois",   // set by publish.pmtiles.conflated_layer_name
      paint: {
        "circle-radius": 4,
        "circle-stroke-width": 1,
        "circle-stroke-color": "#ffffff",
        // Red when stale (conf_mean ≈ 0), green when fresh (≈ 1).
        "circle-color": [
          "interpolate", ["linear"], ["get", "conf_mean"],
          0.0, "#d73027",
          0.3, "#fee08b",
          0.7, "#1a9850",
        ],
      },
    });
    map.on("click", "openpois-points", (e) => {
      const p = e.features[0].properties;
      new maplibregl.Popup()
        .setLngLat(e.lngLat)
        .setHTML(
          `<b>${p.name ?? "(no name)"}</b><br>` +
          `${p.shared_label} · source=${p.source}<br>` +
          `conf_mean = ${Number(p.conf_mean).toFixed(3)}`
        )
        .addTo(map);
    });
  });
</script>
```


## License & attribution

The OpenPOIs dataset is released under the [Open Database License (ODbL) v.1.0](./LICENSE). Any public use must credit OpenStreetMap contributors, the Overture Maps Foundation, and OpenPOIs. Any derivative database must be shared under the same license. See <https://www.openstreetmap.org/copyright> and <https://docs.overturemaps.org/attribution/> for upstream attribution
requirements.

## Citation

If you use this data in research, please cite:

> Henry Spatial Analysis (2026). *OpenPOIs: a unified, confidence-scored
> dataset of U.S. points of interest.* <https://openpois.henryspatialanalysis.com>

## Contact

Questions, bug reports, and contributions welcome via <https://github.com/henryspatialanalysis/openpois/issues>.
