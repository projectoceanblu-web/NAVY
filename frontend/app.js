/* BoB Sentinel map client.
 *
 * Talks only to this app's own API — GFW tiles are proxied server-side so the
 * GFW token is never shipped to the browser.
 */
'use strict';

const AOI = { lat: 21.4, lon: 90.4, zoom: 7 };

const COLORS = {
  matched: '#3fb950',
  dark: '#f2545b',
  indeterminate: '#d9a441',
  anomaly: '#a371f7',
  ais: '#4a9eff',
};

const map = L.map('map', { zoomControl: true }).setView([AOI.lat, AOI.lon], AOI.zoom);

L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  maxZoom: 18,
}).addTo(map);

const layers = {
  eez: L.layerGroup().addTo(map),
  ais: L.layerGroup().addTo(map),
  matched: L.layerGroup().addTo(map),
  dark: L.layerGroup().addTo(map),
  indeterminate: L.layerGroup().addTo(map),
  anomalies: L.layerGroup(),
};

function toast(message) {
  const node = document.getElementById('toast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => node.classList.remove('show'), 5000);
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} -> HTTP ${response.status}`);
  }
  return response.json();
}

function table(rows) {
  const body = rows
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .map(([key, value]) => `<tr><td>${key}</td><td>${value}</td></tr>`)
    .join('');
  return `<table>${body}</table>`;
}

function fmt(value, digits = 1) {
  return typeof value === 'number' ? value.toFixed(digits) : value;
}

/* --- layers ------------------------------------------------------------ */

async function loadRegions() {
  const data = await getJSON('/api/regions');
  layers.eez.clearLayers();
  L.geoJSON(data, {
    style: { color: '#4a9eff', weight: 2, fillOpacity: 0.04, dashArray: '5,5' },
    onEachFeature: (feature, layer) => {
      const props = feature.properties || {};
      layer.bindPopup(
        `<h3>${props.name || 'Region'}</h3>` +
          table([
            ['Kind', props.kind],
            ['MRGID', props.mrgid],
            ['Source', props.source],
          ])
      );
    },
  }).addTo(layers.eez);
  return data.features.length;
}

async function loadAIS(minutes) {
  const data = await getJSON(`/api/ais/latest?minutes=${minutes}`);
  layers.ais.clearLayers();
  data.features.forEach((feature) => {
    const [lon, lat] = feature.geometry.coordinates;
    const props = feature.properties;
    L.circleMarker([lat, lon], {
      radius: 3,
      color: COLORS.ais,
      weight: 1,
      fillOpacity: 0.7,
    })
      .bindPopup(
        `<h3>${props.name || 'Unknown vessel'}</h3>` +
          table([
            ['MMSI', props.mmsi],
            ['Last seen', props.ts],
            ['SOG', props.sog_kn != null ? `${fmt(props.sog_kn)} kn` : null],
            ['COG', props.cog_deg != null ? `${fmt(props.cog_deg, 0)}°` : null],
          ])
      )
      .addTo(layers.ais);
  });
  return data.features.length;
}

async function loadDetections(days) {
  const data = await getJSON(`/api/detections?days=${days}`);
  ['matched', 'dark', 'indeterminate'].forEach((key) => layers[key].clearLayers());

  const counts = { matched: 0, dark: 0, indeterminate: 0 };
  data.features.forEach((feature) => {
    const [lon, lat] = feature.geometry.coordinates;
    const props = feature.properties;
    const status = props.status || 'indeterminate';
    counts[status] = (counts[status] || 0) + 1;

    // Dark detections are drawn larger: they are the reason this map exists.
    L.circleMarker([lat, lon], {
      radius: status === 'dark' ? 7 : 5,
      color: COLORS[status],
      weight: 2,
      fillOpacity: 0.45,
    })
      .bindPopup(
        `<h3>SAR detection #${props.id}</h3>` +
          table([
            ['Status', status],
            ['Acquired', props.ts],
            ['SNR', props.snr_db != null ? `${fmt(props.snr_db)} dB` : null],
            ['Est. length', props.length_m != null ? `${fmt(props.length_m, 0)} m` : null],
            ['Confidence', props.confidence != null ? fmt(props.confidence, 2) : null],
            ['Matched MMSI', props.matched_mmsi],
            [
              'Match distance',
              props.match_distance_m != null ? `${fmt(props.match_distance_m, 0)} m` : null,
            ],
            ['Scene', props.scene_id],
          ]) +
          (status === 'dark'
            ? '<p style="color:#8ba0b8;font-size:11px;margin:8px 0 0">' +
              'No AIS match despite local AIS coverage. Screening signal only — ' +
              'not evidence of illegal activity.</p>'
            : '') +
          (status === 'indeterminate'
            ? '<p style="color:#8ba0b8;font-size:11px;margin:8px 0 0">' +
              'No AIS coverage nearby, so this absence of a match is uninformative.</p>'
            : '')
      )
      .addTo(layers[status]);
  });
  return counts;
}

async function loadAnomalies(days) {
  const data = await getJSON(`/api/anomalies?days=${days}`);
  layers.anomalies.clearLayers();
  data.forEach((anomaly) => {
    if (anomaly.lon == null || anomaly.lat == null) return;
    L.circleMarker([anomaly.lat, anomaly.lon], {
      radius: 6,
      color: COLORS.anomaly,
      weight: 2,
      fillOpacity: 0.3,
    })
      .bindPopup(
        `<h3>${anomaly.kind.replace(/_/g, ' ')}</h3>` +
          table([
            ['MMSI', anomaly.mmsi],
            ['Counterpart', anomaly.counterpart_mmsi],
            ['Start', anomaly.start_ts],
            ['End', anomaly.end_ts],
            ['Score', anomaly.score != null ? fmt(anomaly.score, 2) : null],
            ...Object.entries(anomaly.details || {}).map(([k, v]) => [
              k.replace(/_/g, ' '),
              typeof v === 'object' ? JSON.stringify(v) : v,
            ]),
          ])
      )
      .addTo(layers.anomalies);
  });
  return data.length;
}

async function loadStats() {
  const stats = await getJSON('/api/stats');
  const rows = [
    ['Vessels', stats.vessels],
    ['AIS positions', stats.positions],
    ['SAR scenes', stats.scenes],
    ['Detections', stats.detections],
    ['Dark', stats.dark_detections],
    ['Indeterminate', stats.indeterminate_detections],
    ['Anomalies', stats.anomalies],
    ['Latest AIS', stats.latest_ais_ts ? stats.latest_ais_ts.slice(0, 16).replace('T', ' ') : '—'],
  ];
  document.getElementById('stats').innerHTML = rows
    .map(([key, value]) => `<dt>${key}</dt><dd>${value ?? 0}</dd>`)
    .join('');
}

/* --- wiring ------------------------------------------------------------ */

function bindToggle(id, layer) {
  const input = document.getElementById(id);
  input.addEventListener('change', () => {
    if (input.checked) {
      map.addLayer(layer);
    } else {
      map.removeLayer(layer);
    }
  });
  // Honour the checkbox state the markup shipped with.
  if (!input.checked) map.removeLayer(layer);
}

async function refresh() {
  const button = document.getElementById('refresh');
  button.setAttribute('aria-busy', 'true');
  const minutes = document.getElementById('ais-minutes').value;
  const days = document.getElementById('det-days').value;

  // Settle rather than all: one failing layer should not blank the map.
  const results = await Promise.allSettled([
    loadRegions(),
    loadAIS(minutes),
    loadDetections(days),
    loadAnomalies(days),
    loadStats(),
  ]);
  const failures = results.filter((r) => r.status === 'rejected');
  if (failures.length) {
    console.error(failures.map((f) => f.reason));
    toast(`${failures.length} layer(s) failed to load — is the API up?`);
  }
  button.removeAttribute('aria-busy');
}

bindToggle('layer-eez', layers.eez);
bindToggle('layer-ais', layers.ais);
bindToggle('layer-matched', layers.matched);
bindToggle('layer-dark', layers.dark);
bindToggle('layer-indeterminate', layers.indeterminate);
bindToggle('layer-anomalies', layers.anomalies);

document.getElementById('refresh').addEventListener('click', refresh);
document.getElementById('ais-minutes').addEventListener('change', refresh);
document.getElementById('det-days').addEventListener('change', refresh);

refresh();
setInterval(refresh, 60000); // AIS is live; keep the map current
