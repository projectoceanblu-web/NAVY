/* BoB Sentinel map client.
 *
 * Every layer here is LIVE: each refresh calls this app's API, which proxies
 * Global Fishing Watch with the token held server-side. Nothing is sampled,
 * cached in the page, or shipped as fixture data.
 */
'use strict';

const AOI = { lat: 21.2, lon: 90.4, zoom: 7 };

const COLORS = {
  dark: '#f2545b',
  fishing: '#ffb02e',
  presence: '#4a9eff',
  anomaly: '#a371f7',
  eez: '#4a9eff',
  vessel: '#3fb950',
  track: '#ffffff',
};

const map = L.map('map', { zoomControl: false, worldCopyJump: true })
  .setView([AOI.lat, AOI.lon], AOI.zoom);

L.control.zoom({ position: 'topright' }).addTo(map);
L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(map);

/* --- basemaps ----------------------------------------------------------- */

const basemaps = {
  'Dark (ocean)': L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    { attribution: '&copy; OpenStreetMap &copy; CARTO', maxZoom: 19 }
  ),
  'Satellite': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Esri, Maxar, Earthstar Geographics', maxZoom: 19 }
  ),
  'Nautical chart': L.tileLayer(
    'https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png',
    { attribution: '&copy; OpenSeaMap contributors', maxZoom: 18 }
  ),
  'Ocean basemap': L.tileLayer(
    'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    { attribution: 'Esri, GEBCO, NOAA', maxZoom: 13 }
  ),
  'Street (OSM)': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19,
  }),
};

basemaps['Dark (ocean)'].addTo(map);

// The nautical chart is a transparent seamark overlay, so it is offered as an
// overlay too — on its own it renders over whatever basemap is selected.
const seamarks = L.tileLayer('https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenSeaMap contributors',
  maxZoom: 18,
  opacity: 0.9,
});

const layers = {
  eez: L.layerGroup().addTo(map),
  vessels: L.layerGroup().addTo(map),
  track: L.layerGroup().addTo(map),
  sar: L.layerGroup().addTo(map),
  fishing: L.layerGroup(),
  presence: L.layerGroup(),
  events: L.layerGroup().addTo(map),
};

L.control
  .layers(
    basemaps,
    {
      'Seamarks overlay': seamarks,
      'EEZ boundary': layers.eez,
      'Live AIS vessels': layers.vessels,
      'Selected route': layers.track,
      'SAR dark detections': layers.sar,
      'Fishing effort': layers.fishing,
      'AIS presence': layers.presence,
      'Encounters & loitering': layers.events,
    },
    { position: 'topright', collapsed: false }
  )
  .addTo(map);

/* Basemaps are remote; if they cannot be reached the vector layers still
   render over the dark background, which is the information that matters. */
let basemapWarned = false;
Object.values(basemaps).forEach((layer) =>
  layer.on('tileerror', () => {
    if (basemapWarned) return;
    basemapWarned = true;
    document.body.classList.add('no-basemap');
    toast('Basemap tiles unreachable — data layers still shown.');
  })
);

/* --- map frames --------------------------------------------------------- */

const coordsBox = document.getElementById('coords');
map.on('mousemove', (e) => {
  coordsBox.textContent = `${e.latlng.lat.toFixed(4)}°N  ${e.latlng.lng.toFixed(4)}°E`;
});
map.on('mouseout', () => {
  coordsBox.textContent = '—';
});

document.getElementById('legend').innerHTML = [
  ['dark', 'SAR dark detection'],
  ['fishing', 'Apparent fishing effort'],
  ['presence', 'AIS vessel presence'],
  ['anomaly', 'Encounter / loitering'],
]
  .map(([k, label]) => `<div><span class="swatch ${k}"></span>${label}</div>`)
  .join('');

/* --- helpers ------------------------------------------------------------ */

function toast(message) {
  const node = document.getElementById('toast');
  node.textContent = message;
  node.classList.add('show');
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => node.classList.remove('show'), 6000);
}

async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_) {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.json();
}

function table(rows) {
  return (
    '<table>' +
    rows
      .filter(([, v]) => v !== null && v !== undefined && v !== '')
      .map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`)
      .join('') +
    '</table>'
  );
}

// Grid cells carry wildly different magnitudes; scale radius by rank rather
// than raw value so one hotspot does not flatten everything else.
function radiusFor(value, max) {
  if (!value || !max) return 4;
  return 3 + 7 * Math.sqrt(Math.min(value, max) / max);
}

let bounds = null;
function extend(lat, lon) {
  const p = [lat, lon];
  bounds = bounds ? bounds.extend(p) : L.latLngBounds(p, p);
}

/* --- live layers -------------------------------------------------------- */

async function loadGrid(path, group, color, unitLabel, days) {
  const data = await getJSON(`${path}?days=${days}`);
  group.clearLayers();
  const values = data.features.map((f) => f.properties.value || 0);
  const max = Math.max(...values, 0);

  data.features.forEach((f) => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    extend(lat, lon);
    L.circleMarker([lat, lon], {
      radius: radiusFor(p.value, max),
      color,
      weight: 1,
      fillOpacity: 0.5,
    })
      .bindPopup(
        `<h3>${unitLabel}</h3>` +
          table([
            ['Value', p.value != null ? p.value.toFixed(2) : null],
            ['Date', p.date],
            ['Position', `${lat.toFixed(3)}, ${lon.toFixed(3)}`],
            ['Source', p.source],
          ])
      )
      .addTo(group);
  });
  return data.features.length;
}

async function loadEvents(days) {
  const data = await getJSON(`/api/live/events?days=${days}`);
  layers.events.clearLayers();
  data.features.forEach((f) => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    extend(lat, lon);
    L.circleMarker([lat, lon], {
      radius: 6,
      color: COLORS.anomaly,
      weight: 2,
      fillOpacity: 0.35,
    })
      .bindPopup(
        `<h3>${(p.kind || 'event').replace(/_/g, ' ')}</h3>` +
          table([
            ['Vessel', p.vessel_name],
            ['Flag', p.vessel_flag],
            ['MMSI', p.mmsi],
            ['Start', p.start],
            ['End', p.end],
            ['Source', p.source],
          ])
      )
      .addTo(layers.events);
  });
  return data.features.length;
}

async function loadEEZ() {
  const data = await getJSON('/api/regions');
  layers.eez.clearLayers();
  if (!data.features.length) return 0;
  L.geoJSON(data, {
    style: { color: COLORS.eez, weight: 2, fillOpacity: 0.03, dashArray: '6,6' },
  }).addTo(layers.eez);
  return data.features.length;
}

/* --- live vessels and route selection ----------------------------------- */

let selectedMmsi = null;
const vesselMarkers = new Map();

/* A vessel's heading is worth showing: a triangle oriented to course tells you
   far more at a glance than another dot. Built as a divIcon so it rotates
   without needing an image per heading. */
function vesselIcon(course, selected) {
  const size = selected ? 20 : 14;
  const color = selected ? COLORS.track : COLORS.vessel;
  return L.divIcon({
    className: 'vessel-icon',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    html:
      `<svg viewBox="0 0 24 24" width="${size}" height="${size}" ` +
      `style="transform: rotate(${course || 0}deg)">` +
      `<path d="M12 2 L19 22 L12 17 L5 22 Z" fill="${color}" ` +
      `stroke="#0d1520" stroke-width="1.5"/></svg>`,
  });
}

async function selectVessel(mmsi, name) {
  selectedMmsi = mmsi;
  vesselMarkers.forEach((m, key) =>
    m.setIcon(vesselIcon(m.options._course, key === mmsi))
  );

  layers.track.clearLayers();
  map.addLayer(layers.track);

  let data;
  try {
    data = await getJSON(`/api/ais/track/${mmsi}?hours=720`);
  } catch (err) {
    toast(`No stored route yet for ${name || mmsi}: ${err.message}`);
    return;
  }

  const line = data.features.find((f) => f.geometry.type === 'LineString');
  if (!line) {
    toast(`Only one position known for ${name || mmsi} so far — the route builds as AIS arrives.`);
    return;
  }

  const latlngs = line.geometry.coordinates.map(([lon, lat]) => [lat, lon]);

  // Halo beneath the route keeps it legible over satellite imagery.
  L.polyline(latlngs, { color: '#0d1520', weight: 7, opacity: 0.8 }).addTo(layers.track);
  L.polyline(latlngs, {
    color: COLORS.track,
    weight: 2.5,
    opacity: 0.95,
  }).addTo(layers.track);

  // Mark every reported position so gaps in the track are visible as gaps.
  latlngs.forEach((p, i) =>
    L.circleMarker(p, {
      radius: 2.5,
      color: COLORS.track,
      weight: 1,
      fillOpacity: 0.9,
    })
      .bindTooltip(`Fix ${i + 1} of ${latlngs.length}`)
      .addTo(layers.track)
  );

  L.circleMarker(latlngs[0], { radius: 5, color: COLORS.fishing, weight: 2, fillOpacity: 1 })
    .bindTooltip('Track start')
    .addTo(layers.track);

  map.fitBounds(L.latLngBounds(latlngs), { padding: [80, 80] });
  toast(`Route for ${name || mmsi}: ${latlngs.length} positions over ${line.properties.positions} fixes.`);
}

function clearSelection() {
  selectedMmsi = null;
  layers.track.clearLayers();
  vesselMarkers.forEach((m) => m.setIcon(vesselIcon(m.options._course, false)));
}

async function loadVessels(minutes) {
  const data = await getJSON(`/api/ais/latest?minutes=${minutes}`);
  layers.vessels.clearLayers();
  vesselMarkers.clear();

  data.features.forEach((f) => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    extend(lat, lon);

    const m = L.marker([lat, lon], {
      icon: vesselIcon(p.cog_deg, p.mmsi === selectedMmsi),
      _course: p.cog_deg,
    });
    m.bindPopup(
      `<h3>${p.name || 'Unknown vessel'}</h3>` +
        table([
          ['MMSI', p.mmsi],
          ['Last seen', p.ts],
          ['SOG', p.sog_kn != null ? `${p.sog_kn.toFixed(1)} kn` : null],
          ['COG', p.cog_deg != null ? `${p.cog_deg.toFixed(0)}°` : null],
        ]) +
        `<button class="route-btn" data-mmsi="${p.mmsi}" ` +
        `data-name="${(p.name || '').replace(/"/g, '&quot;')}">Show full route</button>`
    );
    m.on('click', () => selectVessel(p.mmsi, p.name));
    m.addTo(layers.vessels);
    vesselMarkers.set(p.mmsi, m);
  });
  return data.features.length;
}

// Popup buttons are re-created on every render, so delegate from the document.
document.addEventListener('click', (e) => {
  const btn = e.target.closest('.route-btn');
  if (btn) selectVessel(Number(btn.dataset.mmsi), btn.dataset.name);
});
map.on('contextmenu', clearSelection);

/* Pull a fresh slice of live AIS into the database before drawing vessels.
   aisstream is a push feed and serverless has no persistent process, so each
   call drains the socket briefly; tracks accumulate across calls. */
async function collectLiveAIS(seconds) {
  try {
    return await (await fetch(`/api/ais/collect?seconds=${seconds}`, { method: 'POST' })).json();
  } catch (err) {
    console.warn('AIS collection failed', err);
    return null;
  }
}

/* --- orchestration ------------------------------------------------------ */

function setStats(rows) {
  document.getElementById('stats').innerHTML = rows
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`)
    .join('');
}

async function refresh() {
  const button = document.getElementById('refresh');
  button.setAttribute('aria-busy', 'true');
  const days = document.getElementById('days').value;
  bounds = null;

  const tasks = [
    ['SAR dark', () => loadGrid('/api/live/sar', layers.sar, COLORS.dark, 'SAR dark detections', days)],
    ['Fishing effort', () => loadGrid('/api/live/fishing', layers.fishing, COLORS.fishing, 'Apparent fishing hours', days)],
    ['AIS presence', () => loadGrid('/api/live/presence', layers.presence, COLORS.presence, 'AIS presence hours', days)],
    ['Events', () => loadEvents(days)],
    ['EEZ', () => loadEEZ()],
    ['Live vessels', async () => {
      const collected = await collectLiveAIS(8);
      const n = await loadVessels(4320);
      return collected ? `${n} (+${collected.positions_written} new fixes)` : n;
    }],
  ];

  const results = await Promise.allSettled(tasks.map(([, fn]) => fn()));
  const rows = [];
  const failures = [];
  results.forEach((r, i) => {
    const label = tasks[i][0];
    if (r.status === 'fulfilled') {
      rows.push([label, r.value]);
    } else {
      rows.push([label, '—']);
      failures.push(`${label}: ${r.reason.message}`);
    }
  });
  rows.push(['Updated', new Date().toISOString().slice(11, 19) + ' UTC']);
  setStats(rows);

  if (failures.length) {
    console.error(failures);
    toast(failures[0]);
  }
  if (bounds && bounds.isValid() && !selectedMmsi && !refresh._framed) {
    refresh._framed = true;
    map.fitBounds(bounds, { padding: [60, 60], maxZoom: 9 });
  }
  button.removeAttribute('aria-busy');
}

function bind(id, layer) {
  const input = document.getElementById(id);
  input.addEventListener('change', () =>
    input.checked ? map.addLayer(layer) : map.removeLayer(layer)
  );
  if (!input.checked) map.removeLayer(layer);
}

bind('layer-eez', layers.eez);
bind('layer-sar', layers.sar);
bind('layer-fishing', layers.fishing);
bind('layer-presence', layers.presence);
bind('layer-events', layers.events);
bind('layer-vessels', layers.vessels);

document.getElementById('refresh').addEventListener('click', refresh);
document.getElementById('days').addEventListener('change', refresh);

refresh();
// GFW caches server-side for 5 minutes, so polling faster buys nothing.
setInterval(refresh, 300000);
