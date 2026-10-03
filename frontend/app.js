/* ============================================================
   Diabetes risk estimator
   The random forest trained in Python is exported to model.json as
   arrays of [feature, threshold, left, right, leafProbability].
   Scoring below walks those trees, so predictions match sklearn.
   ============================================================ */

const MM_PER_INCH = 25.4;          // dataset stores skinfold in inches ("skin")
const IN_PER_MM = 1 / MM_PER_INCH;

/* ---------- field definitions (UI order, not model order) ---------- */

const FIELDS = [
  { id: 'age', feature: 'age', group: 'about',
    name: 'Age', hint: 'Years. The cohort starts at 21.',
    unit: 'years', min: 21, max: 85, step: 1, dec: 0 },

  { id: 'num_preg', feature: 'num_preg', group: 'about',
    name: 'Times pregnant', hint: 'Every record in the dataset is a woman.',
    unit: '', min: 0, max: 17, step: 1, dec: 0 },

  { id: 'glucose_conc', feature: 'glucose_conc', group: 'labs',
    name: 'Plasma glucose', hint: 'Two hours into an oral glucose tolerance test.',
    unit: 'mg/dL', min: 44, max: 199, step: 1, dec: 0 },

  { id: 'insulin', feature: 'insulin', group: 'labs',
    name: 'Serum insulin', hint: 'Two-hour reading. Often not taken.',
    unit: 'µU/mL', min: 14, max: 846, step: 1, dec: 0, optional: true },

  { id: 'bmi', feature: 'bmi', group: 'body',
    name: 'Body mass index', hint: 'Weight in kg divided by height in metres squared.',
    unit: 'kg/m²', min: 18, max: 67, step: 0.1, dec: 1 },

  { id: 'skin', feature: 'skin', group: 'body',
    name: 'Triceps skinfold', hint: 'Skin fold thickness, measured with callipers.',
    unit: 'mm', min: 7, max: 99, step: 1, dec: 0, optional: true,
    toModel: mm => mm * IN_PER_MM,
    fromModel: inch => inch * MM_PER_INCH },

  { id: 'diastolic_bp', feature: 'diastolic_bp', group: 'body',
    name: 'Diastolic blood pressure', hint: 'The lower of the two numbers.',
    unit: 'mm Hg', min: 24, max: 122, step: 1, dec: 0, optional: true },

  { id: 'diab_pred', feature: 'diab_pred', group: 'family',
    name: 'Family history score', hint: 'Diabetes pedigree function. Higher means more affected relatives.',
    unit: '', min: 0.08, max: 2.42, step: 0.01, dec: 2 }
];

/* ---------- example records taken from the dataset ---------- */

const PRESETS = [
  { label: 'Low-risk record (tested negative)',
    values: { age: 22, num_preg: 2, glucose_conc: 90, insulin: null,
              bmi: 27.3, skin: 17, diastolic_bp: 70, diab_pred: 0.085 } },
  { label: 'Borderline record (tested negative)',
    values: { age: 42, num_preg: 9, glucose_conc: 106, insulin: null,
              bmi: 31.2, skin: null, diastolic_bp: 52, diab_pred: 0.38 } },
  { label: 'High-risk record (tested positive)',
    values: { age: 32, num_preg: 9, glucose_conc: 164, insulin: null,
              bmi: 30.8, skin: 21, diastolic_bp: 84, diab_pred: 0.831 } }
];

const BANDS = [
  { max: 0.25, key: 'low',  label: 'Low',
    note: 'Most people with this profile in the dataset tested negative.' },
  { max: 0.50, key: 'mid',  label: 'Moderate',
    note: 'Mixed. The model sees this profile on both sides of the line.' },
  { max: 1.01, key: 'high', label: 'High',
    note: 'The model would classify this profile as positive. Worth a real test.' }
];

/* ---------- state ---------- */

let MODEL = null;
let predictionRequestId = 0;
const state = {};        // id -> { value (display units), unknown }
const els = {};          // id -> { input, range, checkbox, row }

/* ---------- scoring ---------- */

function predict(vector) {
  let total = 0;
  for (const nodes of MODEL.trees) {
    let i = 0;
    while (nodes[i][0] !== -1) {
      const n = nodes[i];
      i = vector[n[0]] <= n[1] ? n[2] : n[3];
    }
    total += nodes[i][4];
  }
  return total / MODEL.trees.length;
}

// Build the 8-length vector the model expects, in model feature order.
function buildVector(overrides = {}) {
  return MODEL.features.map(feature => {
    if (feature in overrides) return overrides[feature];
    const field = FIELDS.find(f => f.feature === feature);
    const s = state[field.id];
    if (s.unknown) return MODEL.means[feature];
    return field.toModel ? field.toModel(s.value) : s.value;
  });
}

/* ---------- helpers ---------- */

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const fmt = (v, dec) => Number(v).toFixed(dec);
const median = field => {
  const p50 = MODEL.dist[field.feature].p50;
  return field.fromModel ? field.fromModel(p50) : p50;
};
const bandFor = p => BANDS.find(b => p < b.max);

/* ---------- form construction ---------- */

function buildForm() {
  for (const field of FIELDS) {
    const row = document.createElement('div');
    row.className = 'field';
    row.innerHTML = `
      <label class="field-label" for="in-${field.id}">
        <span class="name">${field.name}</span>
        <span class="hint">${field.hint}</span>
      </label>
      <div class="field-value">
        <input type="number" id="in-${field.id}" min="${field.min}" max="${field.max}"
               step="${field.step}" inputmode="decimal">
        <span class="unit">${field.unit}</span>
      </div>
      <div class="field-slider">
        <input type="range" id="rg-${field.id}" min="${field.min}" max="${field.max}"
               step="${field.step}" aria-label="${field.name} slider" tabindex="-1">
      </div>
      ${field.optional ? `
      <label class="unknown">
        <input type="checkbox" id="ck-${field.id}">
        Not measured — handled by the saved model
      </label>` : ''}
    `;
    document.querySelector(`.fields[data-group="${field.group}"]`).appendChild(row);

    const input = row.querySelector('input[type="number"]');
    const range = row.querySelector('input[type="range"]');
    const check = row.querySelector('input[type="checkbox"]');
    els[field.id] = { row, input, range, check };

    // mark the cohort median on the slider track
    const pos = (median(field) - field.min) / (field.max - field.min);
    range.style.setProperty('--median', `${clamp(pos, 0, 1) * 100}%`);

    input.addEventListener('input', () => setValue(field, parseFloat(input.value), 'input'));
    input.addEventListener('blur', () => syncField(field));
    range.addEventListener('input', () => setValue(field, parseFloat(range.value), 'range'));
    if (check) check.addEventListener('change', () => {
      state[field.id].unknown = check.checked;
      clearPreset();
      syncField(field);
      update();
    });
  }
}

function setValue(field, raw, source) {
  if (!Number.isFinite(raw)) return;
  const s = state[field.id];
  s.value = clamp(raw, field.min, field.max);
  if (s.unknown) { s.unknown = false; els[field.id].check.checked = false; }
  clearPreset();
  syncField(field, source);
  update();
}

function syncField(field, source) {
  const s = state[field.id];
  const { input, range, row } = els[field.id];
  const shown = s.unknown
    ? (field.fromModel ? field.fromModel(MODEL.means[field.feature]) : MODEL.means[field.feature])
    : s.value;
  if (source !== 'input') input.value = fmt(shown, field.dec);
  if (source !== 'range') range.value = shown;
  range.disabled = s.unknown;
  row.classList.toggle('is-unknown', s.unknown);
}

function applyValues(values) {
  for (const field of FIELDS) {
    const v = values[field.id];
    const s = state[field.id];
    if (v === null || v === undefined) {
      s.unknown = !!field.optional;
      if (!field.optional) s.value = median(field);
    } else {
      s.unknown = false;
      s.value = clamp(v, field.min, field.max);
    }
    if (els[field.id].check) els[field.id].check.checked = s.unknown;
    syncField(field);
  }
  update();
}

/* ---------- presets ---------- */

function buildPresets() {
  const host = document.getElementById('presets');
  PRESETS.forEach((preset, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'preset';
    b.textContent = preset.label;
    b.setAttribute('aria-pressed', 'false');
    b.dataset.index = i;
    b.addEventListener('click', () => {
      applyValues(preset.values);
      clearPreset();
      b.setAttribute('aria-pressed', 'true');
    });
    host.appendChild(b);
  });
}
const clearPreset = () =>
  document.querySelectorAll('.preset').forEach(b => b.setAttribute('aria-pressed', 'false'));

/* ---------- gauge ---------- */

const CX = 160, CY = 170, R = 110;

function pointAt(p) {
  const a = Math.PI * (1 - clamp(p, 0, 1));
  return [CX + R * Math.cos(a), CY - R * Math.sin(a)];
}
function arc(from, to) {
  const [x1, y1] = pointAt(from), [x2, y2] = pointAt(to);
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

function drawGaugeChrome() {
  document.getElementById('band-low').setAttribute('d', arc(0, 0.25));
  document.getElementById('band-mid').setAttribute('d', arc(0.25, 0.5));
  document.getElementById('band-high').setAttribute('d', arc(0.5, 1));

  const g = document.getElementById('gauge-ticks');
  for (let i = 0; i <= 20; i++) {
    const p = i / 20, major = i % 5 === 0;
    const a = Math.PI * (1 - p);
    const r1 = R + 6, r2 = R + (major ? 15 : 11);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', (CX + r1 * Math.cos(a)).toFixed(2));
    line.setAttribute('y1', (CY - r1 * Math.sin(a)).toFixed(2));
    line.setAttribute('x2', (CX + r2 * Math.cos(a)).toFixed(2));
    line.setAttribute('y2', (CY - r2 * Math.sin(a)).toFixed(2));
    line.setAttribute('class', major ? 'tick tick-major' : 'tick');
    g.appendChild(line);
    if (major) {
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      const rt = R + 27;
      t.setAttribute('x', (CX + rt * Math.cos(a)).toFixed(2));
      t.setAttribute('y', (CY - rt * Math.sin(a) + 4).toFixed(2));
      t.setAttribute('class', 'tick-text');
      t.textContent = Math.round(p * 100);
      g.appendChild(t);
    }
  }
}

/* ---------- render ---------- */

function update() {
  const referenceProbability = predict(buildVector());
  renderDrivers(referenceProbability);
  renderCompare();

  const requestId = ++predictionRequestId;
  const payload = Object.fromEntries(MODEL.features.map(feature => {
    const field = FIELDS.find(item => item.feature === feature);
    const value = state[field.id];
    return [feature, value.unknown ? null : (field.toModel ? field.toModel(value.value) : value.value)];
  }));

  document.getElementById('band-label').textContent = 'Updating prediction';
  document.getElementById('verdict-note').textContent = 'Requesting the saved model from the local API.';

  fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then(async response => {
    const result = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(result.detail)
        ? result.detail.map(issue => issue.msg).join(', ')
        : result.detail;
      throw new Error(detail || `Request failed (${response.status})`);
    }
    return result;
  }).then(result => {
    if (requestId !== predictionRequestId) return;
    const p = Number(result.positive_probability);
    if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error('The API returned an invalid probability.');

    const band = bandFor(p);
    document.querySelector('.instrument').style.setProperty('--risk', `var(--${band.key})`);
    document.getElementById('prob').textContent = (p * 100).toFixed(1);
    document.getElementById('band-label').textContent = `${band.label} estimated risk`;
    document.getElementById('verdict-note').textContent =
      `Saved model classification: ${result.label} (class ${result.prediction}).`;
    document.getElementById('arc-value').setAttribute('d', arc(0, Math.max(p, 0.001)));
    document.getElementById('needle').style.transform = `rotate(${(p * 180 - 90).toFixed(2)}deg)`;
  }).catch(error => {
    if (requestId !== predictionRequestId) return;
    console.error(error);
    document.getElementById('prob').textContent = '—';
    document.getElementById('band-label').textContent = 'Prediction unavailable';
    document.getElementById('verdict-note').textContent = error.message;
    document.getElementById('arc-value').setAttribute('d', '');
  });
}

function renderDrivers(p) {
  const rows = FIELDS.map(field => {
    const atMedian = predict(buildVector({ [field.feature]: MODEL.dist[field.feature].p50 }));
    return { field, delta: p - atMedian };
  }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 5);

  const scale = Math.max(0.04, ...rows.map(r => Math.abs(r.delta)));
  const list = document.getElementById('drivers');
  list.innerHTML = '';

  for (const { field, delta } of rows) {
    const width = (Math.abs(delta) / scale) * 50;
    const up = delta >= 0;
    const li = document.createElement('li');
    li.innerHTML = `
      <span class="dname">${field.name}</span>
      <span class="bar-track">
        <span class="bar-fill ${up ? 'up' : 'down'}"
              style="${up ? 'left:50%' : `right:50%`}; width:${width.toFixed(1)}%"></span>
      </span>
      <span class="dval">${up ? '+' : '−'}${Math.abs(delta * 100).toFixed(1)} pts</span>
    `;
    list.appendChild(li);
  }
}

function renderCompare() {
  const body = document.querySelector('#compare-table tbody');
  body.innerHTML = '';
  for (const field of FIELDS) {
    const d = MODEL.dist[field.feature];
    const conv = v => field.fromModel ? field.fromModel(v) : v;
    const s = state[field.id];
    const yours = s.unknown ? conv(MODEL.means[field.feature]) : s.value;
    const pos = v => clamp((v - conv(d.min)) / (conv(d.max) - conv(d.min)), 0, 1) * 100;

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${field.name}${s.unknown ? ' <span class="hint">(not measured)</span>' : ''}</td>
      <td class="num">${fmt(yours, field.dec)}${field.unit ? ' ' + field.unit : ''}</td>
      <td class="strip-cell">
        <span class="strip">
          <span class="axis"></span>
          <span class="mark neg" style="left:${pos(conv(d.mean_neg)).toFixed(1)}%"></span>
          <span class="mark pos" style="left:${pos(conv(d.mean_pos)).toFixed(1)}%"></span>
          <span class="you" style="left:${pos(yours).toFixed(1)}%"></span>
        </span>
      </td>
      <td class="num">${fmt(conv(d.mean_neg), field.dec)}</td>
      <td class="num">${fmt(conv(d.mean_pos), field.dec)}</td>
    `;
    body.appendChild(tr);
  }
}

function renderModelCard() {
  const m = MODEL.metrics;

  const metrics = [
    ['Accuracy', (m.accuracy * 100).toFixed(1) + '%'],
    ['ROC AUC', m.auc.toFixed(3)],
    ['Recall (positives found)', (m.recall * 100).toFixed(1) + '%'],
    ['Precision', (m.precision * 100).toFixed(1) + '%'],
    ['5-fold accuracy', (m.cv_accuracy * 100).toFixed(1) + '%'],
    ['Records positive', (m.pos_rate * 100).toFixed(1) + '%']
  ];
  document.getElementById('metrics').innerHTML = metrics
    .map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');

  const imps = FIELDS
    .map(f => ({ name: f.name, v: MODEL.importances[f.feature] }))
    .sort((a, b) => b.v - a.v);
  const top = imps[0].v;
  document.getElementById('importances').innerHTML = imps.map(i => `
    <li>
      <span class="iname">${i.name}</span>
      <span class="imp-track"><span class="imp-fill" style="width:${(i.v / top * 100).toFixed(1)}%"></span></span>
      <span class="ival">${(i.v * 100).toFixed(1)}%</span>
    </li>`).join('');
}

/* ---------- boot ---------- */

function start(model) {
  MODEL = model;
  for (const field of FIELDS) state[field.id] = { value: 0, unknown: false };
  buildForm();
  buildPresets();
  drawGaugeChrome();
  renderModelCard();

  const medians = {};
  for (const field of FIELDS) medians[field.id] = median(field);
  applyValues(medians);

  document.getElementById('reset').addEventListener('click', () => {
    applyValues(medians);
    clearPreset();
  });
}

function fail(err) {
  console.error(err);
  document.getElementById('band-label').textContent = 'Cohort reference did not load';
  document.getElementById('verdict-note').textContent =
    'model.json could not be read. Serve this folder over HTTP rather than opening the file directly.';
}

if (window.__MODEL__) {
  start(window.__MODEL__);
} else {
  fetch('model.json').then(r => r.json()).then(start).catch(fail);
}
