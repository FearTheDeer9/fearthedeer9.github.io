// Gathers the automatically captured signals for each day since lock-in into private/journal/signals/<date>.json.
//
//   node scripts/journal/collect.mjs
//
// Sources, all local and read-only:
//   anki           reviews per day, from Anki's collection file (opened read-only, so Anki can stay open)
//   vault-notes    notes created in AI Theory/ or Daily Learning/ (file dates only, never contents)
//   vault-writing  Time Block Writing entries (file dates only)
//   activitywatch  minutes per app and per window title, from ActivityWatch's local API (if it's running)
import { writeFileSync, readdirSync, statSync, existsSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { LOCK_IN } from './score.mjs';

const HOME = process.env.HOME;
const VAULT = process.env.VAULT_DIR ?? `${HOME}/Library/Mobile Documents/iCloud~md~obsidian/Documents/PKM`;
const ANKI = process.env.ANKI_COLLECTION ?? `${HOME}/Library/Application Support/Anki2/User 1/collection.anki2`;
const AW = 'http://localhost:5600/api/0';
const OUT = 'private/journal/signals';
mkdirSync(OUT, { recursive: true });

const today = new Date().toLocaleDateString('en-CA');
const dates = [];
for (let d = new Date(LOCK_IN); d.toLocaleDateString('en-CA') <= today; d.setDate(d.getDate() + 1)) dates.push(d.toLocaleDateString('en-CA'));
const sig = Object.fromEntries(dates.map((d) => [d, { anki: 0, 'vault-notes': 0, 'vault-writing': 0, activitywatch: null }]));

// Anki: revlog ids are millisecond timestamps.
if (existsSync(ANKI)) {
  const rows = execFileSync('sqlite3', [`file:${ANKI}?immutable=1&mode=ro`,
    `select date(id/1000,'unixepoch','localtime'), count(*) from revlog where id >= strftime('%s','${LOCK_IN}')*1000 group by 1;`], { encoding: 'utf8' });
  for (const line of rows.trim().split('\n').filter(Boolean)) { const [d, n] = line.split('|'); if (sig[d]) sig[d].anki = +n; }
}

// Vault: file dates only.
const dated = (name, p) => {
  const m = name.match(/(\d{1,2})\.(\d{1,2})\.(\d{2})\b/); // "Time Block Writing 22.10.26"
  return m ? `20${m[3]}-${m[2].padStart(2, '0')}-${m[1].padStart(2, '0')}` : statSync(p).birthtime.toLocaleDateString('en-CA');
};
(function walk(dir, rel) {
  if (!existsSync(dir)) return;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name), r = rel ? `${rel}/${e.name}` : e.name;
    if (e.isDirectory()) { if (!e.name.startsWith('.')) walk(p, r); continue; }
    if (!e.name.endsWith('.md')) continue;
    const key = /(^|\/)Time Block Writing\//.test(r) ? 'vault-writing' : /^(AI Theory|Daily Learning)\//.test(r) ? 'vault-notes' : null;
    if (!key) continue;
    const d = dated(e.name, p);
    if (sig[d]) sig[d][key] = key === 'vault-writing' ? 1 : sig[d][key] + 1;
  }
})(VAULT, '');

// ActivityWatch: active (not AFK) window time per app and title. Skipped quietly if the app isn't running.
async function activity(date) {
  const buckets = await fetch(`${AW}/buckets/`).then((r) => r.json());
  const win = Object.keys(buckets).find((b) => b.startsWith('aw-watcher-window'));
  const afk = Object.keys(buckets).find((b) => b.startsWith('aw-watcher-afk'));
  if (!win || !afk) return null;
  const start = new Date(`${date}T00:00:00`), end = new Date(start); end.setDate(end.getDate() + 1);
  // Raw events, intersected here: ActivityWatch 0.13's filter_keyvals query function errors server-side.
  const events = async (bucket) => {
    const res = await fetch(`${AW}/buckets/${bucket}/events?start=${start.toISOString()}&end=${end.toISOString()}&limit=-1`).then((r) => r.json());
    if (!Array.isArray(res)) throw new Error(`ActivityWatch events failed: ${res.message ?? JSON.stringify(res)}`);
    return res.map((e) => ({ data: e.data, from: Date.parse(e.timestamp), to: Date.parse(e.timestamp) + e.duration * 1000 }));
  };
  const active = (await events(afk)).filter((e) => e.data.status === 'not-afk');
  const seconds = {};
  for (const w of await events(win)) {
    let overlap = 0;
    for (const a of active) overlap += Math.max(0, Math.min(w.to, a.to) - Math.max(w.from, a.from));
    if (!overlap) continue;
    const key = `${w.data.app} — ${String(w.data.title ?? '').slice(0, 120)}`;
    seconds[key] = (seconds[key] ?? 0) + overlap / 1000;
  }
  // Keep "app — title" with minutes; enough to match a row's pattern against.
  return Object.fromEntries(Object.entries(seconds).filter(([, s]) => s >= 30).map(([k, s]) => [k, Math.round(s / 60)]));
}
let awUp = true, awError = '';
for (const d of dates) {
  if (awUp) {
    try { sig[d].activitywatch = await activity(d); }
    catch (e) { awUp = false; awError = e.cause?.code === 'ECONNREFUSED' ? 'not running' : e.message; }
  }
  writeFileSync(join(OUT, `${d}.json`), JSON.stringify(sig[d], null, 1));
}
console.log(`${dates.length} days collected (${dates[0]} → ${dates.at(-1)}); ActivityWatch ${awUp ? 'read' : `skipped (${awError})`}`);
