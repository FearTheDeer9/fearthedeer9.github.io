// Scores each day since lock-in against the Scorecard: Pursuits earn points, Drains cost them.
//
//   node scripts/journal/score.mjs
//
// Reads  private/journal/scorecard.json        (synced from the Notion "Scorecard" database; falls back to scorecard.default.json)
//        private/journal/days/<date>.md        (the Notion daily log, from sync-notion.mjs)
//        private/journal/signals/<date>.json   (automatic signals, from collect.mjs)
// Writes private/journal/breakdown.json        (points per row: private, never committed)
//        src/data/scores.json                  (date, score, grade only: public)
//        src/data/scorecard.json               (row names and points: public, shown on /progress/)
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// Scoring starts here. Earlier days are never scored: their habits would have to be inferred, not recorded.
export const LOCK_IN = '2026-10-24';
export const GRADES = [[80, 'S'], [60, 'A'], [40, 'B'], [20, 'C'], [-Infinity, 'D']];

const DIR = 'private/journal';

// ---- The daily log: markdown with the template's headings. ----
const SECTION = [
  [/today will be good if/i, 'goals'], [/^habits/i, 'habits'], [/screen time|^numbers/i, 'screen'], [/^food/i, 'food'],
  [/^workout/i, 'workout'], [/articles? read/i, 'articles'], [/books? read/i, 'book'], [/tv|movie/i, 'tv'], [/other notes/i, 'notes'],
];
export function parseDay(md) {
  const out = {};
  let cur = 'preamble';
  for (const raw of md.replace(/^---\n[\s\S]*?\n---\n/, '').split('\n')) {
    const h = raw.match(/^#{1,6}\s*(.+?)\s*:?\s*$/);
    // Known headings get a fixed key; any other heading is kept by name, for rows with the "section" source.
    if (h) { cur = SECTION.find(([re]) => re.test(h[1]))?.[1] ?? `h:${h[1].toLowerCase()}`; continue; }
    const line = raw.replace(/^\s*(?:[-*+]|\d+[.)])\s*/, '').trim();
    if (!line || /^(none|-|n\/a)$/i.test(line)) continue;
    (out[cur] ||= []).push(line);
  }
  return out;
}
const checked = (line) => /^\[x\]|☑️|✅|✔/i.test(line);
const re = (s) => (s ? new RegExp(s, 'i') : null);

// ---- How much of a row happened on a day, before the row's unit is applied. ----
const SOURCES = {
  checklist: (row, day) => ((day.habits ?? []).some((l) => checked(l) && re(row.match || row.name).test(l)) ? 1 : 0),
  // A number typed after a matching line in the Numbers block: "Social media (min): 40", "Sleep (h): 7.5". Hours become minutes.
  'checklist-minutes': (row, day) => {
    const line = (day.screen ?? []).find((l) => re(row.match).test(l)), n = +(line?.match(/([\d.]+)\s*$/)?.[1] ?? 0);
    return /\(h\)|hours?/i.test(line ?? '') ? n * 60 : n;
  },
  anki: (row, day, sig) => ((sig.anki ?? 0) >= (+row.match || 1) ? 1 : 0),
  'vault-notes': (row, day, sig) => sig['vault-notes'] ?? 0,
  'vault-writing': (row, day, sig) => sig['vault-writing'] ?? 0,
  activitywatch: (row, day, sig) => Object.entries(sig.activitywatch ?? {})
    .filter(([k]) => re(row.match).test(k) && !(row.except && re(row.except).test(k))).reduce((n, [, m]) => n + m, 0),
  workout: (row, day) => ((day.workout ?? []).length ? 1 : 0),
  // Lines written under any heading matching the row's pattern, e.g. "Lecture:" or "What I Learned Summary:".
  section: (row, day) => Object.entries(day).filter(([k]) => k.startsWith('h:') && re(row.match).test(k.slice(2))).reduce((n, [, l]) => n + l.length, 0),
  goals: (row, day) => { const g = day.goals ?? []; return g.length ? g.filter(checked).length / g.length : 0; },
  tv: (row, day) => (day.tv ?? []).filter((l) => (!row.match || re(row.match).test(l)) && !(row.except && re(row.except).test(l))).length,
  articles: (row, day) => (day.articles ?? []).filter((l) => (!row.match || re(row.match).test(l)) && !(row.except && re(row.except).test(l))).length,
  // Per-case points Harel writes himself, anywhere in the log: "Bonus 5: cooked dinner with friends".
  bonus: (row, day) => Object.values(day).flat().reduce((n, l) => n + (+(String(l).match(/^bonus\s*[:+]?\s*(\d+)/i)?.[1] ?? 0)), 0),
  // Pages since the last page logged for the same book ("Iron Flame, p. 140"); 30 pages assumed when none are given.
  book: (row, day, sig, ctx) => (day.book ?? []).reduce((n, b) => {
    const m = b.match(/^(.*?)[,\s-]*(?:p\.?|page)\s*(\d+)/i);
    if (!m) return n + 30;
    const key = m[1].trim().toLowerCase(), page = +m[2], prev = ctx.lastPage[key];
    ctx.lastPage[key] = page;
    return n + (prev !== undefined && page > prev ? page - prev : 30);
  }, 0),
};
// Time-based rows (minutes in, points per hour) scale with the time measured; give them no cap.
const UNIT = { 'per-day': (n) => Math.min(n, 1), 'per-item': (n) => n, 'per-hour': (n) => n / 60, 'per-10-min': (n) => Math.floor(n / 10), 'per-10-pages': (n) => Math.floor(n / 10), share: (n) => n };

export function scoreDay(rows, day, sig, ctx) {
  // Rows sharing a name are alternative evidence for one habit (a checkbox or ActivityWatch): the strongest counts once.
  const byName = new Map();
  for (const row of rows.filter((r) => r.active !== false)) {
    const amount = UNIT[row.unit]?.(SOURCES[row.source]?.(row, day, sig, ctx) ?? 0) ?? 0;
    const pts = Math.min(Math.round(amount * row.points), row.cap || Infinity) * (row.side === 'drain' ? -1 : 1);
    const prev = byName.get(row.name);
    if (prev === undefined || Math.abs(pts) > Math.abs(prev)) byName.set(row.name, pts);
  }
  return Object.fromEntries([...byName].filter(([, v]) => v !== 0));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const rows = JSON.parse(readFileSync(existsSync(join(DIR, 'scorecard.json')) ? join(DIR, 'scorecard.json') : 'scripts/journal/scorecard.default.json', 'utf8'));
  const list = (sub, ext) => (existsSync(join(DIR, sub)) ? readdirSync(join(DIR, sub)).filter((f) => f.endsWith(ext)).map((f) => f.slice(0, 10)) : []);
  const logged = new Set(list('days', '.md'));
  const today = new Date().toLocaleDateString('en-CA');
  // A day is scored once it's over or logged, and only from lock-in on.
  const dates = [...new Set([...logged, ...list('signals', '.json')])].filter((d) => d >= LOCK_IN && (d < today || logged.has(d))).sort();
  const ctx = { lastPage: {} };
  const detail = [], series = [];
  for (const date of dates) {
    const md = join(DIR, 'days', `${date}.md`), sj = join(DIR, 'signals', `${date}.json`);
    const day = logged.has(date) ? parseDay(readFileSync(md, 'utf8')) : {};
    const sig = existsSync(sj) ? JSON.parse(readFileSync(sj, 'utf8')) : {};
    const pts = scoreDay(rows, day, sig, ctx);
    const raw = Object.values(pts).reduce((a, b) => a + b, 0);
    const score = Math.max(0, raw);
    const grade = GRADES.find(([min]) => score >= min)[1];
    detail.push({ date, score, grade, logged: logged.has(date), ...pts });
    series.push({ date, score, grade });
  }
  mkdirSync(DIR, { recursive: true });
  writeFileSync(join(DIR, 'breakdown.json'), JSON.stringify(detail, null, 1));
  mkdirSync('src/data', { recursive: true });
  writeFileSync('src/data/scores.json', JSON.stringify(series) + '\n');
  // Public view of the scorecard: one line per habit, the strongest source's points.
  const card = [...new Map(rows.filter((r) => r.active !== false).map((r) => [r.name, { name: r.name, side: r.side, points: r.points, cap: r.cap, unit: r.unit }])).values()];
  writeFileSync('src/data/scorecard.json', JSON.stringify(card, null, 1) + '\n');
  console.log(`${series.length} days scored since ${LOCK_IN}${series.length ? `, latest ${series.at(-1).date}: ${series.at(-1).score} (${series.at(-1).grade})` : ''}`);
}
