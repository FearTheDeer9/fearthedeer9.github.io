// Scores each day of the private daily log and writes the public score series.
//
//   node scripts/journal/score.mjs
//
// Reads  private/journal/days/YYYY-MM-DD.md   (one daily log per file, from sync-notion.mjs)
// Writes private/journal/breakdown.json       (per-habit points: private, never committed)
//        src/data/scores.json                 (date, score, grade only: public)
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

// ---- The formula. Points per occurrence, and the most a habit can earn in one day. ----
export const RULES = {
  anki:     { points: 10, cap: 10 },
  paper:    { points: 8, cap: 16 },  // a research paper in Articles Read
  note:     { points: 6, cap: 18 },  // a note created that day in the vault (AI Theory, Daily Learning)
  writing:  { points: 15, cap: 15 }, // a good writing session, or a Time Block Writing entry that day
  walk:     { points: 8, cap: 8 },   // contemplative walk
  meditate: { points: 8, cap: 8 },
  lesson:   { points: 10, cap: 10 }, // chess or any deliberate lesson
  workout:  { points: 10, cap: 10 },
  book:     { perPages: 10, cap: 8, noPages: 3 }, // 1 point per 10 pages; 3 if pages weren't logged
  goals:    { bonus: 15 },           // 15 × share of "Today will be good if" items ticked
  // Drains (negative). Sport you follow and big live events are neutral.
  tv:       { points: -6 },
  sportClip:{ points: -2 },          // highlights, recaps, sports podcasts
  social:   { perTenMin: -1 },
  fantasy:  { points: -8 },
  streak:   { perDay: 2, cap: 10, minScore: 40 }, // consecutive days scoring at least B
};
export const GRADES = [[80, 'S'], [60, 'A'], [40, 'B'], [20, 'C'], [-Infinity, 'D']];

const NEUTRAL_SPORT = /\b(nfl|packers?|green bay|bucks|milwaukee|brewers|super bowl|playoffs?|finals?|world cup|olympics?|wimbledon|us open|grand slam|el clasico|champions league final)\b/i;
const SPORT_CLIP = /\b(highlights?|recap|pod(cast)?|bs pod|bill simmons|lowe|heavy hands|thinking basketball|kram|ringer|nba|ranking|power rankings|mock draft)\b/i;
const PAPER = /\b(arxiv|paper|et al\.?)\b|\d{4}\.\d{4,5}/i;

const VAULT = process.env.VAULT_DIR ?? `${process.env.HOME}/Library/Mobile Documents/iCloud~md~obsidian/Documents/PKM`;
const DAYS = 'private/journal/days';

// ---- Parsing: a day file is markdown with the daily-log template's headings. ----
const SECTION = [
  [/today will be good if/i, 'goals'], [/^habits/i, 'habits'], [/screen time/i, 'screen'], [/^food/i, 'food'],
  [/^workout/i, 'workout'], [/articles? read/i, 'articles'], [/books? read/i, 'book'], [/tv|movie/i, 'tv'], [/other notes/i, 'notes'],
];
export function parseDay(md) {
  const out = {};
  let cur = 'preamble';
  for (const raw of md.replace(/^---\n[\s\S]*?\n---\n/, '').split('\n')) {
    const h = raw.match(/^#{1,6}\s*(.+?)\s*:?\s*$/);
    if (h) { cur = SECTION.find(([re]) => re.test(h[1]))?.[1] ?? 'other'; continue; }
    const line = raw.replace(/^\s*(?:[-*+]|\d+[.)])\s*/, '').trim();
    if (!line || /^(none|-|n\/a)$/i.test(line)) continue;
    (out[cur] ||= []).push(line);
  }
  return out;
}
const checked = (line) => /^\[x\]|☑️|✅|✔/i.test(line);
const label = (line) => line.replace(/^\[[ x]\]\s*/i, '').replace(/☑️|✅|✔️?/g, '').trim();

// Signals from the vault, by date only; file contents are never read here.
// Notes taken: notes created in AI Theory/ or Daily Learning/. Writing sessions: Time Block Writing entries.
function vaultDays() {
  const notes = {}, writing = {};
  const dated = (name, p) => {
    const m = name.match(/(\d{1,2})\.(\d{1,2})\.(\d{2})\b/); // "Time Block Writing 22.10.26"
    return m ? `20${m[3]}-${m[2].padStart(2, '0')}-${m[1].padStart(2, '0')}` : statSync(p).birthtime.toISOString().slice(0, 10);
  };
  (function walk(dir, rel) {
    if (!existsSync(dir)) return;
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, e.name), r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) { if (!e.name.startsWith('.')) walk(p, r); continue; }
      if (!e.name.endsWith('.md')) continue;
      if (/(^|\/)Time Block Writing\//.test(r)) { const d = dated(e.name, p); writing[d] = true; }
      else if (/^(AI Theory|Daily Learning)\//.test(r)) { const d = dated(e.name, p); notes[d] = (notes[d] ?? 0) + 1; }
    }
  })(VAULT, '');
  return { notes, writing };
}

export function scoreDay(day, ctx) {
  const pts = {};
  const add = (k, v, cap = Infinity) => { pts[k] = Math.min((pts[k] ?? 0) + v, cap); };
  const text = [...(day.notes ?? []), ...(day.preamble ?? []), ...(day.other ?? [])].join('\n');
  const habits = (day.habits ?? []).filter(checked).map((l) => label(l).toLowerCase());
  const has = (re) => habits.some((h) => re.test(h));
  const hasChecklist = !!day.habits;

  // Checklist first; free text only on days without the checklist (the backfill).
  const habit = (key, re, textRe) => { if (has(re) || (!hasChecklist && textRe?.test(text))) add(key, RULES[key].points, RULES[key].cap); };
  habit('anki', /^anki/, /\banki\b/i);
  habit('meditate', /^meditat/, /\bmeditat/i);
  habit('walk', /walk/, /\bwalk(ed|ing)?\b/i);
  habit('writing', /writing/, /\bwriting session\b/i);
  habit('lesson', /^lesson/, /\b(chess|lesson|lecture|course)\b/i);
  if (has(/fantasy/) || (!hasChecklist && /\bfantasy\b/i.test(text))) add('fantasy', RULES.fantasy.points);

  // Goals the day was set up around ("Today will be good if"), which also count walks and workouts ticked there.
  const goals = day.goals ?? [];
  if (goals.length) {
    pts.goals = Math.round((RULES.goals.bonus * goals.filter(checked).length) / goals.length);
    for (const g of goals.filter(checked).map((g) => label(g).toLowerCase())) {
      if (/walk/.test(g) && !pts.walk) add('walk', RULES.walk.points, RULES.walk.cap);
      if (/medita/.test(g) && !pts.meditate) add('meditate', RULES.meditate.points, RULES.meditate.cap);
      if (/anki/.test(g) && !pts.anki) add('anki', RULES.anki.points, RULES.anki.cap);
    }
  }
  if ((day.workout ?? []).length) add('workout', RULES.workout.points, RULES.workout.cap);

  for (const a of day.articles ?? []) {
    if (PAPER.test(a)) add('paper', RULES.paper.points, RULES.paper.cap);
    else if (SPORT_CLIP.test(a) && !NEUTRAL_SPORT.test(a)) add('sportClip', RULES.sportClip.points);
  }
  for (const t of day.tv ?? []) {
    if (NEUTRAL_SPORT.test(t)) continue;
    add(SPORT_CLIP.test(t) ? 'sportClip' : 'tv', SPORT_CLIP.test(t) ? RULES.sportClip.points : RULES.tv.points);
  }
  const social = (day.screen ?? []).map((l) => l.match(/social media[^0-9]*(\d+)/i)?.[1]).find(Boolean);
  if (social) add('social', Math.round(+social / 10) * RULES.social.perTenMin);

  // Book pages: the page reached minus the last page logged for that book.
  for (const b of day.book ?? []) {
    const m = b.match(/^(.*?)[,\s-]*(?:p\.?|page)\s*(\d+)/i) ?? b.match(/^(.*?)\s+(\d+)\s*$/);
    if (!m) { add('book', RULES.book.noPages, RULES.book.cap); continue; }
    const key = m[1].trim().toLowerCase(), page = +m[2], prev = ctx.lastPage[key];
    ctx.lastPage[key] = page;
    add('book', prev !== undefined && page > prev ? Math.floor((page - prev) / RULES.book.perPages) : RULES.book.noPages, RULES.book.cap);
  }
  if (ctx.notes) add('note', Math.min(ctx.notes * RULES.note.points, RULES.note.cap), RULES.note.cap);
  if (ctx.writing && !pts.writing) add('writing', RULES.writing.points, RULES.writing.cap);

  const raw = Object.values(pts).reduce((a, b) => a + b, 0);
  return { pts, raw, hasChecklist };
}

// ---- Run over every day, in order (book pages and streaks depend on the previous days). ----
if (import.meta.url === `file://${process.argv[1]}`) {
  const files = existsSync(DAYS) ? readdirSync(DAYS).filter((f) => /^\d{4}-\d{2}-\d{2}\.md$/.test(f)).sort() : [];
  const vault = vaultDays();
  const ctx = { lastPage: {} };
  let streak = 0;
  const detail = [], series = [];
  for (const f of files) {
    const date = f.slice(0, 10);
    const { pts, raw, hasChecklist } = scoreDay(parseDay(readFileSync(join(DAYS, f), 'utf8')), { ...ctx, notes: vault.notes[date], writing: vault.writing[date] });
    const bonus = Math.min(streak * RULES.streak.perDay, RULES.streak.cap);
    const score = Math.max(0, raw + bonus);
    streak = raw + bonus >= RULES.streak.minScore ? streak + 1 : 0;
    const grade = GRADES.find(([min]) => score >= min)[1];
    detail.push({ date, score, grade, streak: bonus, ...pts });
    series.push({ date, score, grade, baseline: !hasChecklist });
  }
  mkdirSync('private/journal', { recursive: true });
  writeFileSync('private/journal/breakdown.json', JSON.stringify(detail, null, 1));
  mkdirSync('src/data', { recursive: true });
  writeFileSync('src/data/scores.json', JSON.stringify(series) + '\n');
  const g = series.reduce((c, s) => ((c[s.grade] = (c[s.grade] ?? 0) + 1), c), {});
  const mean = series.length ? Math.round(series.reduce((a, s) => a + s.score, 0) / series.length) : 0;
  console.log(`${series.length} days scored, mean ${mean}, grades ${JSON.stringify(g)}`);
}
