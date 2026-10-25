// Pulls the daily-log pages from Harel's personal Notion into private/journal/days/, one markdown file per day.
// Read-only against Notion. Only pages whose title starts with "Daily Log" are fetched.
//
//   node scripts/journal/sync-notion.mjs                 new and edited days since the last run
//   node scripts/journal/sync-notion.mjs --init-scorecard <page-url>   create the "Scorecard" database under that page, once
//   node scripts/journal/sync-notion.mjs --push-scorecard <file.json>  replace its rows (old rows go to Notion's trash)
//
// Each run also reads the "Scorecard" database (Pursuits and Drains) into private/journal/scorecard.json.
//
// Token: NOTION_TOKEN, or the macOS Keychain item "notion-journal", which Harel creates with
//   security add-generic-password -s notion-journal -a "$USER" -w
// (it prompts for the token, so the token never appears in shell history).
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { LOCK_IN } from './score.mjs';

const OUT = 'private/journal/days';
const STATE = 'private/journal/sync-state.json';
const CONFIG = 'private/journal/notion.json';
mkdirSync(OUT, { recursive: true });

// "Daily Log 18.5.25", "Daily Log6.1.23" -> 2025-05-18
function dateOf(title) {
  const m = title.match(/daily log\s*(\d{1,2})\.(\d{1,2})\.(\d{2,4})/i);
  if (!m) return null;
  const y = m[3].length === 2 ? `20${m[3]}` : m[3];
  return `${y}-${m[2].padStart(2, '0')}-${m[1].padStart(2, '0')}`;
}

const token = process.env.NOTION_TOKEN ?? (() => {
  try { return execFileSync('security', ['find-generic-password', '-s', 'notion-journal', '-w'], { encoding: 'utf8' }).trim(); }
  catch { console.error('No Notion token: set NOTION_TOKEN or add the Keychain item "notion-journal" (see the top of this file).'); process.exit(1); }
})();

async function api(path, body, method) {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`https://api.notion.com/v1/${path}`, {
      method: method ?? (body ? 'POST' : 'GET'),
      headers: { Authorization: `Bearer ${token}`, 'Notion-Version': '2022-06-28', 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 429 && attempt < 5) { await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt)); continue; }
    if (!res.ok) throw new Error(`Notion ${path}: ${res.status} ${await res.text()}`);
    return res.json();
  }
}

const text = (rt = []) => rt.map((t) => t.plain_text).join('');
async function toMarkdown(blockId, depth = 0) {
  const lines = [];
  let cursor;
  do {
    const page = await api(`blocks/${blockId}/children?page_size=100${cursor ? `&start_cursor=${cursor}` : ''}`);
    for (const b of page.results) {
      const v = b[b.type] ?? {}, pad = '  '.repeat(depth);
      const line = {
        heading_1: `# ${text(v.rich_text)}`, heading_2: `## ${text(v.rich_text)}`, heading_3: `### ${text(v.rich_text)}`,
        paragraph: text(v.rich_text), bulleted_list_item: `- ${text(v.rich_text)}`, numbered_list_item: `1. ${text(v.rich_text)}`,
        to_do: `[${v.checked ? 'x' : ' '}] ${text(v.rich_text)}`, toggle: `- ${text(v.rich_text)}`,
        quote: `> ${text(v.rich_text)}`, callout: text(v.rich_text),
      }[b.type];
      if (line !== undefined) lines.push(pad + line);
      if (b.has_children && b.type !== 'child_page') lines.push(...(await toMarkdown(b.id, depth + 1)));
    }
    cursor = page.has_more ? page.next_cursor : null;
  } while (cursor);
  return lines;
}

// ---- Scorecard: the Notion database Harel edits; one row per Pursuit or Drain. ----
const SIDES = ['pursuit', 'drain'];
const UNITS = ['per-day', 'per-item', 'per-hour', 'per-10-min', 'per-10-pages', 'share'];
const SOURCE_NAMES = ['checklist', 'checklist-minutes', 'anki', 'vault-notes', 'vault-writing', 'activitywatch', 'workout', 'section', 'goals', 'tv', 'articles', 'book', 'bonus'];
const select = (names) => ({ select: { options: names.map((name) => ({ name })) } });
const rich = (t) => ({ rich_text: t ? [{ text: { content: t } }] : [] });
const props = (r) => ({
  Name: { title: [{ text: { content: r.name } }] }, Side: { select: { name: r.side } }, Points: { number: r.points }, Cap: { number: r.cap ?? null },
  Unit: { select: { name: r.unit } }, Source: { select: { name: r.source } }, Match: rich(r.match), Except: rich(r.except), Active: { checkbox: r.active ?? true },
});

if (process.argv.includes('--push-scorecard')) {
  const rows = JSON.parse(readFileSync(process.argv[process.argv.indexOf('--push-scorecard') + 1], 'utf8'));
  const id = JSON.parse(readFileSync(CONFIG, 'utf8')).scorecard;
  let old = [], cursor;
  do { const r = await api(`databases/${id}/query`, { start_cursor: cursor, page_size: 100 }); old.push(...r.results); cursor = r.has_more ? r.next_cursor : null; } while (cursor);
  for (const p of old) await api(`pages/${p.id}`, { archived: true }, 'PATCH');
  for (const r of rows) await api('pages', { parent: { database_id: id }, properties: props(r) });
  console.log(`Scorecard replaced: ${old.length} old rows moved to Notion's trash, ${rows.length} rows written`);
  process.exit(0);
}

if (process.argv.includes('--init-scorecard')) {
  const url = process.argv[process.argv.indexOf('--init-scorecard') + 1] ?? '';
  const parent = url.replace(/-/g, '').match(/[0-9a-f]{32}/i)?.[0];
  if (!parent) { console.error('Pass the URL of the Notion page to create the Scorecard under.'); process.exit(1); }
  const db = await api('databases', {
    parent: { type: 'page_id', page_id: parent }, title: [{ text: { content: 'Scorecard' } }],
    properties: { Name: { title: {} }, Side: select(SIDES), Points: { number: {} }, Cap: { number: {} }, Unit: select(UNITS),
      Source: select(SOURCE_NAMES), Match: { rich_text: {} }, Except: { rich_text: {} }, Active: { checkbox: {} } },
  });
  const seed = JSON.parse(readFileSync('scripts/journal/scorecard.default.json', 'utf8'));
  for (const r of seed) await api('pages', { parent: { database_id: db.id }, properties: props(r) });
  writeFileSync(CONFIG, JSON.stringify({ scorecard: db.id }, null, 1));
  console.log(`Scorecard created with ${seed.length} rows: ${db.url}`);
  process.exit(0);
}

async function readScorecard() {
  // The id saved at creation; search is only a fallback, since Notion indexes new databases with a delay.
  const saved = existsSync(CONFIG) ? JSON.parse(readFileSync(CONFIG, 'utf8')).scorecard : null;
  const db = saved ? { id: saved } : (await api('search', { query: 'Scorecard', filter: { property: 'object', value: 'database' } }))
    .results.find((d) => text(d.title) === 'Scorecard');
  if (!db) return console.log('No "Scorecard" database shared with the integration yet; scoring uses the default scorecard.');
  const rows = [];
  let cursor;
  do {
    const res = await api(`databases/${db.id}/query`, { start_cursor: cursor, page_size: 100 });
    for (const p of res.results) {
      const v = p.properties;
      rows.push({ name: text(v.Name?.title), side: v.Side?.select?.name ?? 'pursuit', points: v.Points?.number ?? 0, cap: v.Cap?.number ?? null,
        unit: v.Unit?.select?.name ?? 'per-day', source: v.Source?.select?.name ?? 'checklist', match: text(v.Match?.rich_text), except: text(v.Except?.rich_text),
        active: v.Active?.checkbox ?? true });
    }
    cursor = res.has_more ? res.next_cursor : null;
  } while (cursor);
  writeFileSync('private/journal/scorecard.json', JSON.stringify(rows.filter((r) => r.name), null, 1));
  console.log(`Scorecard: ${rows.length} rows`);
}
await readScorecard();

const TODAY = new Date().toLocaleDateString('en-CA'); // a mistyped title can carry a future date
const state = existsSync(STATE) ? JSON.parse(readFileSync(STATE, 'utf8')) : {};
let fetched = 0, cursor;
do {
  const res = await api('search', {
    query: 'Daily Log', filter: { property: 'object', value: 'page' },
    sort: { direction: 'descending', timestamp: 'last_edited_time' }, start_cursor: cursor, page_size: 100,
  });
  for (const p of res.results) {
    const title = text(Object.values(p.properties ?? {}).find((x) => x.type === 'title')?.title);
    const date = dateOf(title);
    // Days before lock-in are never scored, so they're never fetched.
    if (!date || date < LOCK_IN || date > TODAY || state[p.id] === p.last_edited_time) continue;
    writeFileSync(join(OUT, `${date}.md`), (await toMarkdown(p.id)).join('\n') + '\n');
    state[p.id] = p.last_edited_time;
    fetched++;
  }
  cursor = res.has_more ? res.next_cursor : null;
} while (cursor);
writeFileSync(STATE, JSON.stringify(state, null, 1));
console.log(`${fetched} days fetched or updated`);
