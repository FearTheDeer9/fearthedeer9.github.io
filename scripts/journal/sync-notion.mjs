// Pulls the daily-log pages from Harel's personal Notion into private/journal/days/, one markdown file per day.
// Read-only against Notion. Only pages whose title starts with "Daily Log" are fetched.
//
//   node scripts/journal/sync-notion.mjs                 new and edited days since the last run
//   node scripts/journal/sync-notion.mjs --export <dir>  load an old Notion markdown export instead (no token)
//
// Token: NOTION_TOKEN, or the macOS Keychain item "notion-journal", which Harel creates with
//   security add-generic-password -s notion-journal -a "$USER" -w
// (it prompts for the token, so the token never appears in shell history).
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

const OUT = 'private/journal/days';
const STATE = 'private/journal/sync-state.json';
mkdirSync(OUT, { recursive: true });

// "Daily Log 18.5.25", "Daily Log6.1.23" -> 2025-05-18
function dateOf(title) {
  const m = title.match(/daily log\s*(\d{1,2})\.(\d{1,2})\.(\d{2,4})/i);
  if (!m) return null;
  const y = m[3].length === 2 ? `20${m[3]}` : m[3];
  return `${y}-${m[2].padStart(2, '0')}-${m[1].padStart(2, '0')}`;
}

const exportDir = process.argv[process.argv.indexOf('--export') + 1];
if (process.argv.includes('--export')) {
  let n = 0;
  for (const f of readdirSync(exportDir)) {
    const date = dateOf(f);
    if (!date || !f.endsWith('.md')) continue;
    const md = readFileSync(join(exportDir, f), 'utf8')
      .replace(/^(\s*)[-*]\s+\[([ xX])\]/gm, '$1[$2]'); // Notion exports to-dos as "- [x]"; normalise to "[x]"
    writeFileSync(join(OUT, `${date}.md`), md);
    n++;
  }
  console.log(`${n} days loaded from the export`);
  process.exit(0);
}

const token = process.env.NOTION_TOKEN ?? (() => {
  try { return execFileSync('security', ['find-generic-password', '-s', 'notion-journal', '-w'], { encoding: 'utf8' }).trim(); }
  catch { console.error('No Notion token: set NOTION_TOKEN or add the Keychain item "notion-journal" (see the top of this file).'); process.exit(1); }
})();

async function api(path, body) {
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(`https://api.notion.com/v1/${path}`, {
      method: body ? 'POST' : 'GET',
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
    if (!date || state[p.id] === p.last_edited_time) continue;
    writeFileSync(join(OUT, `${date}.md`), (await toMarkdown(p.id)).join('\n') + '\n');
    state[p.id] = p.last_edited_time;
    fetched++;
  }
  cursor = res.has_more ? res.next_cursor : null;
} while (cursor);
writeFileSync(STATE, JSON.stringify(state, null, 1));
console.log(`${fetched} days fetched or updated`);
