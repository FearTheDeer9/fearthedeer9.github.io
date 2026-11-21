// Collects what Harel logged under "Articles Read" and "Book Read" in synced journal days into
// private/reading/journal-additions.json, which scripts/build-reading.mjs merges into the reading log.
// The one-off Notion extraction in reading-all.json ends 2026-10-22; journal days take over from there.
import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { parseDay } from './score.mjs';

const DAYS = 'private/journal/days';
const all = JSON.parse(readFileSync('private/reading/reading-all.json', 'utf8'));
const shows = all.podcasts.map((p) => p.show);
// Journal shorthand for shows, as the extraction resolved them.
const ALIAS = { 'bs pod': 'The Bill Simmons Podcast', bs: 'The Bill Simmons Podcast', lowe: 'The Lowe Post', jre: 'The Joe Rogan Experience', 'heavy hands': 'Heavy Hands', 'latent space': 'Latent Space', dwarkesh: 'Dwarkesh Podcast', lex: 'Lex Fridman Podcast', huberman: 'Huberman Lab' };

function showOf(prefix) {
  const p = prefix.trim().toLowerCase();
  return ALIAS[p] ?? shows.find((s) => s.toLowerCase() === p || s.toLowerCase().replace(/^the /, '').startsWith(p))
    ?? (/\bpod(cast)?\b/.test(p) ? prefix.trim() : null);
}

const out = { articles: [], episodes: [], books: [] };
const days = existsSync(DAYS) ? readdirSync(DAYS).filter((f) => /^\d{4}-\d{2}-\d{2}\.md$/.test(f) && f >= '2026-10-23').sort() : [];
for (const f of days) {
  const date = f.slice(0, 10), day = parseDay(readFileSync(join(DAYS, f), 'utf8'));
  for (const line of day.articles ?? []) {
    const [prefix, ...rest] = line.split(/\s+[-–]\s+/);
    const show = rest.length ? showOf(prefix) : null;
    if (show) out.episodes.push({ show, title: rest.join(' - '), date });
    else out.articles.push({ title: line.replace(/^(youtube( video)?|video)\s*[-–:]\s*/i, ''), medium: /^(youtube|video)/i.test(line) ? 'video' : 'article', date });
  }
  for (const line of day.book ?? []) {
    const m = line.match(/^(.*?)[,\s-]*(?:p\.?|page)\s*(\d+)/i);
    out.books.push({ title: (m?.[1] ?? line).trim(), page: m ? +m[2] : null, date });
  }
}
writeFileSync('private/reading/journal-additions.json', JSON.stringify(out, null, 1));
console.log(`reading from ${days.length} journal days: ${out.articles.length} articles, ${out.episodes.length} podcast episodes, ${out.books.length} book entries`);
