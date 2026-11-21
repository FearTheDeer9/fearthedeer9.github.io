// Builds src/data/reading.json, the public reading log, from the private consolidated log.
//
//   node scripts/build-reading.mjs [shelf-notes-dir]
//
// Input: private/reading/reading-all.json (books, articles, podcasts; gitignored).
// Optional: a directory of Reading Shelf annotations (one <book-id>.json per book: rating, take, idea, for,
// favorite, hidden, year_read, status), as saved from the shelf page.
// Only public fields are copied. Notion page links and Harel's raw journal notes never leave private/.
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const all = JSON.parse(readFileSync('private/reading/reading-all.json', 'utf8'));
mergeJournal(all, existsSync('private/reading/journal-additions.json') ? JSON.parse(readFileSync('private/reading/journal-additions.json', 'utf8')) : null);
const dir = process.argv[2];
const notes = dir && existsSync(dir)
  ? Object.fromEntries(readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => [f.slice(0, -5), JSON.parse(readFileSync(join(dir, f), 'utf8'))]))
  : {};

// Work documents read at an employer are not reading to publish.
const WORK = /SPIRITT|two ?delta|deltasight/i;
const year = (d) => (d ? +String(d).slice(0, 4) : null);
const books = (all.books ?? []).filter((b) => !WORK.test(b.title ?? '')).map((b) => {
  const n = notes[b.id] ?? {};
  const status = n.status ?? b.status ?? null;
  // Unfinished-by-choice books stay off the public page unless Harel un-hides them on the shelf.
  const hidden = n.hidden ?? ['To Read', 'Quit'].includes(status);
  if (hidden) return null;
  return {
    id: b.id, title: b.canonical_title ?? b.title, authors: b.authors ?? (b.logged_author ? [b.logged_author] : []),
    year_published: b.year ?? null, year_read: n.year_read ?? year(b.date_finished ?? b.date_started ?? b.log?.last ?? b.kindle_export_date),
    status, subjects: b.subjects ?? [], blurb: b.blurb ?? null, url: b.url ?? null,
    rating: n.rating ?? null, take: n.take || null, idea: n.idea || null, for: n.for || null, favorite: !!n.favorite,
  };
}).filter(Boolean);

const articles = (all.articles ?? []).filter((a) => a.title && !WORK.test(a.title)).map((a) => ({
  title: a.title, medium: a.medium ?? 'article', author: a.author ?? null, date: a.date ?? null, topic: a.topic ?? null, url: a.url ?? null,
}));
const podcasts = (all.podcasts ?? []).filter((p) => p.show).map((p) => ({
  show: p.show, topic: p.topic ?? null, first: p.first ?? null, last: p.last ?? null,
  episodes: (p.episodes ?? []).filter((e) => e.title && !WORK.test(e.title)).map((e) => ({ title: e.title, date: e.date ?? null })),
}));

// New reading from journal days (scripts/journal/reading-from-journal.mjs), folded into the extracted log.
// Books match by title prefix; a book seen only in the journal is added as "Started" without catalogue details.
function mergeJournal(all, add) {
  if (!add) return;
  const key = (t) => String(t ?? '').toLowerCase().replace(/[^a-z0-9א-ת]+/g, ' ').trim();
  const seen = new Set(all.articles.map((a) => `${key(a.title)}|${a.date}`));
  for (const a of add.articles) if (!seen.has(`${key(a.title)}|${a.date}`)) { all.articles.push({ ...a, author: null, topic: null, url: null }); seen.add(`${key(a.title)}|${a.date}`); }
  for (const e of add.episodes) {
    let show = all.podcasts.find((p) => key(p.show) === key(e.show));
    if (!show) all.podcasts.push(show = { show: e.show, topic: null, first: e.date, last: e.date, episodes: [] });
    if (!show.episodes.some((x) => key(x.title) === key(e.title) && x.date === e.date)) show.episodes.push({ title: e.title, date: e.date });
    show.last = show.last && show.last > e.date ? show.last : e.date;
  }
  for (const b of add.books) {
    let book = all.books.find((x) => [x.title, x.canonical_title].some((t) => t && (key(t).startsWith(key(b.title)) || key(b.title).startsWith(key(t)))));
    if (!book) all.books.push(book = { id: key(b.title).replace(/ /g, '-'), title: b.title, status: 'Started', date_started: b.date, log: { first: b.date, last: b.date } });
    book.log = { ...(book.log ?? {}), last: book.log?.last && book.log.last > b.date ? book.log.last : b.date };
  }
}

mkdirSync('src/data', { recursive: true });
writeFileSync('src/data/reading.json', JSON.stringify({ books, articles, podcasts }, null, 1) + '\n');
console.log(`${books.length} books (${(all.books ?? []).length - books.length} hidden), ${articles.length} articles, ${podcasts.length} shows / ${podcasts.reduce((n, p) => n + p.episodes.length, 0)} episodes`);
