// Copies the notes approved in the publishing ledger from the Obsidian vault into the site.
// Re-run after editing a note in Obsidian; the vault stays the source of truth.
//
//   node scripts/import-vault.mjs            dry run: prints what would change
//   node scripts/import-vault.mjs --write    writes src/content/entries, public/attachments, src/data/manifest.json
//
// Inputs live in private/ledger/ (gitignored): ledger.json, decisions.json, types.json.
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync, copyFileSync, rmSync } from 'node:fs';
import { join, basename } from 'node:path';

const VAULT = process.env.VAULT_DIR ?? `${process.env.HOME}/Library/Mobile Documents/iCloud~md~obsidian/Documents/PKM`;
const WRITE = process.argv.includes('--write');
// Company-internal and private-journal folders. Nothing under these is ever imported, whatever the ledger says.
const NEVER = /^(TwoDelta|Daily Learning|Cover Letters|SPIRITT|Notion|mentor-ledger|Anki|_templates)\/|Time Block Writing\//;
const TYPE_DIR = { 'paper-summary': 'papers', evergreen: 'notes', lecture: 'lectures', concept: 'concepts', excerpt: 'excerpts', essay: 'essays', other: 'misc' };

const read = (p) => JSON.parse(readFileSync(p, 'utf8'));
const { items } = read('private/ledger/ledger.json');
const decisions = read('private/ledger/decisions.json');
const types = read('private/ledger/types.json');

const kept = items.filter((it) => ['approve', 'log'].includes(decisions[it.id]?.status));
// The ledger wrote some site URLs with a dash run where the real one has a single dash.
const urlOf = (it) => it.site_url?.replace(/-{2,}/g, "-") ?? (decisions[it.id].status === 'approve' ? `/${TYPE_DIR[types[it.id]?.type ?? 'other']}/${it.id.replace(/^note-/, '')}/` : null);

// Wikilinks resolve by note name (the vault filename), the way Obsidian does.
const byName = new Map(kept.filter((it) => it.vault_path).map((it) => [basename(it.vault_path, '.md').toLowerCase(), it]));
const files = new Map();
(function index(dir) {
  for (const e of readdirSync(join(VAULT, dir), { withFileTypes: true })) {
    if (e.name.startsWith('.')) continue;
    const rel = dir ? `${dir}/${e.name}` : e.name;
    if (e.isDirectory()) index(rel);
    else if (!files.has(e.name)) files.set(e.name, rel);
  }
})('');

const attachments = new Map(); // public filename -> vault-relative path (two notes may embed the same file)
const missing = [];

function transform(md, it) {
  md = md.replace(/^---\n[\s\S]*?\n---\n/, '');
  md = md.replace(/^\s*#\s+.*\n/, ''); // the layout prints the title
  // Obsidian accepts $$ mid-line ("a penalty $$ ... $$"); remark-math needs display math on its own lines.
  // Only blocks that share a line with text are rewritten, and at that line's indent, so lists stay intact.
  md = md.replace(/\$\$([\s\S]+?)\$\$/g, (whole, m, at, src) => {
    const lineStart = src.lastIndexOf('\n', at - 1) + 1, lineEnd = src.indexOf('\n', at + whole.length);
    const before = src.slice(lineStart, at), after = src.slice(at + whole.length, lineEnd < 0 ? undefined : lineEnd);
    if (!before.trim() && !after.trim()) return whole;
    const indent = before.match(/^\s*/)[0];
    return `${before.trim() ? '\n' + indent : ''}$$\n${indent}${m.trim().replace(/\n\s*/g, '\n' + indent)}\n${indent}$$${after.trim() ? '\n' + indent : ''}`;
  });
  md = md.replace(/!\[\[([^\]|]+)(?:\|[^\]]*)?\]\]/g, (_, target) => {
    const rel = files.get(target) ?? files.get(basename(target));
    if (rel && /\.(png|jpe?g|gif|webp|svg)$/i.test(rel)) {
      const name = `${it.id}-${basename(rel).replace(/[^A-Za-z0-9._-]+/g, '-')}`;
      attachments.set(name, rel);
      return `![](/attachments/${name})`;
    }
    const linked = byName.get(target.toLowerCase());
    if (linked && urlOf(linked)) return `[${linked.title}](${urlOf(linked)})`;
    missing.push(`${it.id}: embed ${target}`);
    return '';
  });
  // Plain markdown images with a vault path, relative to the note or to the vault root.
  md = md.replace(/!\[([^\]]*)\]\((?!https?:|\/attachments\/)([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, alt, raw) => {
    const target = decodeURIComponent(raw);
    const rel = [join(it.vault_path, '..', target), target].find((p) => existsSync(join(VAULT, p))) ?? files.get(basename(target));
    if (!rel) { missing.push(`${it.id}: image ${target}`); return ''; }
    const name = `${it.id}-${basename(rel).replace(/[^A-Za-z0-9._-]+/g, '-')}`;
    attachments.set(name, rel);
    return `![${alt}](/attachments/${name})`;
  });
  md = md.replace(/\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]/g, (_, target, alias) => {
    const linked = byName.get(target.trim().toLowerCase());
    const text = alias ?? target;
    return linked && urlOf(linked) ? `[${text}](${urlOf(linked)})` : text;
  });
  return md.trim() + '\n';
}

const yaml = (v) => JSON.stringify(v ?? null);
const manifest = [];
const entries = [];
for (const it of kept) {
  if (it.vault_path && NEVER.test(it.vault_path)) throw new Error(`refusing private vault path: ${it.vault_path}`);
  const d = decisions[it.id], t = types[it.id] ?? {};
  const status = d.status === 'approve' ? 'published' : 'logged';
  const row = {
    id: it.id, title: d.title || it.title, date: d.date || it.date, type: t.type ?? 'other', subtype: t.subtype ?? null,
    topic: it.topic ?? null, status, url: status === 'published' ? urlOf(it) : null,
    source_title: t.source_title ?? null, source_author: t.source_author ?? null, source_url: t.source_url ?? it.source_url ?? null,
  };
  manifest.push(row);
  // Items already on the site keep their existing file and URL; only new approvals are imported.
  if (status === 'published' && !it.site_url) {
    const body = transform(readFileSync(join(VAULT, it.vault_path), 'utf8'), it);
    const fm = ['title', 'date', 'type', 'subtype', 'topic', 'source_title', 'source_author', 'source_url'].map((k) => `${k}: ${yaml(row[k])}`).join('\n');
    entries.push([`${TYPE_DIR[row.type]}/${it.id}.md`, `---\n${fm}\n---\n\n${body}`]);
  }
}
manifest.sort((a, b) => (a.date ?? '').localeCompare(b.date ?? ''));

const counts = manifest.reduce((c, r) => ((c[`${r.status} ${r.type}`] = (c[`${r.status} ${r.type}`] ?? 0) + 1), c), {});
console.log(counts);
console.log(`${entries.length} entries to write, ${attachments.size} images to copy`);
if (missing.length) console.log(`unresolved embeds (dropped):\n  ${missing.join('\n  ')}`);

if (WRITE) {
  rmSync('src/content/entries', { recursive: true, force: true });
  for (const [path, text] of entries) {
    mkdirSync(join('src/content/entries', path, '..'), { recursive: true });
    writeFileSync(join('src/content/entries', path), text);
  }
  rmSync('public/attachments', { recursive: true, force: true });
  mkdirSync('public/attachments', { recursive: true });
  for (const [name, rel] of attachments) copyFileSync(join(VAULT, rel), join('public/attachments', name));
  mkdirSync('src/data', { recursive: true });
  writeFileSync('src/data/manifest.json', JSON.stringify(manifest, null, 1) + '\n');
  console.log('written');
}
