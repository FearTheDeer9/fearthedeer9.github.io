// Fails if the built site mentions a private vault folder or a local path: the import refuses those
// folders, and this catches anything that slips through another way (a link, a wikilink alias, a pasted path).
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const MARKERS = [/TwoDelta\//, /Daily Learning\//, /Time Block Writing/, /Cover Letters\//, /SPIRITT\//, /mentor-ledger/, /iCloud~md~obsidian/, /\/Users\/harellidar/];
const hits = [];
for (const f of readdirSync('dist', { recursive: true }).filter((f) => /\.(html|xml|json|txt)$/.test(f))) {
  const text = readFileSync(join('dist', f), 'utf8');
  for (const m of MARKERS) if (m.test(text)) hits.push(`${f}: ${m}`);
}
console.log(hits.length ? hits.join('\n') : 'no private markers in dist/');
process.exit(hits.length ? 1 : 0);
