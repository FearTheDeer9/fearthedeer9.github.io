// Fails if any internal href/src in the built site points at a page or file that doesn't exist.
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const root = 'dist';
const html = readdirSync(root, { recursive: true }).filter((f) => f.endsWith('.html'));
const broken = [];
for (const file of html) {
  for (const [, url] of readFileSync(join(root, file), 'utf8').matchAll(/(?:href|src)="(\/[^/"][^"]*|\/)"/g)) {
    const path = decodeURIComponent(url.split(/[?#]/)[0]);
    if (!existsSync(join(root, path)) && !existsSync(join(root, path, 'index.html'))) broken.push(`${file}: ${url}`);
  }
}
console.log(broken.length ? broken.join('\n') : `${html.length} pages, no broken internal links`);
process.exit(broken.length ? 1 : 0);
