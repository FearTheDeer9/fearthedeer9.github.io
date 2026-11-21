# fearthedeer9.github.io

Harel Lidar's site: paper summaries, evergreen notes and essays. Built with Astro, deployed to GitHub Pages.

The rebuild plan, including the private workspace still to come, is in `docs/plans/2026-10-22-site-rebuild.md`. The backlog is in `TODO.md`.

## Run it

Needs Node 22.23.4 or newer.

```bash
npm install
npm run dev
```

The site is at http://localhost:4321. `npm run build` writes the static site to `dist/`, and `node scripts/check-links.mjs` checks its internal links.

## Where things live

| Path | What |
|---|---|
| `src/content/posts/` | Paper summaries and essays. The filename `YYYY-MM-DD-slug.md` sets the URL `/YYYY/MM/DD/slug/`, which matches the old Jekyll URLs. |
| `src/content/notes/` | Evergreen notes at `/notes/<filename>/`. Notes link to each other as `other-note.md`. |
| `src/content/entries/<type>/` | Notes imported from the Obsidian vault by `scripts/import-vault.mjs`, one folder per kind: papers, notes, lectures, concepts, excerpts, essays. Don't edit these by hand; edit the note in Obsidian and re-import. |
| `src/data/manifest.json` | Every published or logged item with its kind and date, written by the import. Drives the archive, home, timeline and feed. |
| `src/data/reading.json` | The public reading log (books, articles, podcasts), written by `scripts/build-reading.mjs` from private data. |
| `src/content/pages/` | Prose for About and the Notes index. |
| `src/content.config.ts` | Frontmatter schemas. A post with bad frontmatter fails the build. |
| `src/pages/` | Routes, including `/archive/`, `/reading/`, `/timeline/`, `/papers/`, `/tags/` and `/feed.xml`. |
| `src/styles/global.css` | Design tokens (light and dark) and prose styles. |

Math is `$…$` or `$$…$$`, rendered to KaTeX at build time. Put display math on its own lines:

```
$$
x = y
$$
```

## Publishing from the vault

What gets published was decided item by item in a review ledger. The decisions and the vault-derived working data live in `private/`, which is gitignored and exists only on Harel's machine.

```bash
node scripts/import-vault.mjs           # dry run: what would change
node scripts/import-vault.mjs --write   # copy approved notes and images into the site
node scripts/build-reading.mjs          # rebuild the public reading log
```

The import refuses private vault folders, and `scripts/check-private.mjs` fails the build if the output mentions one.

## Deploy

Pushing to `main` runs `.github/workflows/pages.yml`: build, link check, private-content check, duplicate-title check, then deploy. Pull requests run the same checks without deploying.

`obsidian_to_blog.py` is the old Obsidian import script. It gets replaced when the site reads the vault directly (milestone 2 of the plan).
