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
| `src/content/pages/` | Prose for About, Writing and the Notes index. |
| `src/content.config.ts` | Frontmatter schemas. A post with bad frontmatter fails the build. |
| `src/pages/` | Routes, including `/timeline/`, `/papers/`, `/archive/`, `/tags/` and `/feed.xml`. |
| `src/styles/global.css` | Design tokens (light and dark) and prose styles. |

Math is `$…$` or `$$…$$`, rendered to KaTeX at build time. Put display math on its own lines:

```
$$
x = y
$$
```

## Deploy

Pushing to `main` runs `.github/workflows/pages.yml`: build, link check, duplicate-title check, then deploy. Pull requests run the same checks without deploying.

`obsidian_to_blog.py` is the old Obsidian import script. It gets replaced when the site reads the vault directly (milestone 2 of the plan).
