# Site rebuild: public blog + private workspace

## Goal

One Astro codebase, two builds:

- **Public** (`astro build`, deployed to GitHub Pages): blog, paper summaries, evergreen notes, about. Something to be proud of showing.
- **Private** (`astro dev` locally, never deployed): the same site plus work notes, journal, tasks and logs, rendered from the Obsidian vault with layouts I control.

Obsidian stays the editor and the source of truth. The site is a rendering and UI layer over the vault.

## Decisions (2026-10-22)

| Question | Choice |
|---|---|
| Where the private half lives | Local-only. Private routes exist only in dev builds. |
| Source of truth | Obsidian vault. Notes reach the public site only with `publish: true` in frontmatter. |
| Stack | Astro (content collections, MDX, islands). Replaces Jekyll and minimal-mistakes. |
| Milestone 1 | Public site redesign |
| Design | "Notebook": Newsreader serif + IBM Plex Mono labels, off-white, rust accent `#8A3B12`. The topic timeline is its own page at `/timeline`. Canvas: https://claude.ai/artifact/Hpeyh2QCC6Xpi2zyaC1h1o |

## Hard constraint: nothing private leaks

TwoDelta notes are company-internal. The public build must not depend on remembering a flag:

- Vault content enters through one loader. In a production build it drops every note without `publish: true`, and drops everything under `TwoDelta/`, `Daily Learning/`, and `Time Block Writing/` even when flagged.
- A post-build check greps `dist/` for private paths and markers, and for any note title from the excluded folders. It fails CI on a hit.
- Private routes (`/work`, `/journal`, `/tasks`) are in a directory that is excluded from production builds, not merely unlinked.

## Milestones

### M1: Public redesign

1. ~~Pick a design direction.~~ Done 2026-10-22: Notebook + timeline page.
2. ~~Scaffold Astro, port the design.~~ Done on branch `astro-rebuild` (Astro 7).
3. ~~Migrate posts and notes.~~ Done. Every URL in the live sitemap resolves.
4. ~~Math at build time.~~ Done with KaTeX through `@astrojs/markdown-remark`: Astro 7's default processor (Sätteri) parses math but cannot render it.
5. ~~Replace the Jekyll workflow.~~ Done: one workflow builds, link-checks, checks duplicate titles, deploys from `main`. The private-content leak check lands with M3, when there is private content to leak.
6. Update the about and home copy. The current tagline (representation, world models, safety) predates the inference work. Content is yours to write; I'll leave placeholders.

### M2: Vault → site pipeline

- Loader reads the vault path from env (`VAULT_DIR`). Handles `[[wikilinks]]` (resolve or render as dead), `![[image.png]]` embeds (copy the attachment), `$`/`$$` math, and the paper-summary template.
- Backlinks and hover previews (the Matuschak pattern).
- `publish: true` is the only publishing gesture. Retire `obsidian_to_blog.py` and `fix_titles.sh`.

### M3: Private workspace

- `/journal`: Time Block Writing and Daily Learning, reverse-chronological, with a calendar strip.
- `/tasks`: tasks parsed from notes (the vault has no `- [ ]` checkboxes yet, so this decides a convention first).
- `/work`: TwoDelta folders with backlinks and graph.
- `/inbox`: clips with their `why:` lines and read status.

### M4+: Backlog

See `TODO.md`: interests-through-time timeline (embeddings + clustering), knowledge graph across paper clusters, activity logging.

## Open questions

- Domain: stay on `fearthedeer9.github.io` or buy one?
- Should `AI Theory/Imperial/` coursework ever be public?
