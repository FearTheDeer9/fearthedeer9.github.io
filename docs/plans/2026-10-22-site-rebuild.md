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

Decided 2026-10-23, replacing the `publish: true` idea: the **publishing ledger** is the publishing gesture.
Harel reviewed 175 candidates in the ledger artifact: 114 approve, 43 log only, 17 reject, 1 later.

- `src/data/manifest.json` (committed) lists every published or logged item: id, title, date, type, topic, source, and the vault path it comes from. Logged-only items carry no body.
- `scripts/import-vault.mjs` re-copies approved notes from the vault into `src/content/` and their pasted images into `public/`. Re-running it picks up edits made in Obsidian, so the vault stays the source of truth. Dry-run by default.
- **Content types** (Harel's taxonomy), shown in separate sections of one page, each with its own presentation:
  - **Paper summaries.**
  - **Evergreen notes.**
  - **Lecture/course notes.**
  - **Concepts:** a glossary.
  - **Excerpts:** article summaries, mostly-quoted notes and course summaries. These also count as read items.
  - **Essays.**
- Every item is labelled with its type wherever it appears: home, timeline, archive.
- **Reading log:** books, articles and podcasts from the Notion journal (2,121 entries, 2023 onward) plus the 2020–23 export. Shown on `/reading/`, with excerpts and logged-only notes included as read items.
- Still to do: backlinks and hover previews (the Matuschak pattern). Retire `obsidian_to_blog.py`.

### M3: Private workspace

- `/journal`: Time Block Writing and Daily Learning, reverse-chronological, with a calendar strip.
- `/tasks`: tasks parsed from notes (the vault has no `- [ ]` checkboxes yet, so this decides a convention first).
- `/work`: TwoDelta folders with backlinks and graph.
- `/inbox`: clips with their `why:` lines and read status.

### M3b: Daily score (decided 2026-10-23)

A daily productivity score from the Notion daily log, graphed publicly at `/progress/`.

- **Capture.** A habit checklist is added to the Notion daily template (`docs/journal-template.md`): Anki, meditation, contemplative walk, good writing session, lesson, fantasy football, plus minutes of social media. The existing sections (goals, workout, articles, book, TV) are scored too, and free text is a fallback.
- **Sync.** A personal Notion integration token, which Harel creates and stores in the macOS Keychain. `scripts/journal/sync-notion.mjs` runs locally each evening and writes one markdown file per day into `private/journal/days/`. The vault supplies "notes taken" (notes created that day in `AI Theory/`).
- **Score.** `scripts/journal/score.mjs` applies the rules at the top of that file:
  - Capped habit points, minus drain points, plus a bonus for the share of daily goals hit, plus a streak bonus.
  - Grades: S ≥ 80, A ≥ 60, B ≥ 40, C ≥ 20, D below.
  - Sports: NFL, Green Bay, Milwaukee and key live events are neutral; highlights, recaps and sports podcasts are slightly negative.
- **Privacy.** Raw days and the per-habit breakdown stay in `private/journal/`. The public `src/data/scores.json` holds only date, score and grade.
- **Graph.** It starts on the lock-in day: the first day with the habit checklist. Earlier days (the 2023 export) show faded, as a baseline.

### M4+: Backlog

See `TODO.md`: interests-through-time timeline (embeddings + clustering), knowledge graph across paper clusters, activity logging.

## Open questions

- Domain: stay on `fearthedeer9.github.io` or buy one?
- ~~Should `AI Theory/Imperial/` coursework ever be public?~~ Yes: Harel approved lecture notes in the ledger.
