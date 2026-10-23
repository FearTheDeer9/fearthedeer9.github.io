import manifest from '../data/manifest.json';

// Every published or logged item, as decided in the publishing ledger (see scripts/import-vault.mjs).
export type Item = {
  id: string; title: string; date: string | null; type: TypeKey; subtype: string | null; topic: string | null;
  status: 'published' | 'logged'; url: string | null;
  source_title: string | null; source_author: string | null; source_url: string | null;
};
export type TypeKey = 'paper-summary' | 'evergreen' | 'lecture' | 'concept' | 'excerpt' | 'essay' | 'other';

// Order here is the order of sections on the archive page.
export const TYPES: Record<TypeKey, { label: string; plural: string; blurb: string; color: string }> = {
  essay: { label: 'Essay', plural: 'Essays', blurb: 'Longer pieces in my own words.', color: 'var(--t-essay)' },
  evergreen: { label: 'Evergreen note', plural: 'Evergreen notes', blurb: 'Atomic notes: each argues one claim, and the title is the claim.', color: 'var(--t-evergreen)' },
  'paper-summary': { label: 'Paper summary', plural: 'Paper summaries', blurb: 'One paper each: the problem, the method, what the results do and don’t show.', color: 'var(--t-paper)' },
  lecture: { label: 'Lecture notes', plural: 'Lecture notes', blurb: 'Notes taken in courses, grouped by course.', color: 'var(--t-lecture)' },
  concept: { label: 'Concept', plural: 'Concepts', blurb: 'A glossary: one idea per entry, explained briefly.', color: 'var(--t-concept)' },
  excerpt: { label: 'Excerpt', plural: 'Excerpts', blurb: 'Mostly other people’s words: article summaries, quoted passages, course summaries. Kept as reading, not as my writing.', color: 'var(--t-excerpt)' },
  other: { label: 'Note', plural: 'Other notes', blurb: 'Notes that fit no other section.', color: 'var(--muted)' },
};

export const SUBTYPES: Record<string, string> = {
  'article-summary': 'Article summary', quoted: 'Quoted passages', 'course-summary': 'Course summary',
};

export const items = (manifest as Item[]).filter((i) => i.title);
export const published = items.filter((i) => i.status === 'published');
export const byUrl = new Map(published.map((i) => [i.url, i]));
export const typeOfUrl = (url: string) => byUrl.get(url)?.type;

// "Intro to ML Lecture 3 - Evaluation" -> { name: "Intro to ML", n: 3 }. Titles without a lecture number group under their own name.
export function courseOf(title: string) {
  const m = title.match(/^(.*?)\s*[-–:]?\s*(?:Lecture|Tutorial)\s*(\d+)/i);
  return m ? { name: m[1].trim(), n: +m[2] } : null;
}
