import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// Keep the filename as the id: the default slugifier lowercases it, and the Jekyll URLs
// being preserved are case-sensitive (/2025/04/14/3D-Gaussian-Splatting-.../).
const keepName = ({ entry }: { entry: string }) => entry.replace(/\.mdx?$/, '');

// Jekyll accepts a space-separated string wherever it accepts a list.
const list = z.union([z.string().transform((s) => s.split(/\s+/)), z.array(z.string())]).default([]);

const posts = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/posts', generateId: keepName }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date().optional(),
    categories: list,
    tags: list,
  }),
});

const notes = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/notes', generateId: keepName }),
  schema: z.object({ title: z.string(), created: z.coerce.date().optional() }),
});

// Notes imported from the vault by scripts/import-vault.mjs, one folder per type (entries/papers/<id>.md).
const entries = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/entries', generateId: keepName }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date().nullable(),
    type: z.enum(['paper-summary', 'evergreen', 'lecture', 'concept', 'excerpt', 'essay', 'other']),
    subtype: z.string().nullable(),
    topic: z.string().nullable(),
    source_title: z.string().nullable(),
    source_author: z.string().nullable(),
    source_url: z.string().nullable(),
  }),
});

const pages = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/pages', generateId: keepName }),
  schema: z.object({ title: z.string() }),
});

export const collections = { posts, notes, entries, pages };
