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

const pages = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/pages', generateId: keepName }),
  schema: z.object({ title: z.string() }),
});

export const collections = { posts, notes, pages };
