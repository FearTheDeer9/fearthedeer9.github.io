import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;
export type Note = CollectionEntry<'notes'>;

// Reproduces Jekyll's /:year/:month/:day/:title/ from the filename, where :title kept case
// and turned runs of URL-unsafe characters into one dash ("alignment-?-a" -> "alignment-a").
export function postParts(post: Post) {
  const [, y, m, d, rest] = post.id.match(/^(\d{4})-(\d{2})-(\d{2})-(.+)$/)!;
  return { year: y, month: m, day: d, slug: rest.replace(/[^A-Za-z0-9+_.-]+/g, '-').replace(/-+/g, '-') };
}

export function postUrl(post: Post) {
  const { year, month, day, slug } = postParts(post);
  return `/${year}/${month}/${day}/${slug}/`;
}

export function postDate(post: Post) {
  const { year, month, day } = postParts(post);
  return post.data.date ?? new Date(`${year}-${month}-${day}`);
}

const ESSAYS = new Set(['on-writing-to-think', 'mse-as-gaussian']);
export const isEssay = (post: Post) => ESSAYS.has(postParts(post).slug);

export const displayTitle = (title: string) => title.replace(/^Paper Review - /, '');

export async function posts() {
  return (await getCollection('posts')).sort((a, b) => +postDate(b) - +postDate(a));
}

export async function notes() {
  return (await getCollection('notes')).sort((a, b) => a.data.title.localeCompare(b.data.title));
}

export const fmtMonth = (d: Date) => d.toLocaleDateString('en-GB', { month: 'short', year: 'numeric', timeZone: 'UTC' });
