import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import { unified } from '@astrojs/markdown-remark';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkNoteLinks from './src/lib/remark-note-links.mjs';

export default defineConfig({
  site: 'https://fearthedeer9.github.io',
  trailingSlash: 'always',
  integrations: [mdx(), sitemap()],
  markdown: {
    // Astro 7's default processor (Sätteri) parses $math$ but has no KaTeX renderer.
    processor: unified({ remarkPlugins: [remarkMath, remarkNoteLinks], rehypePlugins: [rehypeKatex] }),
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
  },
  // The Jekyll blog paginated its archive; the new archive is one page.
  redirects: { '/blog/page2/': '/blog/', '/blog/page3/': '/blog/' },
});
