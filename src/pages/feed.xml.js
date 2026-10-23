import rss from '@astrojs/rss';
import { published, TYPES } from '../lib/manifest';

// /feed.xml is the URL jekyll-feed published, so existing subscribers keep working.
export async function GET(context) {
  return rss({
    title: 'Harel Lidar',
    description: 'Paper summaries, evergreen notes, lecture notes and essays on LLM inference and ML.',
    site: context.site,
    items: published.filter((i) => i.date).sort((a, b) => b.date.localeCompare(a.date))
      .map((i) => ({ title: i.title, pubDate: new Date(i.date), link: i.url, categories: [TYPES[i.type].label] })),
  });
}
