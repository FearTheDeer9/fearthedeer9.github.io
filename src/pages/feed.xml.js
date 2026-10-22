import rss from '@astrojs/rss';
import { posts, postUrl, postDate, displayTitle } from '../lib/site';

// /feed.xml is the URL jekyll-feed published, so existing subscribers keep working.
export async function GET(context) {
  return rss({
    title: 'Harel Lidar',
    description: 'Paper summaries, evergreen notes and essays on LLM inference and ML.',
    site: context.site,
    items: (await posts()).map((p) => ({ title: displayTitle(p.data.title), pubDate: postDate(p), link: postUrl(p) })),
  });
}
