import { visit } from 'unist-util-visit';

// Notes link to each other as `other-note.md` so they also work in Obsidian and on GitHub;
// on the site those become /notes/other-note/.
export default function remarkNoteLinks() {
  return (tree) => {
    visit(tree, 'link', (node) => {
      const m = node.url.match(/^([\w-]+)\.md(#.*)?$/);
      if (m) node.url = `/notes/${m[1]}/${m[2] ?? ''}`;
    });
  };
}
