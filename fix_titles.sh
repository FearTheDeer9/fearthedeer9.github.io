#!/bin/bash

# Script to fix YAML frontmatter in post files by adding quotes around titles with colons

# For files with colons in the title
sed -i '' 's/^title: \(.*:.*\)$/title: "\1"/' _posts/*.md
echo "Fixed quotes around titles with colons"

# For files with spaces in the title (to be safe)
sed -i '' 's/^title: \(Paper Review - .*\)$/title: "\1"/' _posts/*.md
echo "Fixed quotes around Paper Review titles"

echo "All done! Run 'bundle exec jekyll serve' to test" 