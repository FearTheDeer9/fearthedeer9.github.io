#!/usr/bin/env python3
import os
import shutil
import argparse
import re
import yaml
from datetime import datetime
import anthropic  # or import openai

# Configuration
# Use environment variable or default to home directory
OBSIDIAN_DIR = os.path.expanduser('~/Documents/Obsidian_Local')
BLOG_DIR = os.path.expanduser('~/Documents/fearthedeer9.github.io/_posts/')

# Initialize API client
# client = anthropic.Anthropic(api_key="your-api-key-here")
# Or for OpenAI:
# client = openai.OpenAI(api_key="your-api-key-here")


def extract_title_from_content(content):
    """Extract title from the content or filename."""
    # Check for existing title in frontmatter
    frontmatter_match = re.search(r'---\s*\n(.*?)\n---', content, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        try:
            frontmatter_data = yaml.safe_load(frontmatter_text)
            if frontmatter_data and 'title' in frontmatter_data:
                return frontmatter_data['title']
        except Exception as e:
            print(f"Warning: Could not parse existing frontmatter: {e}")

    # Look for first heading
    heading_match = re.search(r'# (.*?)(\n|$)', content)
    if heading_match:
        return heading_match.group(1).strip()

    # Look for first line as fallback
    first_line = content.strip().split('\n')[0].strip()
    if first_line:
        return first_line

    return None


def extract_tags_from_content(content, client):
    """Extract or generate tags using LLM."""
    # Check for existing tags in frontmatter
    frontmatter_match = re.search(r'---\s*\n(.*?)\n---', content, re.DOTALL)
    if frontmatter_match:
        frontmatter_text = frontmatter_match.group(1)
        try:
            frontmatter_data = yaml.safe_load(frontmatter_text)
            if frontmatter_data and ('tags' in frontmatter_data or 'categories' in frontmatter_data):
                tags = frontmatter_data.get(
                    'tags', []) + frontmatter_data.get('categories', [])
                if isinstance(tags, list):
                    return tags
                elif isinstance(tags, str):
                    return [tag.strip() for tag in tags.split(',')]
        except Exception as e:
            print(f"Warning: Could not parse existing frontmatter tags: {e}")

    # Use LLM to generate tags
    prompt = f"""
    Based on the following content, suggest 3-5 relevant tags for categorizing this article on a blog:
    
    {content[:2000]}  # Limiting to first 2000 chars to keep prompt reasonable
    
    Return only the tags as a comma-separated list, like: tag1, tag2, tag3
    """

    # For Anthropic
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=100,
        temperature=0.0,
        system="You are a helpful assistant that suggests relevant tags for blog posts.",
        messages=[{"role": "user", "content": prompt}]
    )
    tags_text = response.content[0].text

    # For OpenAI
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant that suggests relevant tags for blog posts."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0,
    #     max_tokens=100
    # )
    # tags_text = response.choices[0].message.content

    # Clean the response to extract just the tags
    tags_text = tags_text.strip()
    tags = [tag.strip() for tag in tags_text.split(',')]
    return [tag for tag in tags if tag]  # Filter out any empty tags


def find_excerpt_position(content, client):
    """Determine where to insert the excerpt marker using LLM."""
    # Check if excerpt marker already exists
    if '<!-- excerpt-end -->' in content:
        return None

    # Use LLM to find a good position
    prompt = f"""
    I need to add an excerpt marker after the introduction of this blog post. 
    The introduction is typically the first few paragraphs that summarize what the article is about.
    
    Content:
    {content[:3000]}  # Limiting to first 3000 chars
    
    Find the line number (counting from 1) where the introduction ends and the main content begins.
    Return only the number, nothing else.
    """

    # For Anthropic
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=50,
        temperature=0.0,
        system="You are a helpful assistant that analyzes text structure.",
        messages=[{"role": "user", "content": prompt}]
    )
    line_number_text = response.content[0].text

    # For OpenAI
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant that analyzes text structure."},
    #         {"role": "user", "content": prompt}
    #     ],
    #     temperature=0,
    #     max_tokens=50
    # )
    # line_number_text = response.choices[0].message.content

    try:
        line_number = int(re.search(r'\d+', line_number_text).group(0))
        lines = content.split('\n')
        if 1 <= line_number <= len(lines):
            return line_number
        return min(5, len(lines))  # Default to line 5 if outside range
    except:
        return 5  # Default to line 5 if parsing fails


def process_file(filename, client):
    """Process a single file from Obsidian to GitHub blog."""
    # Try multiple possible locations for the file
    possible_paths = [
        os.path.join(OBSIDIAN_DIR, filename),
        os.path.join(OBSIDIAN_DIR, filename + '.md'),
        os.path.join(os.path.expanduser(
            '~/Library/Mobile Documents/iCloud~md~obsidian/Documents/PKM'), filename),
        os.path.join(os.path.expanduser(
            '~/Library/Mobile Documents/iCloud~md~obsidian/Documents/PKM'), filename + '.md')
    ]

    source_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            source_path = path
            break

    if not source_path:
        print(
            f"Error: Could not find file '{filename}' in any of the expected locations")
        print("Tried the following paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return False

    # Read file content
    with open(source_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Check if file already has frontmatter
    has_frontmatter = bool(re.match(r'^---\s*\n.*?\n---', content, re.DOTALL))

    if not has_frontmatter:
        # Extract or generate metadata
        title = extract_title_from_content(
            content) or os.path.splitext(filename)[0]
        creation_date = datetime.fromtimestamp(
            os.path.getctime(source_path)).strftime('%Y-%m-%d')
        tags = extract_tags_from_content(content, client)

        # Create frontmatter
        frontmatter = {
            'layout': 'single',
            'title': title,
            'date': creation_date,
            'tags': tags
        }

        # Add frontmatter to content
        frontmatter_text = yaml.dump(frontmatter, default_flow_style=False)
        content = f"---\n{frontmatter_text}---\n\n{content}"

    # Add excerpt marker if not present
    if '<!-- excerpt-end -->' not in content:
        excerpt_line = find_excerpt_position(content, client)
        if excerpt_line:
            lines = content.split('\n')
            lines.insert(excerpt_line, '\n<!-- excerpt-end -->\n')
            content = '\n'.join(lines)

    # Create destination path
    # For GitHub Pages, the filename should be in format: YYYY-MM-DD-title.md
    creation_date = datetime.fromtimestamp(
        os.path.getctime(source_path)).strftime('%Y-%m-%d')
    base_name = os.path.splitext(filename)[0]
    dest_filename = f"{creation_date}-{base_name}.md"
    dest_path = os.path.join(BLOG_DIR, dest_filename)

    # Write modified content to destination
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f"Successfully processed: {filename} -> {dest_filename}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert Obsidian notes to GitHub blog posts')
    parser.add_argument(
        '--files', nargs='+', help='List of file names to process (use quotes for files with spaces)')
    parser.add_argument(
        '--file-list', help='Path to a text file containing one filename per line')
    parser.add_argument('--api-key', help='API key for LLM service')
    args = parser.parse_args()

    # Initialize client with provided API key
    if args.api_key:
        client = anthropic.Anthropic(
            api_key="")
        # Or for OpenAI:
        # client = openai.OpenAI(api_key=args.api_key)
    else:
        print("Warning: No API key provided. Will use environment variable if available.")
        client = anthropic.Anthropic()
        # Or for OpenAI:
        # client = openai.OpenAI()

    # Get list of files to process
    files_to_process = []

    if args.files:
        files_to_process.extend(args.files)

    if args.file_list:
        try:
            with open(args.file_list, 'r', encoding='utf-8') as f:
                # Strip whitespace and filter out empty lines
                file_list = [line.strip()
                             for line in f.readlines() if line.strip()]
                files_to_process.extend(file_list)
        except Exception as e:
            print(f"Error reading file list: {e}")

    if not files_to_process:
        print("Error: No files specified. Use --files or --file-list option.")
        parser.print_help()
        return

    success_count = 0
    for filename in files_to_process:
        if process_file(filename, client):
            success_count += 1

    print(
        f"Processed {success_count} out of {len(files_to_process)} files successfully.")


if __name__ == "__main__":
    # Print usage examples
    print("Usage examples:")
    print("  With files that have spaces in their names:")
    print('  python obsidian_to_blog.py --files "My Note.md" "Another Note.md" --api-key="your-key"')
    print("")
    print("  Using a file list:")
    print('  python obsidian_to_blog.py --file-list my_files.txt --api-key="your-key"')
    print("  (where my_files.txt contains one filename per line)")
    print("")

    main()
