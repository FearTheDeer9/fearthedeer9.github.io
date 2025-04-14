# Personal Blog

This is my personal blog built with Jekyll and hosted on GitHub Pages.

## Local Development

### Prerequisites

- Ruby 3.1.6 (recommended to use rbenv for version management)
- Bundler
- Python 3.x (for the Obsidian to blog conversion script)

### Setup

1. Install Ruby version manager (rbenv):

   ```bash
   brew install rbenv ruby-build
   ```

2. Install Ruby 3.1.6:

   ```bash
   rbenv install 3.1.6
   rbenv local 3.1.6
   ```

3. Install project dependencies:
   ```bash
   gem install bundler
   bundle install
   ```

### Running the Site Locally

1. Start the Jekyll server:

   ```bash
   bundle exec jekyll serve
   ```

2. View the site at: http://localhost:4000

The server will automatically rebuild when you make changes to your files.

## Converting Obsidian Notes to Blog Posts

1. Copy your Obsidian notes to a local directory:

   ```bash
   mkdir -p ~/Documents/Obsidian_Local
   # Copy your Obsidian notes to this directory
   ```

2. Update the `OBSIDIAN_DIR` path in `obsidian_to_blog.py` to point to your local Obsidian directory.

3. Run the conversion script:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install pyyaml anthropic
   python3 obsidian_to_blog.py --file-list files.txt --api-key="your-api-key"
   ```

## Deployment

1. Commit your changes:

   ```bash
   git add .
   git commit -m "Your commit message"
   ```

2. Push to GitHub:
   ```bash
   git push origin main
   ```

GitHub Pages will automatically build and deploy your site. The site will be available at: https://fearthedeer9.github.io

## Troubleshooting

- If you encounter Ruby version issues, make sure you're using Ruby 3.1.6:

  ```bash
  ruby -v  # Should show 3.1.6
  ```

- If you get permission errors, try:

  ```bash
  bundle config set --local path 'vendor/bundle'
  bundle install
  ```

- If the site doesn't update after pushing to GitHub, check the GitHub Actions tab in your repository for any build errors.
