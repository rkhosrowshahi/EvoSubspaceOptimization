# Publishing documentation on GitHub Pages

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) from this repository.

## Live site

After deployment is configured:

**https://rkhosrowshahi.github.io/cec2013lsgo/**

(Update `site_url` in `mkdocs.yml` if your GitHub username or repository name differs.)

## One-time GitHub setup

1. Push this repository to GitHub (repository root = this `cec2013lsgo` folder).
2. Open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

The workflow `.github/workflows/docs.yml` runs on every push to `main`.

## Build locally

`pip install -r requirements-docs.txt`

`mkdocs serve` — preview at http://127.0.0.1:8000

`mkdocs build` — output in `site/` (gitignored)

## Edit content

| Page | Source file in repo |
|------|------------------------|
| Home | `README.md` (included via `docs/index.md`) |
| Usage | `docs/usage.md` |
| License / Publishing | `docs/license.md`, `docs/publishing.md` |

Edit `README.md` (home page, included) or `docs/usage.md`; push to redeploy the site.

## Custom domain (optional)

Add a `CNAME` file under `docs/` and configure DNS per [GitHub custom domains](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site).
