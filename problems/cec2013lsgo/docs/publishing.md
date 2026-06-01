# Publishing documentation on GitHub Pages

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) from this repository.

## Live site

After deployment is configured:

**https://rkhosrowshahi.github.io/cec2013lsgo/**

(Update `site_url` in `mkdocs.yml` if your GitHub username or repository name differs.)

## One-time GitHub setup

1. Push this repository to GitHub (repository root = this `cec2013lsgo` folder).
2. Open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions** (not “Deploy from a branch”).
4. Save, then run **Actions → Deploy documentation → Run workflow** (or push to `main`).

The workflow `.github/workflows/docs.yml` runs on every push to `main`.

### If deploy fails but “Build site” succeeds

The **deploy** job needs GitHub Pages enabled with **GitHub Actions** as the source. If Pages was never turned on, or is still set to deploy from `main` / `/docs`, the **Deploy to GitHub Pages** step fails and you may also see a separate **pages build and deployment** workflow fail.

Fix: **Settings → Pages → Source → GitHub Actions**, then re-run **Deploy documentation**.

## Build locally

`pip install -r requirements-docs.txt`

`mkdocs serve` — preview at http://127.0.0.1:8000

`mkdocs build` — output in `site/` (gitignored)

## Edit content

| Page | Source file in repo |
|------|------------------------|
| Home | `README.md` (included via `docs/index.md`) |
| Usage | `docs/usage.md` |
| Changelog | `CHANGELOG.md` (included via `docs/changelog.md`) |
| License / Publishing | `docs/license.md`, `docs/publishing.md` |

Edit `README.md` (home page, included) or `docs/usage.md`; push to redeploy the site.

## Custom domain (optional)

Add a `CNAME` file under `docs/` and configure DNS per [GitHub custom domains](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site).
