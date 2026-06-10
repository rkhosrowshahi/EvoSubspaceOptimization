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

### Red “pages build and deployment” (Jekyll) while the site works

GitHub may still run an internal workflow named **pages build and deployment** on every push. Its **build** job runs **Jekyll** on the repository root (Python package + MkDocs sources), which is not a Jekyll site, so **Build with Jekyll** fails after ~20s.

That run is **not** your docs deploy. The real publish path is **Actions → Deploy documentation** (`.github/workflows/docs.yml`), which builds with MkDocs and uploads the `site/` artifact.

| Workflow | What it does | What to check |
|----------|----------------|---------------|
| **Deploy documentation** | `mkdocs build` → GitHub Pages | Must be **green** |
| **pages build and deployment** | Legacy Jekyll on repo files | Often **red** here; safe to ignore if the row above succeeded |

If **Deploy documentation** succeeded, the live site is already up (e.g. **https://rkhosrowshahi.github.io/cec2013lsgo/**). The Jekyll workflow’s **deploy** job is skipped when build fails, so it does not replace the MkDocs site.

To reduce noise: **Settings → Pages** → confirm **Source** is **GitHub Actions** only (no branch/folder selected). Re-save if you recently switched from classic. You cannot disable the internal workflow from this repo.

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
