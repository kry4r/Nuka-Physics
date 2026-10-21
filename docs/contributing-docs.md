# Writing these docs

This Wiki is built with [MkDocs](https://www.mkdocs.org) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme. The pages are Markdown under `docs/`; the
navigation and the Markdown extensions are in `mkdocs.yml` at the repository root.

## Building

```bash
python -m venv .venv-docs
.venv-docs/bin/pip install "mkdocs==1.6.1" "mkdocs-material==9.7.7"
.venv-docs/bin/mkdocs serve            # http://127.0.0.1:8000, reloads on save
.venv-docs/bin/mkdocs build --strict   # the static site in site/; warnings fail it
```

`site/` is build output and stays out of git. The site needs no plugins, and fonts are not fetched from
Google, so it builds and serves offline.

## The actions reference

The pages under `reference/actions/` are generated from `assets/editor/actions.toml`, the same catalog the
editor's `action.help`, the MCP server and the agent skill read. Don't edit them by hand:

```bash
python tools/wiki_actions.py            # rewrite the pages
python tools/wiki_actions.py --check    # exit 1 when a page is out of date
```

A new action gets its entry in `actions.toml` (summary, kind, args, returns, example); the editor's
catalog smoke test fails until it has one.

## Style

- Describe what the editor does, checked against the code; name buttons and menus as the UI shows them,
  in bold, with menu paths as **File → Import Asset...**.
- Keys use the keys extension: `++ctrl+s++` renders as ++ctrl+s++.
- Give the action id beside a UI command when a script or agent would use it, and a short Python example
  with `app.actions.run(...)` where it helps.
- Use tables for controls and options, admonitions (`!!! tip`) sparingly.
- Images go in `docs/media/` as `.webp`, captured at 2560 × 1440 and scaled to 1600 px wide.
- Link to another page by its relative `.md` path and heading anchor, for example
  `../editor/robots.md#inverse-kinematics`; `mkdocs build --strict` reports a link that breaks.
