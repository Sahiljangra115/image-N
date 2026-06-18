# Context Map

This repo is multi-context. Each bounded context owns a `CONTEXT.md` glossary and (optionally) its own `docs/adr/`.

| Context | Path | CONTEXT.md | Notes |
| ------- | ---- | ---------- | ----- |
| _(none yet)_ | `src/<context>/` | `src/<context>/CONTEXT.md` | Add a row per context as it appears. |

System-wide architectural decisions live in `docs/adr/`. Context-scoped decisions live in `src/<context>/docs/adr/`.

`/grill-with-docs` populates the per-context `CONTEXT.md` files and this map lazily as terms and decisions get resolved. Leave the placeholder row until a real context exists.
