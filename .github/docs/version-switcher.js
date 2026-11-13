// Copyright (C) 2026 Jørgen Schartum Dokken
//
// This file is part of io4dolfinx
//
// SPDX-License-Identifier:    MIT
//
// Read the Docs-style version switcher (bottom right). .github/scripts/docs_site.py adds it to
// every page of every version when the web page is put together, so old builds get it without
// being rebuilt. It reads the versions from versions.json at the root of the web page, which is
// one folder up from here.
(() => {
  // Not inside frames, such as the 3D viewer of a pyvista plot
  if (window.self !== window.top) return;
  const root = new URL("..", document.currentScript.src);
  if (!location.pathname.startsWith(root.pathname)) return;
  const [current, ...rest] = location.pathname.slice(root.pathname.length).split("/");
  // The page within its version, "" for the front page
  const page = rest.join("/");

  const home = (version) => new URL(version + "/", root).href;
  const samePage = (version) => new URL(version + "/" + page, root).href;

  // The same page may not exist in the other version: then go to its front page
  async function go(version) {
    let exists = page === "";
    if (!exists) {
      exists = await fetch(samePage(version), { method: "HEAD" }).then(
        (r) => r.ok,
        () => false
      );
    }
    location.href = exists ? samePage(version) + location.search + location.hash : home(version);
  }

  const STYLE = `
    #io4d-versions { position: fixed; right: 1rem; bottom: 1rem; z-index: 1000;
      font-size: 0.85rem; font-family: var(--pst-font-family-base, sans-serif); }
    #io4d-versions > button { cursor: pointer; border-radius: 0.35rem; padding: 0.35rem 0.75rem;
      border: 1px solid var(--pst-color-border, #ccc);
      background: var(--pst-color-surface, #f5f5f5); color: var(--pst-color-text-base, #222);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
    #io4d-versions > button::before { content: "v: "; color: var(--pst-color-primary, #0a7d91); }
    #io4d-versions ul { display: none; position: absolute; right: 0; bottom: calc(100% + 0.4rem);
      margin: 0; padding: 0.3rem 0; list-style: none; min-width: 10rem; border-radius: 0.35rem;
      border: 1px solid var(--pst-color-border, #ccc);
      background: var(--pst-color-surface, #f5f5f5); box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2); }
    #io4d-versions.open ul { display: block; }
    #io4d-versions li a { display: block; padding: 0.25rem 0.9rem; text-decoration: none;
      color: var(--pst-color-text-base, #222); }
    #io4d-versions li a:hover, #io4d-versions li a:focus-visible {
      background: var(--pst-color-background, #fff); }
    #io4d-versions li a[aria-current] { font-weight: bold; color: var(--pst-color-primary, #0a7d91); }
  `;

  fetch(new URL("versions.json", root))
    .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
    .then((versions) => {
      const entry = versions.find((v) => v.version === current);
      if (!entry) return;
      const style = document.createElement("style");
      style.textContent = STYLE;
      document.head.append(style);

      const container = document.createElement("div");
      container.id = "io4d-versions";
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = entry.name;
      button.setAttribute("aria-label", "Choose a version of the documentation");
      button.setAttribute("aria-expanded", "false");
      const list = document.createElement("ul");
      for (const v of versions) {
        const a = document.createElement("a");
        a.textContent = v.name;
        // Points at the same page, so middle-click and copying the link work
        a.href = samePage(v.version);
        if (v.version === current) a.setAttribute("aria-current", "page");
        a.addEventListener("click", (e) => {
          if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
          e.preventDefault();
          if (v.version === current) setOpen(false);
          else go(v.version);
        });
        const li = document.createElement("li");
        li.append(a);
        list.append(li);
      }
      const setOpen = (open) => {
        container.classList.toggle("open", open);
        button.setAttribute("aria-expanded", String(open));
      };
      button.addEventListener("click", () => setOpen(!container.classList.contains("open")));
      document.addEventListener("click", (e) => {
        if (!container.contains(e.target)) setOpen(false);
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") setOpen(false);
      });
      container.append(list, button);
      document.body.append(container);
    })
    .catch(() => {});
})();
