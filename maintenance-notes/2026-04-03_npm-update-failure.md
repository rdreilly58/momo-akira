**Subject: npm global update failure**

The daily `auto-update-system.sh` script is consistently failing on the `npm update -g` step with an `EINVALIDPACKAGENAME` error for a package named `.gdrive-server-credentials.json`.

**What I've tried:**
- `npm uninstall -g @modelcontextprotocol/server-gdrive` (which seemed to be the source)
- `rm -rf` on the suspected directory in the global `node_modules` (it wasn't there)

The error persists, which suggests it might be a corrupted `package.json` file inside one of the other global modules, or a broken symlink. This requires manual inspection of the `npm root -g` directory.

The Homebrew updates completed (after I force-linked `pillow`), so this is a low-priority issue.
