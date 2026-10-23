# Fork builds: one-click updates per branch

A fork of Domestique builds its own Windows installer for every branch you push,
and the installed app updates itself from that branch in one click. Versions are
not compared — branch, merge and cherry-pick as you like.

## How it works

- **CI** — `.github/workflows/fork-channel.yml` runs on every push to a branch of
  the fork (never in `platypus45/domestique`). It builds the app, compiles the
  installer (`packaging/domestique.iss`), installs it silently on the runner and
  checks the installed app starts at this commit, then publishes it as the
  branch's **channel**: a rolling prerelease tagged `channel-<branch>` holding
  `Domestique-Setup.exe` and `build.json` (repo, branch, commit, build time,
  SHA-256). A build that fails the check is not published.
- **The app** — each build carries its branch and commit (`BUILD_INFO.json`,
  shown next to the version: `v3.13.1 · fork/dev@f23a097`). It checks its own
  branch's channel at start-up and every 3 hours. A build is offered when it is a
  different commit built later on the same branch; **Update now** downloads it,
  checks the SHA-256, runs the installer silently, and Domestique restarts.
- **Switching branch** — Settings → **Updates**: pick a branch, **Check**,
  **Install latest build**. The app then follows that branch.
- **Upstream** — the app still watches upstream's releases, but only tells you
  ("Upstream released v3.13.2 — this build is based on v3.13.1"). It never
  installs upstream over your fork. `VERSION` stays upstream's number: do not
  edit it in the fork, and it keeps saying which upstream release you are on.

Your data stays in `%USERPROFILE%\.domestique` and is never touched by an
install, an update or the uninstaller.

## Setting up the fork (once)

1. **Enable Actions** in the fork: GitHub turns workflows off in new forks —
   Actions tab → enable.
2. **Push a branch** (or run "Fork channel build" by hand) and wait for the run
   to finish; the build is published at its end.
3. **Install once by hand**: the fork's Releases page → `channel-<branch>` →
   `Domestique-Setup.exe`. Windows SmartScreen warns once (the installer is not
   signed): More info → Run anyway. It installs for your user only, under
   `%LOCALAPPDATA%\Programs\Domestique`, with no administrator prompt. A folder
   you unzipped from an older release can be deleted.
4. **intervals.icu**: Settings → Intervals.icu → *Advanced: paste an API key
   instead* — your personal key from intervals.icu. A fork build carries no
   OAuth client of its own.

From then on, push and click **Update now**.

## Notes

- **Upstream changes** reach your fork only when you merge them; until then
  every update installs your branch as it is.
- **Deleted branches** keep their channel release until you delete it (the
  release and its `channel-*` tag).
- **Public fork** — Actions are free. A private repository spends its monthly
  minutes, and Windows runners use them faster.
- **Not signed** — updates downloaded by the app itself do not pass through
  SmartScreen, but antivirus software may still look at an unsigned installer.
