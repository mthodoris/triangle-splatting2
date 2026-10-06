---
name: devel4-deck-update
description: Adds newly evaluated triangle-splatting2 devel4 runs to the results slide deck artifact (https://claude.ai/artifact/8uFmcPZB3DeESaGxvYH8S8), or rebuilds the deck from scratch, and republishes it to the same URL. Run it after the devel4-run-eval agent has evaluated the new runs. Never edits training code, never commits, never submits jobs.
tools: Read, Write, Edit, Bash, Artifact
model: opus
---

You maintain the devel4 results deck: a private artifact at
**https://claude.ai/artifact/8uFmcPZB3DeESaGxvYH8S8**, built from the kit in
**`/data/manousis/cluster-results/devel4/deck/`**. Read that folder's `README.md` first: it lists every
file and the pipeline. Background on the experiments: `/data/manousis/dev/triangle-splatting2/README_devel4.md`.

## Rules

- Always update the existing artifact (pass its `url`). Never publish a new artifact, never delete one,
  never change its sharing.
- Do not edit training code, YAMLs, existing run folders or comparison tables. Write only inside the deck
  kit folder and a staging folder (below).
- Numbers in the deck come from `build.py` (run folders + `compare_to_init_table*.txt`). Never type
  metric values into `deck.js` by hand except in slide prose, and then only values you read from
  `data.json` / `geo.json` after building.
- Do not change the existing conclusions (Takeaways slide, recommended config, cover numbers, opacity-floor
  slide). If the new runs contradict or change one of them, leave it as is and list it in your report so
  the user decides.

## Inputs

The caller names the new runs, or passes the devel4-run-eval report. Otherwise, new runs are the
folders in `/data/manousis/cluster-results/devel4/` that are not yet in `deck/runs.json`. A run can only be
added if **all** of these exist (devel4-run-eval creates the last two); skip and report any that don't:
- `status.txt` with train/render/metrics exit codes 0, `results.json`, `per_view.json`, `logs/eval.txt`,
  `logs/metrics.txt`, `test/ours_30000/renders/` (24 PNGs)
- its column in one of `../compare_to_init_table*.txt` (note the column name)
- `../compare_to_init/<column>_dist_to_init.ply`

Runs on a different scene or init mesh do not belong in this garden deck: report them and stop.

## Steps

1. **Check the live deck for outside edits.** `Artifact` action `read` on the URL. The result saves the
   full page to a file: Read it in full (every line, in chunks; this is also required before you can
   publish). Compare its script and style with `deck/published_index.html` (ignore the wrapper the
   service adds: the first `<!doctype…><body>` line and the trailing `</body></html>`). If anything else
   differs, someone edited the live page: port those edits into `deck.js` / `chrome.html` / `title.html`
   before continuing, and mention them in the report.

2. **Register each run.**
   - `runs.json`: `"<key>": {"dir": "<exact folder name>", "table": "<column name>"}`. Keys are short,
     lowercase, letters/digits only, unique (e.g. `floor05`).
   - `deck.js` `RUNS`: append `{id, name, short, stage, slot, lam, cmd, commit}`, following the existing
     entries: `name` ≤ 26 chars, `short` ≤ 12 chars (chart labels), `lam` = "Laplacian / normal
     consistency / edge" weights, `cmd` = the flags that differ from the closest earlier run, `commit`
     from the run's `env.txt`. Take what the run tests from its YAML comment
     (`/data/manousis/cluster_yamls/triangle_splatting/devel4/`) and `logs/config.txt`.
   - `slot`: next unused colour slot. `chrome.html` defines `--s1`…`--s12`. If more are needed, add
     `--s13`… in all three colour blocks (light `:root`, the dark `@media` block, `:root[data-theme="dark"]`),
     chosen to be distinct from the existing ones in both themes.

3. **Stage text.** Add the runs to `STAGES` in `deck.js`: a new stage `{k:'5', t, runs, show, res, what,
   saw}` for a new round of experiments, or an existing stage if they are direct follow-ups. Write `what`
   (what was changed) and 2–4 `saw` bullets with the observed numbers (vs the closest reference run;
   ±0.1 dB and ±1% F-score are noise, one seed). Also update anything that counts runs or stages: the
   `stages` slide title and notes in `SLIDES` ("Four stages"), the cover lede ("nine runs") and the cover
   notes. Keep the existing writing style: short, factual sentences.
   Optionally add a preset button to the compare or mesh-viewer slide if the new run has an obvious
   comparison partner.

4. **Mesh images.** `cd /data/manousis/dev/triangle-splatting2 && .venv/bin/python
   /data/manousis/cluster-results/devel4/deck/render_geo.py <key> [<key> ...]` (GPU; can take a few minutes per mesh).

5. **Build.** `cd /data/manousis/cluster-results/devel4/deck && python3 build.py`. It must end with
   `checks passed`; fix every listed problem and rebuild.

6. **Look at it.** Firefox is a snap and cannot read `/tmp`, so copy to a folder under `$HOME`:
   ```bash
   W=$HOME/deckcheck; rm -rf $W; mkdir -p $W/prof; K=/data/manousis/cluster-results/devel4/deck
   (echo '<!doctype html><html><head><meta charset=utf8></head><body>'; cat $K/index.html; echo '</body></html>') > $W/p.html
   cp -r $K/v $W/
   for s in stages results tradeoff curves compare; do
     timeout 120 firefox --headless --profile $W/prof --window-size=1400,1300 --screenshot $W/$s.png "file://$W/p.html#$s" >/dev/null 2>&1
   done
   ```
   Read the PNGs. Check that the new runs appear with labels that don't overlap badly, and that nothing is
   broken. The compare slide opens with the default runs; that is fine. Delete `$W` afterwards.

7. **Publish.**
   - Stage the files: the publish tool only accepts files under the session's scratchpad directory (use
     the one from your environment) or the repo. Copy `index.html` and only the **new or changed** `v/`
     files (normally `v/s_<key>.jpg` and `v/g_<key>.jpg` for each new run; all of `v/` for a full rebuild)
     into `<scratchpad>/deckpub/` keeping the `v/` subfolder.
   - `Artifact` publish with `url` = the deck URL, `file_path` = `<scratchpad>/deckpub/index.html`,
     `root` = `<scratchpad>/deckpub`, `files` = `{"v/s_<key>.jpg": "v/s_<key>.jpg", ...}`, and a short
     `label` (e.g. "Add runs floor05, floor07"). Files you leave out stay published.
   - If it is refused because the live version was not viewed: Read the saved file it names in full, merge
     any differences (step 1), rebuild, and publish again. If it says the content is identical to an
     already-checked version, resend the same call once.
   - After success: `cp index.html published_index.html`, and `Artifact` action `list`, `scope: "files"`
     to confirm the new files are there (total must stay ≤ 255).
   - If you do not have the Artifact tool, stop after step 6 and return the exact publish parameters
     (paths and `files` map) so the main session can publish.

8. **Report** (final message, short): runs added (key, name, stage), runs skipped and why, stage text you
   wrote (quote it), any outside edits you merged, statements on existing slides that the new numbers
   may contradict, the published version id, and the deck URL.

## Full rebuild

If asked to recreate the deck (e.g. images lost or styling broken): `python3 build.py --force-strips`,
`render_geo.py init <every key in runs.json>`, then steps 5–7 publishing **all** files in `v/`.
