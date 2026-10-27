# Approved illustrated replay design

The 18-soup draft was approved and promoted into the shared HTML viewer for
scripted and evaluation replays, including the sushi scenario. Both the
[canonical replay](../media/18-soups/replay.html) and the
[original draft link](../media/18-soups-ui/replay.html) now use that shared viewer.
Use **Illustrated / Original** to compare the same recorded step. `#step=34`
starts with one pot ready and another cooking.

All 401 recorded states and original PNG frames are preserved. The result remains
18 deliveries, 360 reward, and zero wrong deliveries; the exact optimum remains
unproved. No simulator or policy changes are part of the UI work.

The shared template is `overcooked-v2/replay_template.html`, with presentation
adapters in `replay_scene.py` and portable packaging in `replay.py`.
[Shared artwork, prompts, scope, and verification notes](../../../overcooked-v2/assets/replay/README.md).
The local `assets/` folder preserves the first draft's artwork and generation prompt.

To rebuild this original draft URL from the canonical recorded episode, run from
the repository root with standard-library Python:

```sh
python3 docs/theoretical-maximum/ui-draft/build.py
```

The builder uses the shared viewer and embeds all artwork and frames. It performs
no simulation, evaluation, or training. [First draft preview](../media/18-soups-ui/preview.png).
