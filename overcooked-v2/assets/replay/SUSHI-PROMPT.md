# Sushi and general-layout sprite atlas

Tool: built-in `image_gen.imagegen`, `transparent_background=true`.
Style reference: `kitchen-atlas.png` (no commercial-game assets).

Prompt:

```text
Use case: stylized-concept
Asset type: second sprite atlas for a cooking simulator, matching the supplied reference atlas.
The provided image is ONLY a style reference: match its warm softly painted 2D game art, overhead/high camera, clear silhouettes, white ceramic, warm wood and stainless steel. Create NEW assets listed below. Do not recreate the reference image.
Create exactly 5 columns by 4 rows of 20 isolated sprites on a genuinely transparent background, regular equally sized square cells, total 5:4 aspect ratio. Each sprite centered and completely within the central 80% of its own cell, clear transparent gaps. No text, grid, labels, digits, or background. Keep decorative detail readable at small sizes.
EXACT ORDER, left to right:
Row 1: one whole green cucumber; a little pile of chopped cucumber rounds; a small heap of uncooked white rice; three dark green nori seaweed sheets; a small wooden crate with pink prawns.
Row 2: wooden chopping board with a small chef knife; square stainless steel kitchen sink with faucet and blue water; clean plate drying rack with white plates; wooden return tray with dirty white plates; square dark kitchen waste bin.
Row 3: one dirty white plate with brown food stains; round stainless steel stockpot filled with cooked white rice with TWO short side handles; one white plate with six cucumber maki sushi rolls; wooden supply box with uncooked white rice; wooden supply crate filled with whole green cucumbers.
Row 4: wooden supply box with nori sheets; generic ingredient supply crate with several colourful cloth ingredient pouches and no identifiable vegetables; one small neutral beige tied cloth ingredient pouch; small upright wooden recipe display board with simple ingredient pictograms and NO TEXT; small square recipe-selection switch console with one clearly visible brass push button and NO TEXT.
Consistent equipment/food proportions, same perspective as reference, no chefs, no cross-cell shadows or overlapping sprites. These will be selected using exact source-cell rectangles, so cell alignment matters.
```
