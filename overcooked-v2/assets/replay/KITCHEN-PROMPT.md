# Sprite atlas generation

Tool: built-in `image_gen.imagegen`, `transparent_background=true`.

Prompt:

```text
Use case: stylized-concept
Asset type: production 2D game sprite atlas for an overhead cooking simulator.
Create ONE transparent PNG sprite atlas with exactly 5 columns and 4 rows, 20 equally sized square cells. Overall image aspect ratio exactly 5:4, preferably 2000x1600 pixels. Each sprite centered precisely in its cell, entirely inside the middle 80% of that cell, empty transparent margin at all cell boundaries. No text, numbers, grid lines, labels, dividers, shadows outside sprite bounds, or background.
Style: consistent polished illustrated 2D game art, soft hand-painted shading, warm wood, cream and slate, clear silhouette, viewed directly from above with a slight visible front edge. Cute original chefs with fluffy white chef hats, coloured aprons, small shoes, hands visible. Chefs occupy about 72% of square cell, other props about 75%. Solid opaque art with genuinely transparent surroundings. All sprites consistent scale and camera.
EXACT sprite contents in row-major order:
Row 1 (five cells, left to right): red-apron chef facing NORTH/up showing back; red-apron chef facing EAST/right; red-apron chef facing SOUTH/down showing face; red-apron chef facing WEST/left; blue-apron chef facing NORTH/up showing back.
Row 2: blue-apron chef facing EAST/right; blue-apron chef facing SOUTH/down showing face; blue-apron chef facing WEST/left; a square pale cream floor tile with subtle stone texture; a square warm wooden countertop tile.
Row 3: wooden supply crate filled with golden onions; neat stack of white ceramic plates; square charcoal stovetop with one circular burner; round empty stainless steel stockpot with TWO small side handles and dark empty interior; a single whole golden onion with little green stem.
Row 4: single empty white plate seen from above; white plate containing golden onion soup; square green serving pass/hatch with a small brass bell; square dark slate stone countertop tile; round stainless steel stockpot with golden soup inside and TWO small side handles.
No utensils or food held by the chefs; objects carried will be composited separately by code. Do not merge adjacent cells, do not invent extra sprites. Each sprite must be completely isolated. Treat the grid positions as strict layout, not suggested arrangement.
```
