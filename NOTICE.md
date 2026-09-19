# Third-party notice

## Transformer Explainer

This project vendors and adapts visualization code from:

**Transformer Explainer: Interactive Learning of Text-Generative Models**  
Polo Club of Data Science, Georgia Institute of Technology  
https://github.com/poloclub/transformer-explainer

Pinned upstream baseline:

`bfe50afba10b9b560b84143ee1107d977defa74f`

Upstream license: MIT License

Copyright (c) 2022 Polo Club of Data Science

The upstream MIT license permits use, copying, modification, distribution, sublicensing and sale, provided that the copyright and permission notice are retained in copies or substantial portions of the upstream software.

### Adapted visualization sources

The following files in this repository contain substantial adaptations of upstream visualization code and carry file-level attribution:

- `src/upstream/VectorCanvas.svelte` ← `src/components/common/VectorCanvas.svelte`
- `src/upstream/MatrixSvg.svelte` ← `src/components/common/MatrixSvg.svelte`
- `src/upstream/SankeyFlow.svelte` ← the generic gradient/path/draw machinery in `src/components/Sankey.svelte`

The Cart-Pole-specific code supplies state/token/model data and selector mappings. The D3 matrix rendering, canvas vector rendering, DOM-to-DOM path construction, gradient treatment, resize redraw strategy, and hover/highlight interaction are intentionally derived from Transformer Explainer rather than independently re-invented.
