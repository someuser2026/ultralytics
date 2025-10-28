# Using the Modular Necks in YAML

This guide shows exactly how to plug the new FPN-family necks into your Ultralytics-style YAML graphs. It covers inputs/outputs, options, and ready-to-paste examples for each neck.

> **Key design rule:** the neck **does not** create extra pyramid levels (no P1/P6 internally).  
> You pass in *whatever* feature maps you want, and the neck returns the same number of maps in the **same order** (fine→coarse, if that’s how you fed them).

---

## 1) What the neck expects & returns

- **Inputs:** a **list** of feature maps, typically from your backbone (e.g., strides 4/8/16/32).  
  Order is expected to be **fine → coarse** (small stride → large stride). If your backbone emits in a different order, use `Index` layers to reorder.

- **Outputs:** a **list** of feature maps with the **same length and order** as the inputs.  
  You can pass this list directly to heads like `Detect`, `Segment`, or grab levels by `Index` if your runtime requires it.

---

## 2) Where to put the neck in YAML

Typical pattern:

1. **Extract** backbone features you want (via `Index`).
2. **Feed** them to a neck module (one line: pass the whole list).
3. **Feed** neck outputs to your head (`Detect` / `Segment` / `OBB`).

```yaml
# Example: 4 backbone outputs (strides 4, 8, 16, 32)
head:
  # 1) Grab backbone features (assumes the backbone returns a list-like output)
  - [0, 1, Index, [0]]  # P3 (stride 4)
  - [0, 1, Index, [1]]  # P4 (stride 8)
  - [0, 1, Index, [2]]  # P5 (stride 16)
  - [0, 1, Index, [3]]  # P6 (stride 32)

  # 2) Pass all four into the neck (use a list of 'from' indices)
  - [[1, 2, 3, 4], 1, FPN,   [256, {normalize_channels: true, fusion: add, drop_td: 0.05}]]

  # 3) Send the neck's list output to the head
  #    (If your runtime passes list outputs as-is, use -1)
  - [-1, 1, Detect, [nc]]
```

> If your runtime doesn’t directly pass lists between modules, insert `Index` after the neck to select each level individually before the head:
>
> ```yaml
> - [-1, 1, Index, [0]]
> - [-2, 1, Index, [1]]
> - [-3, 1, Index, [2]]
> - [-4, 1, Index, [3]]
> - [[-4, -3, -2, -1], 1, Detect, [nc]]
> ```

---

## 3) Common options (apply to all necks)

Every neck shares these kwargs (pass them in the YAML args map):

- `out_channels` **(int, required):** final channel width at each level (e.g., 256).
- `normalize_channels` **(bool, default: true):** 1×1 aligns all inputs to `out_channels` before fusion.  
  Keep `true` when input channels differ (most backbones).
- `fusion` **('add' | 'concat' | 'weighted'):** fusion strategy for FPN-style fusions.  
  - `add`: fast, standard FPN.  
  - `weighted`: BiFPN-style fast normalized weights (learnable).  
  - `concat`: concatenates then 1×1 projects to `out_channels` (used in PAFPN).
- `drop_td` **(float, default: 0.0):** dropout on **top-down** fused features.
- `drop_bu` **(float, default: 0.0):** dropout on **bottom-up** fused features (PAN/PAFPN).
- `attn_cfg` **(map, optional):** plug attention in two places:
  - `per_level`: one of `se|eca|cbam|cbam_c|cbam_s|coord|simam`
  - `after_fuse`: same options (applied after each fusion)
  - Example: `{per_level: se, after_fuse: cbam}`
- `conv_cfg` **(map, optional):** conv policy knobs:
  - `dcn: true|false` (use deformable conv in smoothing/down/upsample convs)
  - `dilation: 1` (global dilation)
  - `groups: 1` (grouped/ depthwise if set to channels)

> ⚠️ Some necks fix their internal fusion mode (e.g., PAFPN uses `concat` by design). Supplying a different `fusion` value to those will be ignored.

---

## 4) Variant-specific options

- **BiFPN**
  - `iterations` (int): number of stacked BiFPN passes (≥1).
- **RecursiveFPN**
  - `passes` (int): number of recursive refinement passes (≥1).
- **AugFPN**
  - `pool_bins` (int): RAP-like pooling bin per level (default 3).

---

## 5) Ready-to-paste YAML snippets

### 5.1 FPN (standard)
```yaml
# inputs: 4 levels from the backbone
- [[1, 2, 3, 4], 1, FPN, [256, {normalize_channels: true, fusion: add, drop_td: 0.05,
                                attn_cfg: {per_level: se, after_fuse: cbam},
                                conv_cfg: {dcn: false, dilation: 1}}]]
# then head:
- [-1, 1, Detect, [nc]]
```

**Weighted fusion FPN**
```yaml
- [[1, 2, 3, 4], 1, FPN, [256, {normalize_channels: true, fusion: weighted}]]
- [-1, 1, Detect, [nc]]
```

### 5.2 PANet (top-down FPN + bottom-up)
```yaml
- [[1, 2, 3, 4], 1, PANet, [256, {normalize_channels: true, fusion: add,
                                  drop_td: 0.05, drop_bu: 0.05}]]
- [-1, 1, Detect, [nc]]
```

### 5.3 PAFPN (YOLOv4-style concat both ways)
```yaml
- [[1, 2, 3, 4], 1, PAFPN, [256, {normalize_channels: true,  # pre-align before concat
                                  drop_td: 0.10, drop_bu: 0.10,
                                  attn_cfg: {per_level: se}}]]
- [-1, 1, Detect, [nc]]
```

### 5.4 BiFPN (stacked)
```yaml
- [[1, 2, 3, 4], 1, BiFPN, [256, {iterations: 2, normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

### 5.5 AugFPN (approx)
```yaml
- [[1, 2, 3, 4], 1, AugFPN, [256, {pool_bins: 3, normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

### 5.6 LibraFPN
```yaml
- [[1, 2, 3, 4], 1, LibraFPN, [256, {normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

### 5.7 RepFPN (reparameterizable)
```yaml
- [[1, 2, 3, 4], 1, RepFPN, [256, {normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

### 5.8 RecursiveFPN
```yaml
- [[1, 2, 3, 4], 1, RecursiveFPN, [256, {passes: 2, normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

### 5.9 Scale-Equalizing FPN (approx)
```yaml
- [[1, 2, 3, 4], 1, ScaleEqualizingFPN, [256, {normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

> **Note:** If you use a registry/factory with short keys, map them to the class names accordingly (e.g., `"sefpn" → ScaleEqualizingFPN"`). In YAML, use **your** actual class/registry naming.

---

## 6) Adding extra levels **outside** the neck (optional)

Since the neck won’t generate P1/P6 internally, create them in YAML and include in the input list.

### Add **P6** (downsample the coarsest level by 2×)
```yaml
# Suppose layer 4 is P6_in (coarsest backbone feature)
- [4, 1, Conv, [${c}, 3, 2]]    # stride-2 downsample to make P_extra
# Now feed 5 levels to the neck:
- [[1, 2, 3, 4, 5], 1, FPN, [256, {normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

### Add **P1** (upsample the finest level by 2×)
```yaml
# Suppose layer 1 is P3_in (finest)
- [1, 1, nn.Upsample, [null, 2, nearest]]  # create P1 from finest
# Feed to the neck (now 5 inputs fine→coarse):
- [[6, 1, 2, 3, 4], 1, PAFPN, [256, {normalize_channels: true}]]
- [-1, 1, Detect, [nc]]
```

> Replace `${c}` with the channel count you want on the synthetic level (or follow up with a 1×1 align before neck).

---

## 7) Attention & Conv quick recipes

### Light channel attention per level
```yaml
- [[1, 2, 3, 4], 1, FPN, [256, {attn_cfg: {per_level: eca}}]]
```

### CBAM after every fusion
```yaml
- [[1, 2, 3, 4], 1, FPN, [256, {attn_cfg: {after_fuse: cbam}}]]
```

### Deformable smoothing/downsample
```yaml
- [[1, 2, 3, 4], 1, PANet, [256, {conv_cfg: {dcn: true}}]]
```

---

## 8) Troubleshooting

- **Mismatched channels during fusion**  
  Set `normalize_channels: true` (default) to auto-align with 1×1.  
  PAFPN already pre-aligns before concat, but it’s still safe to keep it on.

- **Spatial mismatch when fusing**  
  The neck resizes using nearest neighbor. Ensure your inputs are ordered **fine→coarse**.

- **Heads not receiving multiple levels**  
  If your runtime doesn’t pass the list automatically, extract with `Index` after the neck and feed those indices to the head (see Section 2).

- **Using `weighted` fusion**  
  Works best when all inputs are aligned (keep `normalize_channels: true`).

---

## 9) Class names / registry keys

Out of the box (class names):

- `FPN`, `PANet`, `PAFPN`, `BiFPN`, `AugFPN`, `LibraFPN`, `RepFPN`, `RecursiveFPN`, `ScaleEqualizingFPN`

If you use a registry/factory with short keys, map appropriately (e.g., `"sefpn" → ScaleEqualizingFPN"`). In YAML, use **your** actual class/registry naming.

---

## 10) Minimal end-to-end example

```yaml
nc: 1

backbone:
  - [-1, 1, Timm, ['coatnet_0_rw_224.sw_in1k', False, 3, True, [0, 1, 2, 3], 28, null, 'auto', False, False, True, False]]

head:
  # Pull 4 scales from the timm backbone
  - [0, 1, Index, [0]]  # stride 4
  - [0, 1, Index, [1]]  # stride 8
  - [0, 1, Index, [2]]  # stride 16
  - [0, 1, Index, [3]]  # stride 32

  # Neck (choose one)
  - [[1, 2, 3, 4], 1, PAFPN, [256, {normalize_channels: true, drop_td: 0.1, drop_bu: 0.1}]]

  # Detect (the neck returns a list → pass directly if supported)
  - [-1, 1, Detect, [nc]]
```
