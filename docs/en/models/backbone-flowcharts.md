# Backbone Flowcharts

Simple structure-only diagrams for:

- `legnet-small-obb.yaml`
- `Mamba-YOLO-L.yaml`
- `VSSBlock`
- LEGNet `LFEModule`

## LEGNet-small-OBB Backbone

```mermaid
flowchart TD
    A["Input"] --> B["Stem"]
    B --> C["Stage 1<br/>LFEModule x1"]
    C --> C1["P2"]
    C --> D["DRFD downsample"]
    D --> E["Stage 2<br/>LFEModule x4"]
    E --> E1["P3"]
    E --> F["DRFD downsample"]
    F --> G["Stage 3<br/>LFEModule x4"]
    G --> G1["P4"]
    G --> H["DRFD downsample"]
    H --> I["Stage 4<br/>LFEModule x2"]
    I --> I1["P5"]
    E1 --> J["OBB head input"]
    G1 --> J
    I1 --> J
```

## Mamba-YOLO-L Backbone

```mermaid
flowchart TD
    A["Input"] --> B["SimpleStem"]
    B --> C["VSSBlock x3"]
    C --> C1["P2/4"]
    C --> D["VisionClueMerge"]
    D --> E["VSSBlock x3"]
    E --> E1["P3/8"]
    E --> F["VisionClueMerge"]
    F --> G["VSSBlock x9"]
    G --> G1["P4/16"]
    G --> H["VisionClueMerge"]
    H --> I["VSSBlock x3"]
    I --> I1["P5/32"]
    I --> J["SPPF"]
```

## Mamba-HRNet-OBB Backbone

```mermaid
flowchart TD
    A["Input"] --> B["SimpleStem"]
    B --> C["Stage 1<br/>Single branch<br/>VSSBlock stack"]

    C --> D["Transition 1<br/>1 branch -> 2 branches"]
    D --> E["Stage 2<br/>B1 + B2"]
    E --> F["Stage 2 exchange fusion"]

    F --> G["Transition 2<br/>2 branches -> 3 branches"]
    G --> H["Stage 3<br/>B1 + B2 + B3"]
    H --> I["Stage 3 exchange fusion"]

    I --> J["Transition 3<br/>3 branches -> 4 branches"]
    J --> K["Stage 4<br/>B1 + B2 + B3 + B4"]
    K --> L["Final 4-branch exchange fusion"]

    L --> M["Final P2"]
    L --> N["Final P3"]
    L --> O["Final P4"]
    L --> P["Final P5"]
    P --> Q["SPPF"]
```

## YOLO-Mamba-Seg-EdgeVSS-All Backbone

```mermaid
flowchart TD
    A["Input"] --> B["EdgeStem"]
    B --> C["EdgeVSSBlock x3"]
    C --> C1["P2/4"]
    C --> D["VisionClueMerge"]
    D --> E["EdgeVSSBlock x3"]
    E --> E1["P3/8"]
    E --> F["VisionClueMerge"]
    F --> G["EdgeVSSBlock x9"]
    G --> G1["P4/16"]
    G --> H["VisionClueMerge"]
    H --> I["EdgeVSSBlock x3"]
    I --> I1["P5/32"]
    I --> J["SPPF"]
```

## VSSBlock

```mermaid
flowchart TD
    A["Input"] --> B["1x1 Conv + BN + SiLU"]
    B --> C["LSBlock"]
    C --> D["Norm"]
    D --> E["SS2D"]
    B --> F["Residual add"]
    E --> F
    F --> G["Norm2"]
    G --> H["RGBlock / MLP"]
    F --> I["Residual add"]
    H --> I
    I --> J["Output"]
```

## EdgeVSSBlock

```mermaid
flowchart TD
    A["Input"] --> B["1x1 Conv + BN + SiLU"]
    B --> C["LSBlock"]
    C --> D["Edge branch"]
    D --> D1["LFEA + Scharr/Gaussian"]
    D1 --> E["Norm"]
    E --> F["SS2D"]
    B --> G["Residual add"]
    F --> G
    G --> H["Norm2"]
    H --> I["RGBlock / MLP"]
    G --> J["Residual add"]
    I --> J
    J --> K["Output"]
```

## LEGNet LFEModule

```mermaid
flowchart TD
    A["Input x"] --> B["Edge branch"]
    B --> B1["Scharr or Gaussian"]
    A --> C["LFEA"]
    B1 --> C
    C --> D["MLP<br/>1x1 -> BN -> Act -> 1x1"]
    D --> E["DropPath + BN"]
    A --> F["Residual add"]
    E --> F
    F --> G["Output"]
```
