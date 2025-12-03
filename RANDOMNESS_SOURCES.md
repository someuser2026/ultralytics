# Sources of Randomness in Ultralytics Codebase

This document catalogs all sources of randomness found in the Ultralytics codebase.

## 1. Seed Initialization and Control

### Primary Seed Initialization
- **File**: `ultralytics/utils/torch_utils.py`
  - `init_seeds(seed=0, deterministic=False)` (lines 586-609)
    - `random.seed(seed)` - Python's random module
    - `np.random.seed(seed)` - NumPy random generator
    - `torch.manual_seed(seed)` - PyTorch CPU random generator
    - `torch.cuda.manual_seed(seed)` - PyTorch CUDA random generator
    - `torch.cuda.manual_seed_all(seed)` - PyTorch multi-GPU random generator
    - `os.environ["PYTHONHASHSEED"]` - Python hash randomization (when deterministic=True)
    - `torch.use_deterministic_algorithms(True)` - PyTorch deterministic algorithms (when deterministic=True)
    - `torch.backends.cudnn.deterministic = True` - CuDNN deterministic mode

- **File**: `ultralytics/engine/trainer.py`
  - Line 133: `init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)`

- **File**: `ultralytics/data/build.py`
  - `seed_worker(worker_id)` (lines 115-119) - Seeds for dataloader workers
    - `np.random.seed(worker_seed)`
    - `random.seed(worker_seed)`
  - Line 213: `generator.manual_seed(6148914691236517205 + RANK)` - PyTorch generator for dataloader

- **File**: `tests/conftest.py`
  - Line 39: `init_seeds()` - Test session initialization

## 2. Data Augmentation Randomness

### File: `ultralytics/data/augment.py`

#### Mosaic Augmentation
- Line 347: `random.randint(0, len(self.dataset) - 1)` - Random image selection
- Line 390: `random.uniform(0, 1)` - Probability check for mosaic application
- Line 446: `random.randint(0, len(self.dataset) - 1)` - Random image index
- Line 567: `random.choices(list(self.dataset.buffer), k=self.n - 1)` - Random buffer sampling
- Line 569: `random.randint(0, len(self.dataset) - 1)` - Random dataset indices
- Line 685: `random.uniform(-x, 2 * s + x)` - Mosaic center coordinates (x, y)

#### MixUp Augmentation
- Line 926: `np.random.beta(32.0, 32.0)` - Mixup ratio from beta distribution

#### CutMix Augmentation
- Line 986: `np.random.beta(self.beta, self.beta)` - CutMix lambda parameter
- Line 993-994: `np.random.randint(width)`, `np.random.randint(height)` - Random cut center
- Line 1028: `np.random.choice(idx)` - Random area selection

#### RandomPerspective (Affine Transformations)
- Lines 1155-1156: `random.uniform(-self.perspective, self.perspective)` - Perspective distortion (x, y)
- Line 1160: `random.uniform(-self.degrees, self.degrees)` - Rotation angle
- Line 1162: `random.uniform(1 - self.scale, 1 + self.scale)` - Scale factor
- Lines 1168-1169: `random.uniform(-self.shear, self.shear)` - Shear angles (x, y)
- Lines 1173-1174: `random.uniform(0.5 - self.translate, 0.5 + self.translate)` - Translation (x, y)

#### RandomHSV
- Line 1486: `np.random.uniform(-1, 1, 3)` - Random HSV gains

#### RandomFlip
- Line 1580: `random.random() < self.p` - Vertical flip probability
- Line 1585: `random.random() < self.p` - Horizontal flip probability

#### Albumentations
- Line 2031: `torch.initial_seed()` - Seed for Albumentations transforms
- Line 2069: `random.random() > self.p` - Probability check for Albumentations

#### RandomCLAHE
- Line 2303: `np.random.random() > self.p` - Probability check
- Line 2310: `np.random.uniform(0.0, self.clip_limit)` - Random clip limit (commented out)

#### RandomGamma
- Line 2395: `np.random.random() > self.p` - Probability check
- Line 2402: `np.random.uniform(self.gamma_range[0], self.gamma_range[1])` - Random gamma value

#### RandomUnsharpMask
- Line 2509: `np.random.random() > self.p` - Probability check
- Line 2517: `np.random.randint(...)` - Random kernel size
- Line 2520: `np.random.uniform(0.1, self.sigma_limit)` - Random sigma
- Line 2521: `np.random.uniform(self.amount_range[0], self.amount_range[1])` - Random amount

#### BGR Channel Ordering
- Line 3763: `random.uniform(0, 1) > self.bgr` - Random BGR/RGB ordering

#### RandomLoadText
- Line 3999: `random.sample(pos_labels, k=self.max_samples)` - Random positive label sampling
- Line 4001: `random.randint(*self.neg_samples)` - Random negative sample count
- Line 4003: `random.sample(neg_labels, k=neg_samples)` - Random negative label sampling
- Line 4025: `random.randrange(len(prompts))` - Random prompt selection
- Line 4032: `random.choices(self.padding_value, k=num_padding)` - Random padding values

## 3. Data Loading and Shuffling

### File: `ultralytics/data/build.py`
- Line 217: `shuffle=shuffle and sampler is None` - DataLoader shuffle parameter
- Line 213: `generator.manual_seed(...)` - Random generator for shuffling

### File: `ultralytics/data/split.py`
- Line 86: `random.shuffle(image_files)` - Shuffle files before splitting
- Line 123: `random.seed(0)` - Seed for reproducibility in autosplit
- Line 124: `random.choices([0, 1, 2], weights=weights, k=n)` - Random split assignment

### File: `ultralytics/data/utils.py`
- Line 76: `random.sample(files, min(max_files, len(files)))` - Random file sampling

### File: `ultralytics/data/base.py`
- Line 340: `random.choice(self.im_files)` - Random image file selection
- Line 374: `random.choice(self.im_files)` - Random image sampling

## 4. Model Architecture Randomness

### Dropout Layers
- **File**: `ultralytics/nn/modules/block.py`
  - Line 2102: `torch.rand(shape, dtype=x.dtype, device=x.device)` - Dropout random mask generation
  - Custom DropPath implementation uses random tensor for stochastic depth

### Model Initialization
- **File**: `ultralytics/nn/modules/head.py`
  - Line 2073: `torch.randn(num_out, d)` - Random parameter initialization for level embedding

### Denoising Training (DETR-style)
- **File**: `ultralytics/models/utils/ops.py`
  - Line 263: `torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)` - Class label noise mask
  - Line 266: `torch.randint_like(idx, 0, num_classes, ...)` - Random class label assignment
  - Line 274: `torch.randint_like(dn_bbox, 0, 2)` - Random sign for box noise
  - Line 275: `torch.rand_like(dn_bbox)` - Random box noise values

### Point Sampling (SAM)
- **File**: `ultralytics/utils/loss.py`
  - Line 2289: `torch.rand(B, k, 2, ...)` - Random point sampling for uncertainty estimation
  - Line 2304: `torch.rand(B, num_points - pick.shape[1], 2, ...)` - Random point filling

### Cascade RCNN Target Sampling
- **File**: `ultralytics/models/cascade_rcnn/targets.py`
  - Line 93: `torch.randperm(pos_inds.numel(), device=device)` - Random positive sample permutation
  - Line 103: `torch.randperm(neg_inds.numel(), device=device)` - Random negative sample permutation
  - Lines 176, 180, 285, 289: Additional `torch.randperm` calls for sampling

## 5. Training Process Randomness

### File: `ultralytics/models/yolo/detect/train.py`
- Line 138: `random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))` - Random image size during training

## 6. Hyperparameter Tuning

### File: `ultralytics/engine/tuner.py`
- Line 279: `random.choices(range(len(x)), weights=weights, k=k)` - Weighted random selection for evolution
- Line 283: `np.random.uniform(lo - alpha * span, hi + alpha * span)` - Random hyperparameter mutation
- Line 334: `np.random.random(ng) < mutation` - Mutation mask
- Line 335: `np.random.randn(ng) * (sigma * gains)` - Random step for hyperparameter evolution

## 7. Utility and Helper Functions

### File: `ultralytics/utils/events.py`
- Line 62: `round(random.random() * 1e15)` - Random session ID generation

### File: `ultralytics/utils/tqdm.py`
- Line 422: `random.randint(10, 20)` - Random iteration count (likely for testing/demos)

### File: `ultralytics/data/converter.py`
- Line 672: `random.randint(480, 640)` - Random image size generation
- Line 676: `random.randint(0, 255)` - Random color generation (RGB)

## 8. Example/Demo Code (Non-Core)

The following files contain random operations but are in example/demo code:
- `examples/YOLOv8-OpenCV-ONNX-Python/main.py`: Color palette generation
- `examples/YOLOv8-TFLite-Python/main.py`: Color palette generation
- `examples/YOLOv8-ONNXRuntime/main.py`: Color palette generation
- `examples/RTDETR-ONNXRuntime-Python/main.py`: Color palette generation
- `examples/YOLOv8-Action-Recognition/action_recognition.py`: Random crop generation

## Summary by Category

1. **Seed Management**: 4 locations (torch_utils, trainer, build, conftest)
2. **Data Augmentation**: ~30+ random operations in augment.py
3. **Data Loading**: 5 locations (shuffling, sampling, splitting)
4. **Model Architecture**: 8+ locations (dropout, initialization, denoising)
5. **Training**: 1 location (random image size)
6. **Hyperparameter Tuning**: 4 locations in tuner.py
7. **Utilities**: 3 locations (events, tqdm, converter)

## Notes

- Most randomness is controlled via `init_seeds()` which sets seeds for Python's `random`, NumPy, and PyTorch
- DataLoader workers have separate seeding via `seed_worker()` function
- When `deterministic=True`, additional deterministic settings are applied
- Some randomness (like dropout) is only active during training (`model.training == True`)
- Albumentations library has its own seed setting mechanism

