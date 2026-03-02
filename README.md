# restore-batch

`restore-batch` is a Python CLI for batch restoration of scanned print photos.

This implementation currently includes:

- batch ingestion from a directory
- EXIF orientation correction
- color-space normalization to sRGB
- safe alpha flattening
- conversion to a consistent internal float32 RGB working format
- image-type classification (`true-color` vs `near-grayscale`) using chromaticity variance
- conservative white balance correction for color images only
- global luminance-based tonal normalization:
  - color workflow: white balance followed by luminance tonal remap for neutral whites, deep blacks, and natural midtones
  - near-grayscale workflow: robust percentile-based black/white point estimation, full-range luminance remap, and a gentle midtone curve
- edge-aware denoising with conservative defaults and detail guardrails
- edge-aware sharpening on luminance with halo/smooth-region protection
- optional conservative face-local sharpening boost (no generative face enhancement)
- optional conservative red-eye reduction for color photos (detector/guardrail gated)
- face enhancement with conservative local blending and safety guardrails
- final JPG output only, capped at 3 MiB per image (deterministic quality/downscale guardrail)
- JSONL metadata logging per image

## Usage

```bash
./restore-batch /absolute/path/to/input /absolute/path/to/output --recursive
```

or install entry point:

```bash
python3 -m pip install -e .
restore-batch /absolute/path/to/input /absolute/path/to/output --recursive
```

Install pinned face-enhancement dependencies (GFPGAN path):

```bash
python3 -m pip install -e ".[face-enhance]"
python3 -m pip install -r requirements-face-enhance.lock.txt
```

Optional output format:

```bash
restore-batch /absolute/path/to/input /absolute/path/to/output --recursive --output-format jpg
```

JPG size-cap controls:

```bash
restore-batch /absolute/path/to/input /absolute/path/to/output --recursive \
  --jpg-max-mb 3.0 \
  --jpg-quality-max 92 \
  --jpg-quality-min 62 \
  --jpg-quality-step 4 \
  --jpg-downscale-step 0.90 \
  --jpg-min-side 320
```

## Production preset (one-command)

Use the locked production preset with a timestamped output directory:

```bash
./restore-batch-production /absolute/path/to/input
```

Optional output root:

```bash
./restore-batch-production /absolute/path/to/input /absolute/path/to/outputs
```

Pass extra flags after `--` (for example, a fixed GFPGAN model path):

```bash
./restore-batch-production /absolute/path/to/input /absolute/path/to/outputs -- \
  --face-model-path /absolute/path/to/GFPGANv1.4.pth
```

The preset arguments are defined in:

`/Users/scelsenus/Desktop/photo_restore/presets/production.args`

Classification, white balance, tonal, denoise, sharpening, and face enhancement tuning:

```bash
restore-batch /absolute/path/to/input /absolute/path/to/output \
  --classification-threshold 0.0012 \
  --white-balance shades-of-gray \
  --wb-strength 0.6 \
  --wb-white-patch-percentile 99.2 \
  --wb-gray-edge-sigma 1.0 \
  --wb-min-valid-pixels 1024 \
  --wb-confidence-reduce-threshold 0.45 \
  --wb-confidence-skip-threshold 0.08 \
  --tonal-black-percentile 0.5 \
  --tonal-white-percentile 99.5 \
  --tonal-strength 0.65 \
  --tonal-contrast 0.25 \
  --denoise edge-aware-luma \
  --denoise-strength 0.22 \
  --denoise-chroma-strength 0.08 \
  --sharpen edge-aware-unsharp \
  --sharpen-amount 0.30 \
  --sharpen-radius 1.20 \
  --sharpen-threshold 0.02 \
  --face-sharpen \
  --face-sharpen-boost 1.15 \
  --redeye on \
  --redeye-strength 0.70 \
  --redeye-red-ratio 1.65 \
  --redeye-min-red 0.20 \
  --redeye-min-red-excess 0.12 \
  --redeye-max-mask-fraction 0.12 \
  --face-enhance-backend gfpgan \
  --face-model-path /absolute/path/to/GFPGANv1.4.pth \
  --face-enhance-strength 0.35 \
  --face-blend 0.60 \
  --face-feather 0.15 \
  --min-face-px 80 \
  --min-face-conf 0.60 \
  --face-min-change-apply 0.002 \
  --face-max-change 0.12 \
  --face-max-change-reject 0.22 \
  --face-require-eye-evidence \
  --face-eye-check-max-px 10000 \
  --face-min-eye-count 1 \
  --face-small-eye-gate-px 145 \
  --face-small-min-eye-count 2 \
  --face-min-detail-lapvar 7.5e-05 \
  --face-detail-check-max-px 150 \
  --strict-icc \
  --debug-stats
```

`--auto-strength` is enabled by default. Use `--no-auto-strength` to disable adaptive denoise strength.

Face enhancement is part of the standard pipeline. It is applied per-face only when guardrails indicate a likely good outcome; otherwise that face is skipped and the reason is recorded in metadata.
If `--face-model-path` is not provided for GFPGAN, the pipeline auto-resolves a local model file when available and otherwise uses the official GFPGAN model URL.

## Metadata output

By default metadata is written to:

`<output_dir>/normalization_metadata.jsonl`

Each line is a JSON record including source format/mode, orientation handling, image-type classification, white-balance method/scales, tonal black/white points, tonal clipping fractions, denoise method/strength, noise and sharpness proxies before/after, sharpening parameters and edge/face stats, face enhancement backend/guardrail decisions, clipping counters, timing, and output details (including final JPG quality, size-cap metadata, and downscale iterations when needed).
If multiple source files share the same stem in one folder (for example `photo.jpg` and `photo.tif`), output names are disambiguated with source extension suffixes (`photo__jpg.jpg`, `photo__tif.jpg`).

## Extending the pipeline

The pipeline is stage-based:

- `restore_batch/pipeline.py` defines a `PipelineStage` protocol
- `restore_batch/models.py` defines `ImageContext` passed through stages
- `restore_batch/normalize.py` is the first stage

Add future restoration stages by implementing `process(context: ImageContext) -> ImageContext` and appending them in `restore_batch/cli.py`.
