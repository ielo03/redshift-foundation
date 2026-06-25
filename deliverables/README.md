# A Unimodal Foundation Model for DESI Spectra

Physics 303 / CS 486 / CS 686 Final Project  
University of San Francisco, Spring 2026

## Late Note

Sorry this is late. While finishing the deliverables, I hit a storage problem that blocked the final benchmark runs. I had to spend hours freeing space and fixing that before finishing the evaluations and packaging the results.

## Deliverables

```text
deliverables/
  README.md
  requirements.txt
  trained_models/
    local_530k_26m_redshift_token_best_model.pt
    local_530k_26m_redshift_token_best_model_metadata.json
    a100_141k_87m_redshift_token_best_model.pt
    a100_141k_87m_redshift_token_best_model_metadata.json
  benchmark_results/
    final_eval_summary.json
    desi_unseen_hdf5_local_530k_26m_eval/
    desi_unseen_hdf5_a100_141k_87m_eval/
    desi_unseen_hdf5_aion_base_desi_zeroshot_eval/
    ood_sdss_dr17_plate4444_zwarn0_local_530k_26m_eval/
    ood_sdss_dr17_plate4444_zwarn0_a100_141k_87m_eval/
    ood_sdss_dr17_plate4444_zwarn0_aion_base_sdss_rerun_eval/
  scripts/
  src/
```

Install dependencies:

```powershell
cd deliverables
python -m pip install -r requirements.txt
python scripts\check_environment.py
```

Example evaluation command from inside the submitted `deliverables/` folder:

```powershell
python scripts\evaluate_redshift_ood_bundle.py `
  --input <processed_benchmark_bundle.pt> `
  --model trained_models\local_530k_26m_redshift_token_best_model.pt `
  --output-dir <evaluation_output_dir> `
  --d-model 512 `
  --num-layers 8 `
  --nhead 8 `
  --patch-size 61 `
  --batch-size 64 `
  --mask-prob 0.15
```

## Making Predictions With the Submitted Models

Use `scripts/evaluate_redshift_ood_bundle.py` to run either submitted model on a processed spectra bundle. The input bundle should contain a `batch` dictionary with normalized or raw `flux` and redshift labels `z`, following the same format produced by the processing scripts in `scripts/`.

Local 530k/26M model:

```powershell
cd deliverables

python scripts\evaluate_redshift_ood_bundle.py `
  --input <processed_spectra_bundle.pt> `
  --model trained_models\local_530k_26m_redshift_token_best_model.pt `
  --output-dir <prediction_output_dir> `
  --d-model 512 `
  --num-layers 8 `
  --nhead 8 `
  --patch-size 61 `
  --batch-size 64 `
  --mask-prob 0.15
```

A100 141k/87M model:

```powershell
cd deliverables

python scripts\evaluate_redshift_ood_bundle.py `
  --input <processed_spectra_bundle.pt> `
  --model trained_models\a100_141k_87m_redshift_token_best_model.pt `
  --output-dir <prediction_output_dir> `
  --d-model 768 `
  --num-layers 12 `
  --nhead 12 `
  --patch-size 61 `
  --batch-size 32 `
  --mask-prob 0.15
```

Each run writes:

- `redshift_predictions.json`: actual redshifts, predicted redshifts, and metrics
- `redshift_metrics.json`: MAE, RMSE, bias, and reconstruction metrics
- `redshift_pred_vs_actual.png`: prediction scatter plot
- `spectrum_reconstruction.png`: example masked-spectrum reconstruction

The model returns two tensors internally: reconstructed flux values for the spectrum and one scalar encoded redshift prediction per spectrum.

## Goal

The assignment was to build a spectra-only foundation model:

- input: DESI spectra
- auxiliary target: redshift `z`
- no images, magnitudes, Subaru/HSC, or other modalities
- output: redshift prediction and masked spectrum reconstruction

The core assignment requirement was to improve AION-1's redshift handling. In this model, redshift is part of the joint training objective instead of only being predicted later by a downstream head on a frozen encoder.

## Model

Final architecture:

```text
SpectraTransformerWithRedshiftToken
```

I did not use a separate learned tokenizer like AION. I designed the model to feed raw flux patches directly into the transformer: each fixed-width wavelength patch is linearly embedded inside the model, then the model predicts:

- masked flux values through a reconstruction head
- redshift through a jointly trained MLP head

The processed spectra used by the submitted checkpoints have length `7781`. With `patch_size=61`, the model pads each spectrum to `7808` values internally, giving 128 spectral patch tokens plus one learned redshift query token.

Implementation files:

```text
src/neural_baseline.py
scripts/train_redshift_baseline.py
scripts/train_redshift_token_foundation.py
```

## How the Design Evolved

I started with a simple raw-spectrum baseline so there was a stable reference point. That baseline used random spectral masking and redshift prediction, but the redshift error was still high:

```text
baseline MAE: 0.190866
baseline RMSE: 0.351038
```

I then tried redshift-focused variants. The most useful early change was SmoothL1 redshift loss plus tail-focused sampling, since high-redshift objects were rare but important. That improved the small-model redshift result:

```text
tail-focused variant MAE: 0.137573
tail-focused variant RMSE: 0.275679
```

I also tried AION-style tokenizer/codec experiments. Those models could reconstruct token IDs, but the redshift predictions were much worse than the raw-spectrum path. That is why I stopped using the tokenizer for the larger runs.

I tested several other ideas, including domain masking, uncertainty prediction, coarse-to-fine redshift prediction, redshift bins, and redshift-only variants. None beat the raw-spectrum/redshift-token direction, so I focused on scaling the best working design instead of adding more late-stage variants.

I also tried simply scaling an earlier raw tail model on a larger 139k HDF5 split. That did not immediately help; the early full-HDF5 evaluations were worse than the smaller best runs. This pushed me toward the final redshift-token foundation wrapper, where redshift prediction is built directly into the transformer training objective.

Short version of the experiment path:

| Step | What I tried | Outcome |
|---|---|---|
| Raw baseline | masked reconstruction + redshift prediction | worked, but redshift error was high |
| Tail-focused variant | SmoothL1 + more high-z emphasis | much better redshift MAE |
| AION codec/tokenizer | learned token reconstruction path | token reconstruction worked better than redshift |
| Other heads | domain, uncertainty, coarse-to-fine, token bins | useful tests, not final direction |
| Large raw tail model | scaled earlier raw tail model to 139k HDF5 | did not beat smaller best runs |
| Redshift-token model | joint redshift + reconstruction objective | best local direction |
| 100k -> 530k fine-tune | trained 100k first, then continued on 530k | best unseen DESI model |

## Tokenization Choice

Early small models using a tokenizer/codec path performed much worse. Based on that, I decided to use raw spectral patches in the later large runs. The assignment allowed custom tokenization as long as we could explain it.

In the final model, "tokens" just means internal transformer patch embeddings. There is no separate tokenizer/codebook stage.

## Submitted Models

### Local 530k/26M

```text
file: trained_models/local_530k_26m_redshift_token_best_model.pt
training spectra: 530,071
train/val/test: 318,042 / 79,511 / 132,518
initial training: 100k DESI, 150 epochs
fine-tune: initialized from the 100k best model, then continued on 530k
input length: 7781
d_model: 512
layers: 8
heads: 8
patch_size: 61
parameters: 26.01M
```

I include this model because it was best on unseen DESI.

The 100k starting checkpoint was:

```text
outputs/token_large_lr1e4_150ep_final/best_model.pt
100k result MAE: 0.076195
100k result RMSE: 0.203432
```

The 530k fine-tune used `lr=5e-5` instead of `1e-4` because it was adapting an already good model rather than training from scratch.

### A100 141k/87M

```text
file: trained_models/a100_141k_87m_redshift_token_best_model.pt
training spectra: 141,799
train/val/test: 85,079 / 21,270 / 35,450
input length: 7781
d_model: 768
layers: 12
heads: 12
patch_size: 61
parameters: 86.73M
```

I trained this model on fewer spectra because Colab RAM/Drive limits prevented full 530k A100 training. It generalized better to SDSS by MAE and bias.

The assignment mentioned 300M parameters as a realistic target. I did not reach that scale in the final trained models; the largest final model here is 86.73M parameters. I use `87M` in the file names so the deliverable names match the actual model.

## Data

### Hugging Face DESI Parquet

Source:

```text
MultimodalUniverse/desi
edr_sv3/*.parquet
```

Download:

```powershell
cd <project root>

@'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MultimodalUniverse/desi",
    repo_type="dataset",
    allow_patterns="edr_sv3/*.parquet",
    local_dir="data/mmu_desi_hf",
)
'@ | .\final_project_venv\Scripts\python.exe -
```

Process:

```powershell
.\final_project_venv\Scripts\python.exe scripts\process_mmu_desi_parquet.py `
  --input data\mmu_desi_hf\edr_sv3\*.parquet `
  --output data\processed\mmu_desi_hf_edr_sv3.pt `
  --summary outputs\tables\mmu_desi_hf_edr_sv3_summary.json
```

### Flatiron MMU DESI HDF5

Source:

```text
https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/desi/edr_sv3/
```

Download shards:

```powershell
cd <project root>

$base = "https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/desi/edr_sv3"
$out = "data\raw_full"

$healpix = @(
  1002,1599,1619,1625,1628,1629,1703,1709,1728,
  583,589,594,620,624,634,637,657,701,703,958
)

foreach ($hp in $healpix) {
  $dir = Join-Path $out "healpix=$hp"
  New-Item -ItemType Directory -Force $dir | Out-Null
  $url = "$base/healpix=$hp/001-of-001.hdf5"
  $target = Join-Path $dir "001-of-001.hdf5"
  curl.exe -L -C - --fail --retry 10 --retry-delay 10 --connect-timeout 30 `
    -o "$target" "$url"
}
```

Process:

```powershell
$files = Get-Content outputs\tables\good_local_hdf5_files.txt

.\final_project_venv\Scripts\python.exe scripts\process_raw_hdf5.py `
  --input $files `
  --output data\processed\mmu_desi_raw_full20_all.pt `
  --summary outputs\tables\mmu_desi_raw_full20_all_summary.json
```

Combine:

```powershell
.\final_project_venv\Scripts\python.exe scripts\combine_processed_bundles.py `
  --input data\processed\mmu_desi_hf_edr_sv3.pt `
          data\processed\mmu_desi_raw_full20_300k.pt `
          data\processed\mmu_desi_raw_full20_remaining.pt `
  --output data\processed\mmu_desi_combined_hf_hdf5_all_lean.pt `
  --summary outputs\tables\mmu_desi_combined_hf_hdf5_all_lean_summary.json `
  --lean
```

`process_raw_hdf5.py` also supports `--skip-items` to avoid reprocessing already processed spectra.

## Evaluation Data

### Unseen DESI

I used a Flatiron healpix shard not included in the training shard list.

Download:

```powershell
cd <project root>

$base = "https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/desi/edr_sv3"
$out = "data\raw_unseen_desi"
$hp = 720

$dir = Join-Path $out "healpix=$hp"
New-Item -ItemType Directory -Force $dir | Out-Null

curl.exe -L -C - --fail --retry 10 --retry-delay 10 --connect-timeout 30 `
  -o "$dir\001-of-001.hdf5" "$base/healpix=$hp/001-of-001.hdf5"
```

Process:

```powershell
.\final_project_venv\Scripts\python.exe scripts\process_raw_hdf5.py `
  --input data\raw_unseen_desi\healpix=720\001-of-001.hdf5 `
  --output data\processed\desi_unseen_hdf5_eval.pt `
  --summary outputs\tables\desi_unseen_hdf5_eval_summary.json
```

Count:

```text
13,202 spectra
```

### SDSS OOD

OOD data came from SDSS DR17 eBOSS `speclite` spectra:

```text
plate: 4444
mjd: 55538
fibers: 0001-1000
clean subset: ZWARNING == 0
final count: 816 spectra
```

Download:

```powershell
cd <project root>

$out = "data\ood\sdss_dr17_plate4444_speclite"
New-Item -ItemType Directory -Force $out | Out-Null

$base = "https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/4444"

1..1000 | ForEach-Object {
  $fiber = "{0:d4}" -f $_
  $url = "$base/spec-4444-55538-$fiber.fits"
  $dest = "$out\spec-4444-55538-$fiber.fits"

  try {
    Invoke-WebRequest -Uri $url -OutFile $dest -ErrorAction Stop
    Write-Host "downloaded fiber $fiber"
  } catch {
    Write-Host "skipped fiber $fiber"
    if (Test-Path $dest) { Remove-Item $dest }
  }
}
```

Convert to our model format:

```powershell
.\final_project_venv\Scripts\python.exe scripts\process_sdss_speclite_ood.py `
  --input-dir data\ood\sdss_dr17_plate4444_speclite `
  --reference-bundle data\processed\_smoke_mmu_desi_hf_256.pt `
  --output data\processed\ood_sdss_dr17_plate4444_speclite.pt `
  --summary outputs\tables\ood_sdss_dr17_plate4444_speclite_summary.json
```

I used the SDSS conversion script to resample SDSS spectra onto the DESI wavelength grid.

## Training Commands

### Initial Local 100k/26M

This produced the checkpoint used to initialize the 530k model.

```powershell
.\final_project_venv\Scripts\python.exe scripts\train_redshift_token_foundation.py `
  --input data\processed\mmu_desi_hf_edr_sv3.pt `
  --output-dir outputs\token_large_lr1e4_150ep_final `
  --epochs 150 `
  --d-model 512 `
  --num-layers 8 `
  --nhead 8 `
  --patch-size 61 `
  --batch-size 64 `
  --gradient-accumulation 1 `
  --lr 1e-4 `
  --alpha 1.25 `
  --reconstruction-weight 1 `
  --mask-prob 0.15 `
  --tail-power 1.0 `
  --patience 25
```

### Local 530k/26M

```powershell
.\final_project_venv\Scripts\python.exe scripts\train_redshift_token_foundation.py `
  --input data\processed\mmu_desi_combined_hf_hdf5_all_lean.pt `
  --output-dir outputs\token_large_530k_finetune_lr5e5_tail1 `
  --epochs 999 `
  --d-model 512 `
  --num-layers 8 `
  --nhead 8 `
  --patch-size 61 `
  --batch-size 64 `
  --gradient-accumulation 1 `
  --lr 5e-5 `
  --alpha 1.25 `
  --reconstruction-weight 1 `
  --mask-prob 0.15 `
  --tail-power 1.0 `
  --patience 999 `
  --init-model outputs\token_large_lr1e4_150ep_final\best_model.pt
```

### A100 141k/87M

```bash
python scripts/train_redshift_token_foundation.py \
  --input data/processed/mmu_desi_combined_lean.pt \
  --output-dir /content/drive/MyDrive/redshift_foundation_outputs/a100_xlarge_lean_b256_80ep \
  --epochs 80 \
  --d-model 768 \
  --num-layers 12 \
  --nhead 12 \
  --patch-size 61 \
  --batch-size 256 \
  --gradient-accumulation 1 \
  --lr 5e-5 \
  --alpha 1.25 \
  --reconstruction-weight 1 \
  --mask-prob 0.15 \
  --tail-power 1.0 \
  --patience 15
```

The 512-wide model trained well with `lr=1e-4`. For the larger 768-wide model, I switched to `5e-5` because it was more stable.

## Evaluation Commands

Local 530k/26M:

```powershell
.\final_project_venv\Scripts\python.exe scripts\evaluate_redshift_ood_bundle.py `
  --input data\processed\desi_unseen_hdf5_eval.pt `
  --model deliverables\trained_models\local_530k_26m_redshift_token_best_model.pt `
  --output-dir outputs\desi_unseen_hdf5_local_530k_26m_eval `
  --d-model 512 `
  --num-layers 8 `
  --nhead 8 `
  --patch-size 61 `
  --batch-size 64 `
  --mask-prob 0.15
```

A100 141k/87M:

```powershell
.\final_project_venv\Scripts\python.exe scripts\evaluate_redshift_ood_bundle.py `
  --input data\processed\desi_unseen_hdf5_eval.pt `
  --model deliverables\trained_models\a100_141k_87m_redshift_token_best_model.pt `
  --output-dir outputs\desi_unseen_hdf5_a100_141k_87m_eval `
  --d-model 768 `
  --num-layers 12 `
  --nhead 12 `
  --patch-size 61 `
  --batch-size 32 `
  --mask-prob 0.15
```

AION DESI zero-shot:

```powershell
.\final_project_venv\Scripts\python.exe scripts\evaluate_aion_zero_shot_redshift.py `
  --input data\processed\desi_unseen_hdf5_eval.pt `
  --aion-dir models\aion-base `
  --output-dir outputs\desi_unseen_hdf5_aion_base_desi_zeroshot_eval `
  --batch-size 32 `
  --spectrum-modality desi `
  --raw-flux-key flux_raw
```

AION SDSS zero-shot:

```powershell
.\final_project_venv\Scripts\python.exe scripts\evaluate_aion_zero_shot_redshift.py `
  --input data\processed\ood_sdss_dr17_plate4444_speclite_zwarn0.pt `
  --aion-dir models\aion-base `
  --output-dir outputs\ood_sdss_dr17_plate4444_zwarn0_aion_base_sdss_rerun_eval `
  --batch-size 32 `
  --spectrum-modality sdss `
  --raw-flux-key flux_raw
```

AION note: I tried to run AION on DESI using the intended downstream redshift-head setup, but I could not find a released pretrained redshift head to download. I searched the local AION files and model assets and only found the base model/codecs. Because of that, these AION results use zero-shot token generation, not a trained downstream regressor.

## Evaluation Design

I evaluated on two datasets:

- **Unseen DESI:** a Flatiron/MMU DESI HDF5 healpix shard that was not in the training shard list.
- **SDSS OOD:** SDSS DR17 spectra from a different instrument, converted onto the DESI wavelength grid.

This matches the assignment's two evaluation goals: redshift prediction and masked spectrum reconstruction, with an extra non-DESI OOD check.

I included AION-base because the assignment is framed around AION-1. The comparison is useful, but it is not perfectly fair:

- AION-base is a released pretrained model, not a model I trained on this DESI-only setup.
- I tried to use AION's downstream redshift-head path for DESI, but could not find a released head to download.
- I therefore evaluated AION through zero-shot redshift token generation.
- Our models are DESI-only and trained directly for redshift + reconstruction.

That explains the strange AION split. On DESI, AION-base collapsed to predicting `z = 0`, likely because I was using its zero-shot token path rather than a trained downstream redshift head. On SDSS, AION did very well because I could use its native SDSS spectrum codec. That result shows AION has useful SDSS-specific pretraining, but it is not the same as a DESI-only foundation model generalizing to SDSS.

## Results

### Redshift Prediction

| Dataset | Model | MAE | RMSE | Bias | N |
|---|---:|---:|---:|---:|---:|
| Unseen DESI | Local 530k/26M | **0.05957** | **0.16800** | 0.00291 | 13202 |
| Unseen DESI | A100 141k/87M | 0.06948 | 0.18459 | -0.00712 | 13202 |
| Unseen DESI | AION-base DESI codec | 0.54143 | 0.77925 | -0.54131 | 13202 |
| SDSS OOD | Local 530k/26M | 0.19447 | **0.32334** | -0.04405 | 816 |
| SDSS OOD | A100 141k/87M | **0.16829** | 0.36340 | **-0.02639** | 816 |
| SDSS OOD | AION-base SDSS codec | 0.01686 | 0.15589 | -0.00182 | 816 |

### Masked Reconstruction

| Dataset | Model | Masked Flux MAE | Masked Flux RMSE |
|---|---:|---:|---:|
| Unseen DESI | Local 530k/26M | **0.54660** | **0.76999** |
| Unseen DESI | A100 141k/87M | 0.55045 | 0.77426 |
| SDSS OOD | Local 530k/26M | **0.40804** | 0.56191 |
| SDSS OOD | A100 141k/87M | 0.40926 | **0.56102** |

I did not report AION reconstruction because this comparison only used AION redshift token generation.

## Takeaways

- The local 530k/26M model is best on unseen DESI.
- The A100 141k/87M model is better on SDSS OOD by MAE and bias.
- AION-base collapsed to `z = 0` on DESI zero-shot.
- AION-base did very well on SDSS with its native SDSS codec, but that is not a DESI-only comparison.

My interpretation:

- The local model wins on unseen DESI because it saw much more DESI data: it trained on 100k first, then fine-tuned on 530k.
- The A100 model probably transfers better to SDSS because it is larger: 87M parameters, 12 layers, and wider embeddings. Even with fewer spectra, it may have learned smoother spectral features.
- The A100 SDSS RMSE is worse than the local model, so it still has larger outlier errors. Its MAE and bias are better, which is why I describe it as better on average for SDSS.
- AION's SDSS result should be treated as context, not as the main benchmark, because it uses AION's native SDSS pathway.
- A fairer AION comparison would require training or obtaining the missing downstream redshift head.

## README Coverage

This README includes the main items I wanted in the final submission:

- late note: included, but shorter than the original draft so it does not dominate the report
- model-running commands: included under `Evaluation Commands`
- recreation commands: data download, processing, combining, training, and evaluation commands are included
- data used: HF DESI parquet, Flatiron DESI HDF5, unseen DESI, and SDSS OOD are listed
- processing scripts: included under `deliverables/scripts/` and referenced in commands
- training details: sample counts, splits, model sizes, and hyperparameters are included
- 100k to 530k lineage: included in `Submitted Models` and `Training Commands`
- A100 limitation: included under `Submitted Models`
- OOD conversion: included under `SDSS OOD`
- design decisions: included under `How the Design Evolved`
- learning-rate issue: included under `Training Commands`
- results and interpretation: included under `Results` and `Takeaways`
- AION comparison: included under `Evaluation Design`, including the missing downstream-head issue
- future work: included below

The main difference from my original plan is that the late note is shorter. I kept it brief because the storage problem mattered for timing, but the model design and results should be the focus.

Another difference is the model naming. I originally described the A100 model loosely as a 300M-class run, but the final trained A100 model is actually 86.73M parameters. I renamed it `a100_141k_87m` to avoid overstating the scale.

## What Went Right / Wrong

What went right:

- The final models satisfy the scope limit: spectra plus redshift only.
- Joint redshift training worked better than the early baselines.
- The 100k -> 530k fine-tune produced the best unseen DESI result.
- The larger A100 model gave the best SDSS OOD MAE and bias among my models.
- The deliverables include weights, metadata, scripts, dependencies, commands, plots, predictions, and metrics.

What went wrong:

- The tokenizer/codec path did not give good redshift predictions.
- Simply scaling the earlier raw tail model did not automatically improve results.
- Colab RAM and Drive limits prevented full 530k training on A100.
- I did not have time to train a fair AION downstream redshift head.
- Final storage problems delayed the last evaluations and packaging.

## What Could Be Better

- Train the 87M model on the full 530k DESI set.
- Train longer.
- Sweep learning rates and model sizes more carefully.
- Explore the model-size ceiling more systematically. I tried larger directions locally/A100, but I did not find the top bound or train a true 300M model.
- Train a fair AION downstream redshift head on frozen AION embeddings.
- Evaluate on more SDSS plates and more non-DESI instruments.
- Use a streaming dataset instead of large `.pt` bundles loaded into CPU RAM.

## References

Project context:

```text
PHYS303_Final-Project_2026.pdf
deliverables/benchmark_results/final_eval_summary.json
```

Data/model sources:

- `MultimodalUniverse/desi`
- `https://users.flatironinstitute.org/~polymathic/data/MultimodalUniverse/v1/desi/edr_sv3/`
- `https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/`
- AION assets from the `polymathic-ai` organization
