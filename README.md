# Anomaly detection evaluation across UCF-Crime categories

This project refactors and extends the testing pipeline and metrics for the [re-implementation](https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch) of [“Real-world Anomaly Detection in Surveillance Videos”](https://www.crcv.ucf.edu/projects/real-world/). I kept the original training/inference pipeline and replaced only the testing function and evaluation logic (metrics calculation and reporting).

# How to prepare the directory

[Here](https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch?tab=readme-ov-file) is the initial repository. To run it:

1) Download the full code from [Google Drive](https://drive.google.com/file/d/1xYsBiCSmXjE0BwoiH_Bcm4AWaA1pESHF/view?usp=sharing) in an I3D_approach folder;

2) Download the [I3D features](https://drive.google.com/file/d/18nlV4YjPM93o-SdnPQrvauMN_v-oizmZ/view?usp=sharing);

3) Replace UCF-crime/all_flows and UCF-crime/all_rgbs with the corresponding folders from the downloaded I3D features (in Google Drive, there are empty folders UCF-crime/all_flows, UCF-crime/all_rgbs);

4) In I3D_approach directory, run test.py --FFC on GPU. This returns AUC on the test set.

# Original metrics calculation and what is wrong with it

There are 140 anomalous test videos, 150 normal test videos. 

Metrics (AUC) are calculated in test.py, function test_abnormal:

- For 140 test pairs (anomalous video, normal video):

        - Predict which frames are anomalous (for test videos, we have anomaly timecodes in terms of frames; predictions are in terms of frame blocks): score_list, score_list2;
        - Calculate auc += metrics.auc(fpr, tpr) -> a longer video in the pair contributes more than a shorter one (their frame scores are concatenated).

- Problem: in the end, auc = auc / 140;

- So authors try to estimate AUC = P(score(x_pos) > score(x_neg)), where x is a frame. But they compare video pairs, not frames (and only 140 pairs; and each video has different number of frames => different contribution)!

My fix: aggregate all the scores first, compare per block so each video has the same contribution in the metrics.

The correct AUC is 83.97.

# My fix 

Goal: fix AUC calculation, add Accuracy, Precision, Recall for all test set and per anomaly category.

- I’ve changed dataset.py so the code doesn’t fail if some files are missing, but prints warnings;

- I’ve changed test.py to calculate correct metrics:

    - Added a function collate_fn to process the missing files from the dataset and therefore changed DataLoader and NormalLoader;

    - Changed the function test_abnormal (testing function):
    
    - Rewrote AUC calculation;
    
    - Added Accuracy, Precision, and Recall calculation for the different thresholds (0.5, 0.7, 0.9. 0.95);

- Metric calculation is now possible per anomaly category and for all test set;

- Metric calculation is now possible on frame-level and block-level (block-level: important to guess.

## Additional functionality (extended metrics and evaluation modes)

I further extended `test.py` and tooling to support robust, reproducible evaluation beyond AUC.

### CLI flags and model control
- `--ckpt PATH`: select checkpoint explicitly (must match architecture and input size)
- `--modality {TWO,RGB,FLOW}` and `--input_dim {2048,1024}`: control feature modality and first layer size
- `--FFC`/`-r`: switch to FFC head (`Learner2`)
- `--eval`: choose evaluation modes (can be combined)
  - `auc` – frame/block ROC-AUC (Overall and per-category)
  - `auc_video` – video-level ROC-AUC (max over segments)
  - `pr_frame`, `pr_block`, `pr_video` – Precision/Recall/F1 at thresholds
  - `fpr_frame`, `fpr_block`, `fpr_video` – TPR/FPR by thresholds (Overall and per-category)
- `--thresholds t1 t2 ...`: thresholds used for PR metrics and FPR/TPR tables
- Debug printing: `--print_scores` and `--print_limit` to quickly inspect scores

### Balanced per-category PR metrics
- For per-category PR metrics (frame/block/video), we evaluate on a balanced set: all positives from the category plus an equal number of sampled normal negatives. This makes categories more comparable and avoids domination by the large normal pool.
- Overall rows remain unbalanced (use the full dataset), reflecting the “as-is” deployment distribution.

### Video-level metrics
- Added video-level scoring by taking the max over the 32 segment scores per video.
- AUC and PR modes are provided at video level; FPR/TPR tables also available.

### Cross-fitting (CV) threshold selection
- `--cv_k K`, `--cv_seed`: K-fold cross-fitting for PR modes (frame/block/video). On each fold, a threshold maximizing F1 is selected on K−1 folds and applied to the held-out fold.
- Outputs:
  - `{mode}_cv_global.csv`: per-category medians with a single global threshold (median across folds); per-category metrics are computed on balanced validation sets (category + same number of normals).
  - `{mode}_cv_percat.csv`: per-category medians with per-category thresholds (if enabled) or the global one otherwise.
- Per-category thresholds (optional):
  - `--cv_per_category` to enable; `--cv_per_category_min_n` to set minimal number of training positive videos required to fit a per-category threshold.
  - Both global and per-category CSVs are saved when enabled.

### FPR/TPR tables by thresholds
- `fpr_*` modes write `category, threshold, tpr, fpr` for frame/block/video using all negatives (unbalanced), which is the usual definition for TPR/FPR.

### Reproducibility and robustness
- All random sampling (balancing, CV folds) is seeded via `--cv_seed`
- Safe handling of missing feature files (warnings, robust `collate_fn`)

## Usage examples (how to call test.py)

### 1) Quick AUC run (frame + block + video)
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval auc auc_video
```

### 2) PR metrics (Precision/Recall/F1) at thresholds
- Frame level:
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval pr_frame --thresholds 0.1 0.2 0.3 0.5 0.7 0.9
```
- Block (event) level:
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval pr_block --thresholds 0.1 0.2 0.3 0.5 0.7 0.9
```
- Video level:
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval pr_video --thresholds 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3 1e-2 0.1 0.2 0.3 0.5 0.7 0.9
```
Results will be written to `results/pr_*.csv` (Overall = unbalanced; per-category = balanced: category + same number of normals).

### 3) FPR/TPR tables at thresholds
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval fpr_video --thresholds 0.1 0.2 0.3 0.5 0.7 0.9
```
Results: `results/fpr_video.csv` (Overall and per-category; unbalanced, uses all normals).

### 4) Cross-fitting (CV) for threshold selection (PR modes)
- Global threshold (medians across folds), video level:
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval pr_video --thresholds 0.1 0.2 0.3 0.5 0.7 0.9 \
  --cv_k 5 --cv_seed 42
```
Outputs:
- `results/pr_video_cv_global.csv` — Overall (unbalanced) and per-category (balanced) medians with a single global threshold.

- Per-category thresholds where enough data (and global for the rest):
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval pr_video --thresholds 0.1 0.2 0.3 0.5 0.7 0.9 \
  --cv_k 5 --cv_seed 42 --cv_per_category --cv_per_category_min_n 10
```
Outputs:
- `results/pr_video_cv_global.csv` — medians with global threshold
- `results/pr_video_cv_percat.csv` — medians with per-category thresholds where applicable (otherwise global)

### 5) Combine multiple evaluation modes
```bash
python test.py --FFC --modality TWO --input_dim 2048 --ckpt ./checkpoint/ffc_85_45.pth \
  --eval auc auc_video pr_video fpr_video \
  --thresholds 0.1 0.2 0.3 0.5 0.7 0.9
```

