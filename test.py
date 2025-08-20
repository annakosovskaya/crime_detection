from torch.utils.data import DataLoader
from learner import Learner
from loss import *
from dataset import *
import os
import argparse
from FFC import *
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, precision_recall_curve, f1_score
import torch
import random
import csv

# --- For reproducibility ---
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch MIL Training')
# not used in testing: learning rate / weight decay
# parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
# parser.add_argument('--w', default=0.0010000000474974513, type=float, help='weight_decay')

# --modality TWO: RGB + Flow -> --input_dim 2048
# --modality RGB/FLOW -> --input_dim 1024
parser.add_argument('--modality', default='TWO', type=str, choices=['TWO', 'RGB', 'FLOW'], help='feature modality to use')
parser.add_argument('--input_dim', default=2048, type=int, help='model input dimension (2048 for TWO, 1024 for RGB/FLOW)')

# Model architecture selector: FFC (Learner2) vs base (Learner)
parser.add_argument('--FFC', '-r', action='store_true', help='use FFC (Learner2) architecture')
parser.add_argument('--drop', default=0.6, type=float, help='dropout rate (has no effect in eval mode)')

# Checkpoint path (must match architecture and input_dim)
parser.add_argument('--ckpt', type=str, default='./checkpoint/ffc_85_45.pth', help='path to model checkpoint .pth')

# Evaluation modes: can pass one or several (e.g., --eval roc block)
parser.add_argument('--eval', nargs='+', choices=['auc', 'auc_video', 'pr_block', 'pr_frame', 'pr_video', 'fpr_block', 'fpr_frame', 'fpr_video'], default=['auc', 'auc_video', 'pr_block', 'pr_frame', 'pr_video'], help='evaluation modes to run (auc computes AUCs; pr_* compute F1/Precision/Recall; fpr_* dumps TPR/FPR by thresholds)')

# Thresholds for precision/recall/f1 metrics (used in block/frame modes)
parser.add_argument('--thresholds', nargs='+', type=float, default=[0.5, 0.7, 0.9, 0.95], help='thresholds for PR/F1 evaluation')
parser.add_argument('--cv_k', type=int, default=1, help='K-fold cross-fitting for threshold selection (K>1 enables per-fold evaluation)')
parser.add_argument('--cv_seed', type=int, default=42, help='random seed for CV splits')
parser.add_argument('--cv_per_category_min_n', type=int, default=5, help='if >0, use per-category threshold in CV when a category has more than this many training videos (positives)')
parser.add_argument('--cv_per_category', action='store_true', help='enable per-category threshold selection in CV (uses --cv_per_category_min_n as minimum positives)')

# Debug printing of score arrays
parser.add_argument(
    '--print_scores',
    nargs='+',
    choices=['segment', 'frame', 'normal_segment', 'event', 'video', 'block', 'all'],
    default=[],
    help='print selected score arrays for debugging (limited by --print_limit per split)'
)
parser.add_argument('--print_limit', type=int, default=3, help='max number of videos per split to print detailed scores for')

args = parser.parse_args()

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None, None
    return torch.utils.data.dataloader.default_collate(batch)

def to_python_int(value):
    """Convert potentially nested/torch value to a plain Python int."""
    try:
        # Unwrap single-item containers
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0]
        if torch.is_tensor(value):
            return int(value.item())
        return int(value)
    except Exception:
        # Fallback: try best-effort conversion
        return int(value)

def should_print(kind: str) -> bool:
    return ('all' in args.print_scores) or (kind in args.print_scores)

def summarize_scores(label: str, arr, max_vals: int = 10):
    arr_np = np.array(arr, dtype=float).ravel()
    if arr_np.size == 0:
        print(f"[{label}] empty")
        return
    head = arr_np[:max_vals].round(6).tolist()
    tail = arr_np[-max_vals:].round(6).tolist() if arr_np.size > max_vals else []
    summary = f"[{label}] len={arr_np.size}, min={arr_np.min():.6f}, max={arr_np.max():.6f}, mean={arr_np.mean():.6f}, head={head}"
    if tail:
        summary += f", tail={tail}"
    print(summary)

# ====================================================================
# 1. Create DataLoaders
# ====================================================================
anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=args.modality)
normal_test_dataset = Normal_Loader(is_train=0, modality=args.modality)

anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if args.FFC:
    model = Learner2(input_dim=args.input_dim, drop_p=args.drop).to(device)
else:
    model = Learner(input_dim=args.input_dim, drop_p=args.drop).to(device)

checkpoint = torch.load(args.ckpt, map_location=device)
model.load_state_dict(checkpoint['net'])

def print_and_log_metrics(writer, eval_name, category, labels, scores, threshold):
    """Calculates, prints, and logs metrics for a given threshold."""
    preds = [1 if s >= threshold else 0 for s in scores]
    
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    
    print(f"[{eval_name}][{category}] Threshold: {threshold:.2f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    
    # Log to CSV
    writer.writerow([category, threshold, f1, prec, rec])


def test_abnormal():
    model.eval()

    # --- Inference and accumulation of scores/labels ---
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    # Accumulators
    all_frame_scores = []
    all_frame_labels = []
    all_normal_frame_scores = []
    all_normal_frame_labels = []
    event_scores = []  # max score inside each anomalous event
    normal_segment_scores = []  # 32 segment scores for each normal video
    category_metrics = {}  # category -> {event_scores, frame_scores, frame_labels, video_pos_scores}
    video_pos_scores = []  # per anomalous video max over segments
    video_neg_scores = []  # per normal video max over segments

    # Per-video storages for CV
    anomalous_videos = []  # list of dicts: {category, frame_scores, frame_labels, event_scores, video_score}
    normal_videos = []     # list of dicts: {frame_scores, segment_scores, video_score}

    with torch.no_grad():
        anom_vid_count = 0
        normal_vid_count = 0
        # ===== Inference on ANOMALOUS videos =====
        for data in anomaly_test_loader:
            if data is None:
                continue
            inputs, gts, frames, category = data
            category = category[0] if isinstance(category, tuple) else category
            anom_vid_count += 1

            if category not in category_metrics:
                category_metrics[category] = {'event_scores': [], 'frame_scores': [], 'frame_labels': []}

            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            segment_scores = model(inputs).cpu().numpy()
            if should_print('segment') and anom_vid_count <= args.print_limit:
                summarize_scores(f"segment_scores anomalous#{anom_vid_count} category={category}", segment_scores)

            frames_len = to_python_int(frames)
            frame_scores = np.zeros(frames_len)
            step = np.round(np.linspace(0, frames_len // 16, 33)).astype(int)
            for j in range(32):
                frame_scores[step[j]*16: step[j+1]*16] = segment_scores[j]
            if should_print('frame') and anom_vid_count <= args.print_limit:
                summarize_scores(f"frame_scores anomalous#{anom_vid_count} category={category}", frame_scores)

            # video-level score (max over segments)
            video_max_score = float(np.max(segment_scores))
            video_pos_scores.append(video_max_score)
            if should_print('video') and anom_vid_count <= args.print_limit:
                print(f"[video] anomalous#{anom_vid_count} category={category} max_score={video_max_score:.6f}")

            gt_list = np.zeros(frames_len)
            this_event_scores = []
            for k in range(len(gts) // 2):
                s, e = gts[k * 2], min(gts[k * 2 + 1], frames_len)
                if s < e:
                    gt_list[s - 1:e] = 1
                    max_score_in_event = np.max(frame_scores[s - 1:e])
                    event_scores.append(max_score_in_event)
                    category_metrics[category]['event_scores'].append(max_score_in_event)
                    this_event_scores.append(max_score_in_event)
            if should_print('event') and anom_vid_count <= args.print_limit and this_event_scores:
                summarize_scores(f"event_scores anomalous#{anom_vid_count} category={category}", this_event_scores)

            all_frame_scores.extend(frame_scores)
            all_frame_labels.extend(gt_list)
            category_metrics[category]['frame_scores'].extend(frame_scores)
            category_metrics[category]['frame_labels'].extend(gt_list)
            # init and store per-category video-level score for anomalous videos
            if 'video_pos_scores' not in category_metrics[category]:
                category_metrics[category]['video_pos_scores'] = []
            category_metrics[category]['video_pos_scores'].append(video_max_score)

            # store anomalous video for CV
            anomalous_videos.append({
                'category': category,
                'frame_scores': frame_scores.astype(float),
                'frame_labels': gt_list.astype(int),
                'event_scores': np.array(this_event_scores, dtype=float),
                'video_score': video_max_score,
            })

        # ===== Inference on NORMAL videos =====
        for data2 in normal_test_loader:
            if data2 is None:
                continue
            inputs2, gts2, frames2 = data2
            normal_vid_count += 1
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device)
            score2 = model(inputs2).cpu().numpy()
            seg_flat = score2.flatten().astype(float)
            normal_segment_scores.extend(seg_flat)
            if should_print('normal_segment') and normal_vid_count <= args.print_limit:
                summarize_scores(f"segment_scores normal#{normal_vid_count}", seg_flat)

            # video-level score (max over segments) for normal videos
            video_neg_scores.append(float(np.max(seg_flat)))
            if should_print('video') and normal_vid_count <= args.print_limit:
                print(f"[video] normal#{normal_vid_count} max_score={float(np.max(seg_flat)):.6f}")

            # Frame-level scores for normal videos
            frames2_len = to_python_int(frames2)
            normal_scores = np.zeros(frames2_len)
            step2 = np.round(np.linspace(0, frames2_len // 16, 33)).astype(int)
            for kk in range(32):
                normal_scores[step2[kk] * 16: step2[kk + 1] * 16] = seg_flat[kk]
            all_normal_frame_scores.extend(normal_scores)
            all_normal_frame_labels.extend(np.zeros_like(normal_scores))
            if should_print('frame') and normal_vid_count <= args.print_limit:
                summarize_scores(f"frame_scores normal#{normal_vid_count}", normal_scores)

            # store normal video for CV
            normal_videos.append({
                'frame_scores': normal_scores.astype(float),
                'segment_scores': seg_flat,
                'video_score': float(np.max(seg_flat)),
            })

    # Prepare combined arrays
    final_frame_scores = np.concatenate((all_frame_scores, all_normal_frame_scores))
    final_frame_labels = np.concatenate((all_frame_labels, all_normal_frame_labels))
    block_labels = np.concatenate((np.ones(len(event_scores)), np.zeros(len(normal_segment_scores))))
    block_scores = np.concatenate((event_scores, normal_segment_scores))
    video_labels = np.concatenate((np.ones(len(video_pos_scores)), np.zeros(len(video_neg_scores))))
    video_scores = np.concatenate((video_pos_scores, video_neg_scores))

    if should_print('block'):
        summarize_scores("block_scores (overall)", block_scores)
    if should_print('frame'):
        summarize_scores("frame_scores (overall)", final_frame_scores)
    if should_print('video'):
        summarize_scores("video_scores (overall)", video_scores)

    # =========================
    # Threshold selection via cross-fitting (per-fold)
    # =========================
    def compute_pr_metrics(y_true, y_score, thr):
        preds = (np.asarray(y_score) >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        return f1, prec, rec

    def select_best_threshold(y_true, y_score, thr_list):
        best_f1 = -1.0
        best_thr = None
        best_tuple = (0.0, 0.0, 0.0)
        for thr in thr_list:
            f1, prec, rec = compute_pr_metrics(y_true, y_score, thr)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_tuple = (f1, prec, rec)
        return best_thr, best_tuple

    def run_cv_for_mode(mode_name, pos_label='anomaly'):
        # Build fold indices for anomalous and normal videos
        k = max(1, int(args.cv_k))
        if k <= 1:
            return
        rng = np.random.RandomState(args.cv_seed)
        an_idx = np.arange(len(anomalous_videos))
        no_idx = np.arange(len(normal_videos))
        rng.shuffle(an_idx)
        rng.shuffle(no_idx)
        an_folds = np.array_split(an_idx, k)
        no_folds = np.array_split(no_idx, k)

        # Accumulators for per-fold results to compute medians later
        overall_vals = []          # list of tuples (thr_global, f1, prec, rec) for Overall using global threshold
        cat_vals_global = {}       # category -> list of tuples (thr_global, f1, prec, rec)
        cat_vals_percat = {}       # category -> list of tuples (thr_percat, f1, prec, rec) only when per-category threshold used

        for fold_id in range(k):
                rng_cv = np.random.RandomState(args.cv_seed + fold_id)
                val_an_idx = set(an_folds[fold_id].tolist())
                val_no_idx = set(no_folds[fold_id].tolist())
                train_an_idx = [i for arr in an_folds if not np.array_equal(arr, an_folds[fold_id]) for i in arr.tolist()]
                train_no_idx = [i for arr in no_folds if not np.array_equal(arr, no_folds[fold_id]) for i in arr.tolist()]

                # Assemble train/val arrays per mode
                if mode_name in ('pr_video', 'pr_video_normal'):
                    train_scores = [anomalous_videos[i]['video_score'] for i in train_an_idx] + [normal_videos[i]['video_score'] for i in train_no_idx]
                    train_labels = [1] * len(train_an_idx) + [0] * len(train_no_idx)
                    val_scores = [anomalous_videos[i]['video_score'] for i in val_an_idx] + [normal_videos[i]['video_score'] for i in val_no_idx]
                    val_labels = [1] * len(val_an_idx) + [0] * len(val_no_idx)
                elif mode_name in ('pr_block', 'pr_block_normal'):
                    train_pos_blocks = np.concatenate([anomalous_videos[i]['event_scores'] for i in train_an_idx if anomalous_videos[i]['event_scores'].size > 0]) if train_an_idx else np.array([])
                    train_neg_blocks = np.concatenate([normal_videos[i]['segment_scores'] for i in train_no_idx]) if train_no_idx else np.array([])
                    val_pos_blocks = np.concatenate([anomalous_videos[i]['event_scores'] for i in val_an_idx if anomalous_videos[i]['event_scores'].size > 0]) if val_an_idx else np.array([])
                    val_neg_blocks = np.concatenate([normal_videos[i]['segment_scores'] for i in val_no_idx]) if val_no_idx else np.array([])
                    train_scores = np.concatenate((train_pos_blocks, train_neg_blocks))
                    train_labels = np.concatenate((np.ones(len(train_pos_blocks)), np.zeros(len(train_neg_blocks))))
                    val_scores = np.concatenate((val_pos_blocks, val_neg_blocks))
                    val_labels = np.concatenate((np.ones(len(val_pos_blocks)), np.zeros(len(val_neg_blocks))))
                elif mode_name in ('pr_frame', 'pr_frame_normal'):
                    train_pos_frames_scores = np.concatenate([anomalous_videos[i]['frame_scores'] for i in train_an_idx]) if train_an_idx else np.array([])
                    train_pos_frames_labels = np.concatenate([anomalous_videos[i]['frame_labels'] for i in train_an_idx]) if train_an_idx else np.array([])
                    train_neg_frames_scores = np.concatenate([normal_videos[i]['frame_scores'] for i in train_no_idx]) if train_no_idx else np.array([])
                    train_neg_frames_labels = np.zeros_like(train_neg_frames_scores)
                    train_scores = np.concatenate((train_pos_frames_scores, train_neg_frames_scores))
                    train_labels = np.concatenate((train_pos_frames_labels, train_neg_frames_labels))

                    val_pos_frames_scores = np.concatenate([anomalous_videos[i]['frame_scores'] for i in val_an_idx]) if val_an_idx else np.array([])
                    val_pos_frames_labels = np.concatenate([anomalous_videos[i]['frame_labels'] for i in val_an_idx]) if val_an_idx else np.array([])
                    val_neg_frames_scores = np.concatenate([normal_videos[i]['frame_scores'] for i in val_no_idx]) if val_no_idx else np.array([])
                    val_neg_frames_labels = np.zeros_like(val_neg_frames_scores)
                    val_scores = np.concatenate((val_pos_frames_scores, val_neg_frames_scores))
                    val_labels = np.concatenate((val_pos_frames_labels, val_neg_frames_labels))
                else:
                    continue

                # Invert labels for *_normal modes
                if mode_name.endswith('_normal'):
                    train_labels = 1 - np.asarray(train_labels)
                    val_labels = 1 - np.asarray(val_labels)
                else:
                    train_labels = np.asarray(train_labels)
                    val_labels = np.asarray(val_labels)

                # Select threshold on train
                best_thr, _ = select_best_threshold(train_labels, train_scores, args.thresholds)
                # Evaluate on val (overall)
                f1, prec, rec = compute_pr_metrics(val_labels, val_scores, best_thr)
                overall_vals.append((best_thr, f1, prec, rec))

                # Evaluate per-category on val anomalies vs val normals
                per_cat_min = max(0, int(args.cv_per_category_min_n))
                if mode_name in ('pr_video', 'pr_video_normal'):
                    val_normals_scores = [normal_videos[i]['video_score'] for i in val_no_idx]
                    train_normals_scores = [normal_videos[i]['video_score'] for i in train_no_idx]
                    for cat in sorted({anomalous_videos[i]['category'] for i in val_an_idx}):
                        # Global threshold metrics (always)
                        use_thr_global = best_thr
                        # Per-category threshold if enough train positives
                        train_cat_pos = [anomalous_videos[i]['video_score'] for i in train_an_idx if anomalous_videos[i]['category'] == cat]
                        use_thr_percat = None
                        if args.cv_per_category and per_cat_min > 0 and len(train_cat_pos) > per_cat_min:
                            tr_scores = np.concatenate((np.array(train_cat_pos, dtype=float), np.array(train_normals_scores, dtype=float)))
                            tr_labels = np.concatenate((np.ones(len(train_cat_pos)), np.zeros(len(train_normals_scores))))
                            use_thr_percat, _ = select_best_threshold(tr_labels, tr_scores, args.thresholds)
                        cat_pos = [anomalous_videos[i]['video_score'] for i in val_an_idx if anomalous_videos[i]['category'] == cat]
                        # Balanced negatives: sample as many normals as positives
                        neg_pool = np.array(val_normals_scores, dtype=float)
                        if len(cat_pos) == 0 or neg_pool.size == 0:
                            continue
                        replace = neg_pool.size < len(cat_pos)
                        sel_idx = rng_cv.choice(neg_pool.size, size=len(cat_pos), replace=replace)
                        sel_neg = neg_pool[sel_idx]
                        y = np.concatenate((np.ones(len(cat_pos)), np.zeros(len(sel_neg))))
                        s = np.concatenate((np.array(cat_pos, dtype=float), sel_neg))
                        f1g, precg, recg = compute_pr_metrics(y, s, use_thr_global)
                        cat_vals_global.setdefault(cat, []).append((use_thr_global, f1g, precg, recg))
                        if use_thr_percat is not None:
                            f1p, precp, recp = compute_pr_metrics(y, s, use_thr_percat)
                            cat_vals_percat.setdefault(cat, []).append((use_thr_percat, f1p, precp, recp))
                elif mode_name in ('pr_block', 'pr_block_normal'):
                    val_normals_blocks = np.concatenate([normal_videos[i]['segment_scores'] for i in val_no_idx]) if val_no_idx else np.array([])
                    train_normals_blocks = np.concatenate([normal_videos[i]['segment_scores'] for i in train_no_idx]) if train_no_idx else np.array([])
                    for cat in sorted({anomalous_videos[i]['category'] for i in val_an_idx}):
                        use_thr_global = best_thr
                        train_cat_blocks = np.concatenate([anomalous_videos[i]['event_scores'] for i in train_an_idx if anomalous_videos[i]['category'] == cat and anomalous_videos[i]['event_scores'].size > 0]) if train_an_idx else np.array([])
                        train_cat_vid_count = len([i for i in train_an_idx if anomalous_videos[i]['category'] == cat])
                        use_thr_percat = None
                        if args.cv_per_category and per_cat_min > 0 and train_cat_vid_count > per_cat_min and train_cat_blocks.size > 0:
                            tr_scores = np.concatenate((np.array(train_cat_blocks, dtype=float), np.array(train_normals_blocks, dtype=float)))
                            tr_labels = np.concatenate((np.ones(len(train_cat_blocks)), np.zeros(len(train_normals_blocks))))
                            use_thr_percat, _ = select_best_threshold(tr_labels, tr_scores, args.thresholds)
                        cat_blocks = np.concatenate([anomalous_videos[i]['event_scores'] for i in val_an_idx if anomalous_videos[i]['category'] == cat and anomalous_videos[i]['event_scores'].size > 0])
                        # Balanced negatives: sample as many normal segments as positive blocks
                        neg_pool = np.array(val_normals_blocks, dtype=float)
                        if cat_blocks.size == 0 or neg_pool.size == 0:
                            continue
                        replace = neg_pool.size < cat_blocks.size
                        sel_idx = rng_cv.choice(neg_pool.size, size=cat_blocks.size, replace=replace)
                        sel_neg = neg_pool[sel_idx]
                        y = np.concatenate((np.ones(cat_blocks.size), np.zeros(sel_neg.size)))
                        s = np.concatenate((np.array(cat_blocks, dtype=float), sel_neg))
                        f1g, precg, recg = compute_pr_metrics(y, s, use_thr_global)
                        cat_vals_global.setdefault(cat, []).append((use_thr_global, f1g, precg, recg))
                        if use_thr_percat is not None:
                            f1p, precp, recp = compute_pr_metrics(y, s, use_thr_percat)
                            cat_vals_percat.setdefault(cat, []).append((use_thr_percat, f1p, precp, recp))
                elif mode_name in ('pr_frame', 'pr_frame_normal'):
                    val_normals_frames = np.concatenate([normal_videos[i]['frame_scores'] for i in val_no_idx]) if val_no_idx else np.array([])
                    val_normals_labels = np.zeros_like(val_normals_frames)
                    train_normals_frames = np.concatenate([normal_videos[i]['frame_scores'] for i in train_no_idx]) if train_no_idx else np.array([])
                    train_normals_labels = np.zeros_like(train_normals_frames)
                    for cat in sorted({anomalous_videos[i]['category'] for i in val_an_idx}):
                        use_thr_global = best_thr
                        tr_cat_frames_scores = np.concatenate([anomalous_videos[i]['frame_scores'] for i in train_an_idx if anomalous_videos[i]['category'] == cat]) if train_an_idx else np.array([])
                        tr_cat_frames_labels = np.concatenate([anomalous_videos[i]['frame_labels'] for i in train_an_idx if anomalous_videos[i]['category'] == cat]) if train_an_idx else np.array([])
                        train_cat_vid_count = len([i for i in train_an_idx if anomalous_videos[i]['category'] == cat])
                        use_thr_percat = None
                        if args.cv_per_category and per_cat_min > 0 and train_cat_vid_count > per_cat_min and tr_cat_frames_scores.size > 0:
                            tr_scores = np.concatenate((tr_cat_frames_scores, train_normals_frames))
                            tr_labels = np.concatenate((tr_cat_frames_labels, train_normals_labels))
                            use_thr_percat, _ = select_best_threshold(tr_labels, tr_scores, args.thresholds)
                        cat_frames_scores = np.concatenate([anomalous_videos[i]['frame_scores'] for i in val_an_idx if anomalous_videos[i]['category'] == cat])
                        cat_frames_labels = np.concatenate([anomalous_videos[i]['frame_labels'] for i in val_an_idx if anomalous_videos[i]['category'] == cat])
                        # Balanced negatives: sample as many normal frames as category frames
                        neg_pool = np.array(val_normals_frames, dtype=float)
                        if cat_frames_scores.size == 0 or neg_pool.size == 0:
                            continue
                        replace = neg_pool.size < cat_frames_scores.size
                        sel_idx = rng_cv.choice(neg_pool.size, size=cat_frames_scores.size, replace=replace)
                        sel_neg_scores = neg_pool[sel_idx]
                        sel_neg_labels = np.zeros_like(sel_neg_scores, dtype=int)
                        y = np.concatenate((cat_frames_labels, sel_neg_labels))
                        s = np.concatenate((cat_frames_scores, sel_neg_scores))
                        f1g, precg, recg = compute_pr_metrics(y, s, use_thr_global)
                        cat_vals_global.setdefault(cat, []).append((use_thr_global, f1g, precg, recg))
                        if use_thr_percat is not None:
                            f1p, precp, recp = compute_pr_metrics(y, s, use_thr_percat)
                            cat_vals_percat.setdefault(cat, []).append((use_thr_percat, f1p, precp, recp))

        # After collecting all folds, write two CSVs:
        # 1) Global threshold metrics per category
        csv_global = os.path.join(results_dir, f"{mode_name}_cv_global.csv")
        with open(csv_global, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'threshold_median', 'f1_median', 'precision_median', 'recall_median'])

            overall_thr_med = None
            overall_f1_med = None
            overall_pr_med = None
            overall_rc_med = None
            if overall_vals:
                overall_thr_med = float(np.median([v[0] for v in overall_vals]))
                overall_f1_med = float(np.median([v[1] for v in overall_vals]))
                overall_pr_med = float(np.median([v[2] for v in overall_vals]))
                overall_rc_med = float(np.median([v[3] for v in overall_vals]))
                writer.writerow(['Overall', overall_thr_med, overall_f1_med, overall_pr_med, overall_rc_med])

            for cat in sorted(cat_vals_global.keys()):
                vals = cat_vals_global[cat]
                cat_f1_med = float(np.median([v[1] for v in vals]))
                cat_pr_med = float(np.median([v[2] for v in vals]))
                cat_rc_med = float(np.median([v[3] for v in vals]))
                thr_med = overall_thr_med if overall_thr_med is not None else float(np.median([v[0] for v in vals]))
                writer.writerow([cat, thr_med, cat_f1_med, cat_pr_med, cat_rc_med])

        print(f"CV median metrics (global) saved to {csv_global}")

        # 2) Per-category threshold metrics (include all categories; if no per-category threshold was used, fall back to global metrics)
        csv_percat = os.path.join(results_dir, f"{mode_name}_cv_percat.csv")
        with open(csv_percat, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'threshold_median', 'f1_median', 'precision_median', 'recall_median'])
            # Overall row for reference based on global thresholds
            if overall_vals:
                writer.writerow(['Overall', overall_thr_med, overall_f1_med, overall_pr_med, overall_rc_med])

            all_cats = sorted(set(cat_vals_global.keys()) | set(cat_vals_percat.keys()))
            for cat in all_cats:
                if cat in cat_vals_percat and len(cat_vals_percat[cat]) > 0:
                    vals = cat_vals_percat[cat]
                    thr_med = float(np.median([v[0] for v in vals]))
                else:
                    vals = cat_vals_global.get(cat, [])
                    # If no global vals (edge case), skip
                    if not vals:
                        continue
                    thr_med = overall_thr_med if overall_thr_med is not None else float(np.median([v[0] for v in vals]))
                cat_f1_med = float(np.median([v[1] for v in vals]))
                cat_pr_med = float(np.median([v[2] for v in vals]))
                cat_rc_med = float(np.median([v[3] for v in vals]))
                writer.writerow([cat, thr_med, cat_f1_med, cat_pr_med, cat_rc_med])

        print(f"CV median metrics (per-category) saved to {csv_percat}")

    # Trigger CV runs for requested modes
    if args.cv_k and args.cv_k > 1:
        if 'pr_video' in args.eval:
            run_cv_for_mode('pr_video')
        if 'pr_block' in args.eval:
            run_cv_for_mode('pr_block')
        if 'pr_frame' in args.eval:
            run_cv_for_mode('pr_frame')
        if 'pr_video_norm' in args.eval:
            run_cv_for_mode('pr_video_normal')
        if 'pr_block_norm' in args.eval:
            run_cv_for_mode('pr_block_normal')
        if 'pr_frame_norm' in args.eval:
            run_cv_for_mode('pr_frame_normal')

    # =========================
    # ROC-AUC mode
    # =========================
    if 'auc' in args.eval:
        roc_csv = os.path.join(results_dir, 'auc.csv')
        with open(roc_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['evaluation_level', 'category', 'roc_auc'])

            # Overall frame-level AUC
            frame_fpr, frame_tpr, _ = roc_curve(final_frame_labels, final_frame_scores, pos_label=1)
            frame_auc = auc(frame_fpr, frame_tpr)
            print(f"Frame-level ROC AUC: {frame_auc:.4f}")
            writer.writerow(['frame', 'Overall', frame_auc])

            # Overall block-level AUC
            block_fpr, block_tpr, _ = roc_curve(block_labels, block_scores, pos_label=1)
            block_auc = auc(block_fpr, block_tpr)
            print(f"Block-level ROC AUC: {block_auc:.4f}")
            writer.writerow(['block', 'Overall', block_auc])

            # Per-category AUCs
            for category, data in sorted(category_metrics.items()):
                # Block-level per-category
                cat_event_scores = data['event_scores']
                if cat_event_scores:
                    cat_block_labels = np.concatenate((np.ones(len(cat_event_scores)), np.zeros(len(normal_segment_scores))))
                    cat_block_scores = np.concatenate((cat_event_scores, normal_segment_scores))
                    cat_block_fpr, cat_block_tpr, _ = roc_curve(cat_block_labels, cat_block_scores, pos_label=1)
                    cat_block_auc = auc(cat_block_fpr, cat_block_tpr)
                    writer.writerow(['block', category, cat_block_auc])

                # Frame-level per-category
                cat_frame_labels = np.concatenate((data['frame_labels'], all_normal_frame_labels))
                cat_frame_scores = np.concatenate((data['frame_scores'], all_normal_frame_scores))
                cat_frame_fpr, cat_frame_tpr, _ = roc_curve(cat_frame_labels, cat_frame_scores, pos_label=1)
                cat_frame_auc = auc(cat_frame_fpr, cat_frame_tpr)
                writer.writerow(['frame', category, cat_frame_auc])

        print(f"ROC-AUC saved to {roc_csv}")

    # =========================
    # Video-level ROC-AUC mode
    # =========================
    if 'auc_video' in args.eval:
        auc_video_csv = os.path.join(results_dir, 'auc_video.csv')
        with open(auc_video_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['evaluation_level', 'category', 'roc_auc'])

            # Overall video-level AUC
            v_fpr, v_tpr, _ = roc_curve(video_labels, video_scores, pos_label=1)
            v_auc = auc(v_fpr, v_tpr)
            print(f"Video-level ROC AUC: {v_auc:.4f}")
            writer.writerow(['video', 'Overall', v_auc])

            # Per-category using positives from the category vs all normal videos
            for category, data in sorted(category_metrics.items()):
                cat_pos_videos = data.get('video_pos_scores', [])
                if not cat_pos_videos:
                    continue
                cat_v_labels = np.concatenate((np.ones(len(cat_pos_videos)), np.zeros(len(video_neg_scores))))
                cat_v_scores = np.concatenate((np.array(cat_pos_videos, dtype=float), np.array(video_neg_scores, dtype=float)))
                cat_v_fpr, cat_v_tpr, _ = roc_curve(cat_v_labels, cat_v_scores, pos_label=1)
                cat_v_auc = auc(cat_v_fpr, cat_v_tpr)
                writer.writerow(['video', category, cat_v_auc])

        print(f"Video-level ROC-AUC saved to {auc_video_csv}")

    # =========================
    # Block-level PR/F1 mode
    # =========================
    if 'pr_block' in args.eval:
        block_csv = os.path.join(results_dir, 'pr_block.csv')
        with open(block_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'threshold', 'f1', 'precision', 'recall'])

            for t in args.thresholds:
                print_and_log_metrics(writer, 'block', 'Overall', block_labels, block_scores, t)

            for category, data in sorted(category_metrics.items()):
                cat_event_scores = data['event_scores']
                if not cat_event_scores:
                    continue
                # Balanced normals: sample as many normal segments as positive blocks
                rng_eval = np.random.RandomState(args.cv_seed)
                normal_seg_np = np.array(normal_segment_scores, dtype=float)
                num_pos = len(cat_event_scores)
                if normal_seg_np.size == 0:
                    continue
                replace = normal_seg_np.size < num_pos
                sel_idx = rng_eval.choice(normal_seg_np.size, size=num_pos, replace=replace)
                neg_samples = normal_seg_np[sel_idx]
                cat_block_labels = np.concatenate((np.ones(num_pos), np.zeros(len(neg_samples))))
                cat_block_scores = np.concatenate((np.array(cat_event_scores, dtype=float), neg_samples))
                for t in args.thresholds:
                    print_and_log_metrics(writer, 'block', category, cat_block_labels, cat_block_scores, t)

        print(f"Block-level metrics saved to {block_csv}")

    # =========================
    # Frame-level PR/F1 mode
    # =========================
    if 'pr_frame' in args.eval:
        frame_csv = os.path.join(results_dir, 'pr_frame.csv')
        with open(frame_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'threshold', 'f1', 'precision', 'recall'])

            for t in args.thresholds:
                print_and_log_metrics(writer, 'frame', 'Overall', final_frame_labels, final_frame_scores, t)

            for category, data in sorted(category_metrics.items()):
                rng_eval = np.random.RandomState(args.cv_seed)
                cat_frame_labels = np.array(data['frame_labels'], dtype=int)
                cat_frame_scores = np.array(data['frame_scores'], dtype=float)
                neg_frames = np.array(all_normal_frame_scores, dtype=float)
                if neg_frames.size == 0 or cat_frame_scores.size == 0:
                    continue
                num_cat = cat_frame_scores.size
                replace = neg_frames.size < num_cat
                sel_idx = rng_eval.choice(neg_frames.size, size=num_cat, replace=replace)
                sel_neg_scores = neg_frames[sel_idx]
                sel_neg_labels = np.zeros_like(sel_neg_scores, dtype=int)
                combined_scores = np.concatenate((cat_frame_scores, sel_neg_scores))
                combined_labels = np.concatenate((cat_frame_labels, sel_neg_labels))
                for t in args.thresholds:
                    print_and_log_metrics(writer, 'frame', category, combined_labels, combined_scores, t)

        print(f"Frame-level metrics saved to {frame_csv}")

    # =========================
    # Video-level PR/F1 mode
    # =========================
    if 'pr_video' in args.eval:
        video_csv = os.path.join(results_dir, 'pr_video.csv')
        with open(video_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'threshold', 'f1', 'precision', 'recall'])

            for t in args.thresholds:
                print_and_log_metrics(writer, 'video', 'Overall', video_labels, video_scores, t)

            for category, data in sorted(category_metrics.items()):
                cat_pos_videos = data.get('video_pos_scores', [])
                if not cat_pos_videos:
                    continue
                rng_eval = np.random.RandomState(args.cv_seed)
                neg_v = np.array(video_neg_scores, dtype=float)
                num_pos = len(cat_pos_videos)
                if neg_v.size == 0:
                    continue
                replace = neg_v.size < num_pos
                sel_idx = rng_eval.choice(neg_v.size, size=num_pos, replace=replace)
                sel_neg = neg_v[sel_idx]
                cat_v_labels = np.concatenate((np.ones(num_pos), np.zeros(num_pos)))
                cat_v_scores = np.concatenate((np.array(cat_pos_videos, dtype=float), sel_neg))
                for t in args.thresholds:
                    print_and_log_metrics(writer, 'video', category, cat_v_labels, cat_v_scores, t)

        print(f"Video-level metrics saved to {video_csv}")

    # =========================
    # TPR/FPR tables for normals (per threshold)
    # =========================
    def dump_fpr_tpr(csv_path, y_true, y_score, thresholds, level_name, per_category=None):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'threshold', 'tpr', 'fpr'])
            for t in thresholds:
                preds = (np.asarray(y_score) >= t).astype(int)
                tp = int(((preds == 1) & (y_true == 1)).sum())
                fp = int(((preds == 1) & (y_true == 0)).sum())
                fn = int(((preds == 0) & (y_true == 1)).sum())
                tn = int(((preds == 0) & (y_true == 0)).sum())
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                writer.writerow(['Overall', t, tpr, fpr])
            if per_category is not None:
                for cat, (y, s) in per_category.items():
                    for t in thresholds:
                        preds = (np.asarray(s) >= t).astype(int)
                        tp = int(((preds == 1) & (y == 1)).sum())
                        fp = int(((preds == 1) & (y == 0)).sum())
                        fn = int(((preds == 0) & (y == 1)).sum())
                        tn = int(((preds == 0) & (y == 0)).sum())
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        writer.writerow([cat, t, tpr, fpr])

    if 'fpr_block' in args.eval:
        per_cat = {}
        for category, data in sorted(category_metrics.items()):
            cat_event_scores = data['event_scores']
            if not cat_event_scores:
                continue
            y = np.concatenate((np.ones(len(cat_event_scores)), np.zeros(len(normal_segment_scores))))
            s = np.concatenate((np.array(cat_event_scores, dtype=float), np.array(normal_segment_scores, dtype=float)))
            per_cat[category] = (y, s)
        dump_fpr_tpr(os.path.join(results_dir, 'fpr_block.csv'), block_labels, block_scores, args.thresholds, 'block', per_cat)

    if 'fpr_frame' in args.eval:
        per_cat = {}
        for category, data in sorted(category_metrics.items()):
            y = np.concatenate((np.array(data['frame_labels'], dtype=int), np.array(all_normal_frame_labels, dtype=int)))
            s = np.concatenate((np.array(data['frame_scores'], dtype=float), np.array(all_normal_frame_scores, dtype=float)))
            per_cat[category] = (y, s)
        dump_fpr_tpr(os.path.join(results_dir, 'fpr_frame.csv'), final_frame_labels, final_frame_scores, args.thresholds, 'frame', per_cat)

    if 'fpr_video' in args.eval:
        per_cat = {}
        for category, data in sorted(category_metrics.items()):
            pos = np.array(data.get('video_pos_scores', []), dtype=float)
            if pos.size == 0:
                continue
            y = np.concatenate((np.ones(len(pos)), np.zeros(len(video_neg_scores))))
            s = np.concatenate((pos, np.array(video_neg_scores, dtype=float)))
            per_cat[category] = (y, s)
        dump_fpr_tpr(os.path.join(results_dir, 'fpr_video.csv'), video_labels, video_scores, args.thresholds, 'video', per_cat)

    # =========================
    # PR/F1 with NORMAL as positive (useful to see performance on normals)
    # =========================
    # removed *_norm modes per user request

if __name__ == '__main__':
    test_abnormal()