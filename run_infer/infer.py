import os
import sys
import errno

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

import torch
from PIL import Image
from diffsynth import save_video,VideoData, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffusers.utils import export_to_video, load_video
from typing import List
import itertools
import imageio
import torch.distributed as dist
import argparse
import pandas as pd
import json
import hashlib
import fcntl
import ast
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm

import gc
import time
import random
from datetime import timedelta


def _cleanup_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available() and hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


PG_TIMEOUT_MINUTES = _get_env_int("THINKV2V_PG_TIMEOUT_MINUTES", 30)
FINAL_SYNC_TIMEOUT_MINUTES = _get_env_int("THINKV2V_FINAL_SYNC_TIMEOUT_MINUTES", PG_TIMEOUT_MINUTES)
FINAL_SYNC_POLL_SECONDS = _get_env_int("THINKV2V_FINAL_SYNC_POLL_SECONDS", 5)
EDITED_RESULT_COLUMN = "edited_result_path"
THINKING_TEXT_COLUMN = "thinking_text"
RESPONSE_TEXT_COLUMN = "response_text"


def _ensure_distributed_initialized() -> None:
    if not dist.is_initialized():
        world_size = _get_env_int("WORLD_SIZE", 1)
        backend = "gloo" if world_size == 1 else "nccl"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=PG_TIMEOUT_MINUTES),
        )


def _build_final_sync_dir(output_dir: str, output_csv_path: str) -> str:
    sync_root = output_dir or os.path.dirname(output_csv_path or "") or "."
    run_id = os.environ.get("TORCHELASTIC_RUN_ID") or "|".join([
        os.environ.get("MASTER_ADDR", ""),
        os.environ.get("MASTER_PORT", ""),
        os.environ.get("WORLD_SIZE", ""),
    ])
    sync_key = "|".join([
        os.path.abspath(sync_root),
        os.path.abspath(output_csv_path) if output_csv_path else "",
        str(run_id),
    ])
    sync_hash = hashlib.sha1(sync_key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(sync_root, ".finalize_sync", sync_hash)


def _rank_done_marker_path(sync_dir: str, rank: int) -> str:
    return os.path.join(sync_dir, f"rank_{rank}.done.json")


def _write_rank_done_marker(sync_dir: str, rank: int, assigned_samples: int) -> None:
    os.makedirs(sync_dir, exist_ok=True)
    marker_path = _rank_done_marker_path(sync_dir, rank)
    payload = {
        "rank": rank,
        "assigned_samples": assigned_samples,
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _wait_for_all_rank_done_markers(sync_dir: str, world_size: int, timeout_seconds: int, poll_seconds: int) -> list[int]:
    deadline = time.time() + max(timeout_seconds, 1)
    poll_seconds = max(poll_seconds, 1)
    missing_ranks = list(range(world_size))

    while True:
        missing_ranks = [rank for rank in range(world_size) if not os.path.exists(_rank_done_marker_path(sync_dir, rank))]
        if not missing_ranks:
            return []
        if time.time() >= deadline:
            return missing_ranks
        time.sleep(poll_seconds)


def _cleanup_rank_done_markers(sync_dir: str, world_size: int) -> None:
    for rank in range(world_size):
        marker_path = _rank_done_marker_path(sync_dir, rank)
        try:
            os.remove(marker_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: failed to remove sync marker {marker_path}. Error: {e}")
    try:
        os.rmdir(sync_dir)
    except OSError:
        pass


def _load_text_record(output_dir: str, video_path_str: str):
    """Load per-sample text record saved by Qwen3Thinking.

    Returns: (thinking_text, clean_text)
    """
    if not output_dir or not video_path_str:
        return "", ""

    record_dir = os.path.join(output_dir, "text_records")
    stem = os.path.splitext(os.path.basename(video_path_str))[0]
    h = hashlib.sha1(video_path_str.encode("utf-8")).hexdigest()[:10]
    record_path = os.path.join(record_dir, f"{stem}__{h}.json")
    if not os.path.exists(record_path):
        return "", ""

    try:
        with open(record_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return str(obj.get("thinking_text") or ""), str(obj.get("clean_text") or "")
    except Exception:
        return "", ""


def _build_result_columns(base_columns, include_text: bool) -> list[str]:
    excluded = {
        "absolute_video_path",
        "basename",
        "thinking_text",
        "clean_text",
        RESPONSE_TEXT_COLUMN,
        "edited_video",
        EDITED_RESULT_COLUMN,
    }
    cols = [c for c in base_columns if c not in excluded]
    if include_text:
        cols.extend([THINKING_TEXT_COLUMN, RESPONSE_TEXT_COLUMN])
    cols.append(EDITED_RESULT_COLUMN)
    return cols


def _attach_text_columns(record, thinking_text: str, response_text: str) -> None:
    record[THINKING_TEXT_COLUMN] = thinking_text
    record[RESPONSE_TEXT_COLUMN] = response_text


def _join_input_root_with_overlap(input_root_dir: str, video_path_str: str) -> str:
    input_root_abs = os.path.abspath(input_root_dir)
    rel_path = os.path.normpath(str(video_path_str or "").strip())
    if not rel_path or rel_path == ".":
        return input_root_abs

    root_parts = [p for p in os.path.normpath(input_root_abs).split(os.sep) if p]
    rel_parts = [p for p in rel_path.split(os.sep) if p and p != "."]

    max_overlap = min(len(root_parts), len(rel_parts))
    overlap = 0
    for size in range(max_overlap, 0, -1):
        if root_parts[-size:] == rel_parts[:size]:
            overlap = size
            break

    return os.path.abspath(os.path.join(input_root_abs, *rel_parts[overlap:]))


def _normalize_original_video_path(video_path_str: str, input_root_dir: str | None = None) -> str:
    value = str(video_path_str or "").strip()
    if not value:
        return ""

    normalized = os.path.normpath(value)
    if not input_root_dir:
        return os.path.abspath(normalized) if os.path.isabs(normalized) else normalized

    input_root_abs = os.path.abspath(input_root_dir)
    if os.path.isabs(normalized):
        abs_normalized = os.path.abspath(normalized)
        try:
            relative_to_root = os.path.relpath(abs_normalized, input_root_abs)
        except ValueError:
            return abs_normalized
        if not relative_to_root.startswith("..") and relative_to_root != os.pardir:
            return _join_input_root_with_overlap(input_root_abs, relative_to_root)
        return abs_normalized

    return _join_input_root_with_overlap(input_root_abs, normalized)


def _append_result_to_csv_locked(csv_path: str, output_df: pd.DataFrame, columns: list[str]):
    """Append one record to a CSV file with an exclusive lock.

    NOTE: Some filesystems (certain network/remote mounts) do not support flock/fsync
    and may raise OSError(ENOSYS/EOPNOTSUPP). In that case callers should fall back
    to per-rank sharded CSVs and merge at the end.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a+", encoding="utf-8", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        header = f.tell() == 0
        output_df.to_csv(f, header=header, index=False, columns=columns)
        f.flush()
        # fsync may be unsupported on some FS; treat as best-effort.
        try:
            os.fsync(f.fileno())
        except OSError as e:
            if e.errno not in (errno.ENOSYS, errno.EOPNOTSUPP):
                raise
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _append_result_to_csv_unlocked(csv_path: str, output_df: pd.DataFrame, columns: list[str]):
    """Append without flock/fsync.

    This is safe only if *one writer per file* (e.g. per-rank sharded CSV).
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a+", encoding="utf-8", newline="") as f:
        f.seek(0, os.SEEK_END)
        header = f.tell() == 0
        output_df.to_csv(f, header=header, index=False, columns=columns)
        f.flush()


def _rank_csv_path(base_csv_path: str, rank: int) -> str:
    base, ext = os.path.splitext(base_csv_path)
    if ext.lower() != ".csv":
        return f"{base_csv_path}.rank{rank}.csv"
    return f"{base}.rank{rank}{ext}"


def _detect_shared_csv_lock_supported(csv_path: str) -> bool:
    """Return whether flock works on csv_path.

    We only test locking. If locking is unsupported, multi-process appends to a single
    CSV will be racy/corrupt; use per-rank sharded CSV instead.
    """
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    try:
        with open(csv_path, "a+", encoding="utf-8", newline="") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            finally:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        return True
    except OSError as e:
        if e.errno in (errno.ENOSYS, errno.EOPNOTSUPP):
            return False
        # Other errors are real problems (permission, path, etc.).
        raise


def _parse_sample_range(sample_range_arg: str | None):
    if sample_range_arg is None:
        return None
    try:
        values = ast.literal_eval(sample_range_arg)
    except Exception as e:
        raise ValueError(f"Invalid --sample_range format: {sample_range_arg}") from e

    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("--sample_range must be a list like [1, 10]")

    start, end = int(values[0]), int(values[1])
    if start <= 0 or end <= 0:
        raise ValueError("--sample_range indices must be positive integers")
    if start > end:
        raise ValueError("--sample_range start must be <= end")
    return start, end

def adjust_num_frames(num_frames):
    if num_frames % 4 == 1:
        return num_frames
    else:
        for i in range(num_frames, 0, -1):
            if i % 4 == 1:
                return i
        return 1

def get_video_info(video_path):
    reader = imageio.get_reader(video_path)
    try:
        num_frames = reader.count_frames()
        
        first_frame = reader.get_data(0)
        height, width = first_frame.shape[:2]
    finally:
        reader.close()
    return num_frames, width, height

def determine_dimensions(original_width, original_height, reso='720p'):
    is_landscape = original_width >= original_height
    if reso == '480p':
        if is_landscape:
            target_height, target_width = 480, 832
        else:
            target_height, target_width = 832, 480
    elif reso == '720p':
        if is_landscape:
            target_height, target_width = 704, 1280
        else:
            target_height, target_width = 1280, 704
    else:
        raise ValueError(f"Unsupported resolution parameter: {reso}, please use '480p' or '720p'.")
    return target_width, target_height

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input benchmark CSV file.")
    parser.add_argument("--input_root_dir", type=str, required=True, help="Root directory used to resolve original_video paths.")
    parser.add_argument("--max_num_samples", type=int, default=-1, help="Max number of samples to process. -1 for all.")
    parser.add_argument("--sample_range", type=str, default=None, help="1-based inclusive sample range, e.g. '[1, 10]'. Default: use all samples.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output videos")
    parser.add_argument(
        "--output_csv_path",
        type=str,
        default=None,
        help="Path to save the output CSV file with results. If not set, it will be saved under output_dir.",
    )
    parser.add_argument(
        "--wan_component_dir",
        type=str,
        required=True,
        help="Directory containing Wan shared components: google/umt5-xxl, models_t5_umt5-xxl-enc-bf16.pth, and Wan2.2_VAE.pth.",
    )
    parser.add_argument("--dit_checkpoint_path", type=str, required=True, help="Path to the ThinkV2V DiT checkpoint or checkpoint directory.")
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        required=True,
        help="Path to the Qwen3-VL-8B-Thinking model directory.",
    )
    parser.add_argument("--resolution", type=str, default="720p", choices=["480p", "720p"], help="Resolution for inference (480p or 720p)")
    parser.add_argument(
        "--inference_time_thinking_scaling",
        action="store_true",
        help="Enable the fixed n=8 sequential-refinement best-of-n thinking path. Disabled by default.",
    )
    parser.add_argument(
        "--enable_text_save",
        action="store_true",
        help="Save thinking_text/clean_text into output CSV, and save one JSON per sample under output_dir/text_records/",
    )
    args = parser.parse_args()

    if not args.output_dir:
        raise ValueError("--output_dir is required")

    if not args.output_csv_path:
        args.output_csv_path = os.path.join(args.output_dir, "infer_results.csv")

    _ensure_distributed_initialized()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # Decide CSV write strategy.
    # - If the filesystem supports flock, all ranks can safely append to a single CSV.
    # - Otherwise, write one CSV per-rank (no lock) and merge at the end on rank0.
    csv_lock_supported = True
    if dist.get_rank() == 0:
        try:
            csv_lock_supported = _detect_shared_csv_lock_supported(args.output_csv_path)
        except Exception as e:
            print(f"[Rank 0] CRITICAL: Failed to probe CSV lock support for {args.output_csv_path}. Error: {e}")
            raise
    obj_list = [csv_lock_supported]
    dist.broadcast_object_list(obj_list, src=0)
    csv_lock_supported = bool(obj_list[0])
    if dist.get_rank() == 0:
        if csv_lock_supported:
            print(f"CSV write mode: shared CSV with flock ({args.output_csv_path})")
        else:
            print(f"CSV write mode: per-rank sharded CSVs (flock unsupported). Base CSV: {args.output_csv_path}")

    model_path = args.dit_checkpoint_path
    qwen_model_path = args.qwen_model_path
    sample_range = _parse_sample_range(args.sample_range)

    wan_component_dir = os.path.abspath(args.wan_component_dir)
    umt5_tokenizer_dir = os.path.join(wan_component_dir, "google", "umt5-xxl")
    umt5_encoder_path = os.path.join(wan_component_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    wan_vae_path = os.path.join(wan_component_dir, "Wan2.2_VAE.pth")

    processed_videos_by_path = set()
    processed_videos_by_filename = set()

    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Loading CSV from {args.csv_path}")
        print(f"Output CSV path: {args.output_csv_path}")

        if os.path.exists(args.output_dir):
            try:
                # Only treat existing output videos as completed samples.
                # Avoid counting large numbers of JSON/text_records or sync directories.
                for entry in os.scandir(args.output_dir):
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if name.endswith(".mp4"):
                        processed_videos_by_filename.add(name)
                if processed_videos_by_filename:
                    print(
                        f"Found {len(processed_videos_by_filename)} existing mp4 files in output directory '{args.output_dir}'."
                    )
            except Exception as e:
                print(f"Warning: Could not scan output directory '{args.output_dir}'. Error: {e}")

        # Collect processed original_video paths from existing CSV(s).
        # Note: when flock is unsupported, some runs may leave infer_results.rank*.csv unmerged.
        csv_candidates = []
        if args.output_csv_path:
            if os.path.exists(args.output_csv_path):
                csv_candidates.append(args.output_csv_path)
            try:
                import glob

                csv_candidates.extend(sorted(glob.glob(os.path.splitext(args.output_csv_path)[0] + ".rank*.csv")))
            except Exception:
                pass

        processed_basenames_in_csv = set()

        for csv_path in csv_candidates:
            try:
                if not os.path.exists(csv_path) or os.path.getsize(csv_path) <= 0:
                    continue
                processed_df = pd.read_csv(csv_path)
                # We store absolute paths in the output CSV, so we compare with absolute paths.
                if "original_video" in processed_df.columns:
                    processed_videos_by_path.update([
                        _normalize_original_video_path(path, args.input_root_dir)
                        for path in processed_df["original_video"].tolist()
                        if path
                    ])

                # Track which output basenames are already recorded in CSV.
                result_column = EDITED_RESULT_COLUMN if EDITED_RESULT_COLUMN in processed_df.columns else "edited_video"
                if result_column in processed_df.columns:
                    processed_basenames_in_csv.update(
                        [os.path.basename(str(p)) for p in processed_df[result_column].tolist() if p]
                    )
                elif "original_video" in processed_df.columns:
                    processed_basenames_in_csv.update(
                        [os.path.basename(str(p)) for p in processed_df["original_video"].tolist() if p]
                    )
            except Exception as e:
                print(f"Warning: Could not read existing output CSV '{csv_path}'. Error: {e}")

        if processed_videos_by_path:
            print(f"Found {len(processed_videos_by_path)} entries in existing CSV(s) under '{args.output_dir}'.")

        # Backfill CSV rows for videos already written by interrupted runs.
        # This avoids rerunning inference when a video exists but its CSV row is missing.
        try:
            missing_basenames = sorted(processed_videos_by_filename - processed_basenames_in_csv)
            if missing_basenames:
                source_df = pd.read_csv(args.csv_path)
                # Map basename -> source row (first match).
                basename_to_row = {}
                if "original_video" in source_df.columns:
                    for _, r in source_df.iterrows():
                        b = os.path.basename(str(r.get("original_video") or ""))
                        if b and b not in basename_to_row:
                            basename_to_row[b] = r

                backfilled = 0
                for b in missing_basenames:
                    src_row = basename_to_row.get(b)
                    if src_row is None:
                        continue

                    original_rel = str(src_row.get("original_video") or "")
                    if not original_rel:
                        continue
                    original_abs = _normalize_original_video_path(original_rel, args.input_root_dir)
                    edited_abs = os.path.abspath(os.path.join(args.output_dir, b))
                    if not os.path.exists(edited_abs):
                        continue

                    thinking_text, clean_text = "", ""
                    if args.enable_text_save:
                        thinking_text, clean_text = _load_text_record(args.output_dir, original_abs)

                    result_record = src_row.copy()
                    result_record["original_video"] = original_abs
                    result_record[EDITED_RESULT_COLUMN] = edited_abs
                    if args.enable_text_save:
                        _attach_text_columns(result_record, thinking_text, clean_text)

                    output_df = pd.DataFrame([result_record])
                    cols = _build_result_columns(src_row.index, include_text=args.enable_text_save)

                    if args.output_csv_path:
                        if csv_lock_supported:
                            _append_result_to_csv_locked(args.output_csv_path, output_df, cols)
                        else:
                            shard_path = _rank_csv_path(args.output_csv_path, 0)
                            _append_result_to_csv_unlocked(shard_path, output_df, cols)
                        processed_videos_by_path.add(original_abs)
                        processed_basenames_in_csv.add(b)
                        backfilled += 1

                if backfilled:
                    print(f"Backfilled {backfilled} missing CSV rows from existing mp4 files.")
        except Exception as e:
            print(f"Warning: failed to backfill CSV from existing mp4s. Error: {e}")
    
    processed_list_path = list(processed_videos_by_path)
    processed_list_filename = list(processed_videos_by_filename)
    processed_state = [processed_list_path, processed_list_filename]
    dist.broadcast_object_list(processed_state, src=0)
    processed_list_path, processed_list_filename = processed_state
    processed_videos_by_path = set(processed_list_path)
    processed_videos_by_filename = set(processed_list_filename)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    df = pd.read_csv(args.csv_path)
    total_rows_in_csv = len(df)

    if sample_range is not None:
        start_idx, end_idx = sample_range
        if end_idx > total_rows_in_csv:
            print(f"Requested sample_range end {end_idx} exceeds total rows {total_rows_in_csv}; using {total_rows_in_csv} instead.")
            end_idx = total_rows_in_csv
        if start_idx > total_rows_in_csv:
            print(f"Requested sample_range start {start_idx} exceeds total rows {total_rows_in_csv}; no samples will be processed.")
            df = df.iloc[0:0].copy()
        else:
            df = df.iloc[start_idx - 1:end_idx].copy()

    if args.max_num_samples != -1:
        df = df.iloc[:args.max_num_samples]

    df = df.reset_index(drop=True)
        
    df['absolute_video_path'] = df['original_video'].apply(
        lambda x: _normalize_original_video_path(x, args.input_root_dir)
    )
    df['basename'] = df['original_video'].apply(os.path.basename)
    
    original_total = len(df)
    remaining_df = df[
        ~df['absolute_video_path'].isin(processed_videos_by_path) & 
        ~df['basename'].isin(processed_videos_by_filename)
    ].copy()
    remaining_df = remaining_df.reset_index(drop=True)
    if dist.get_rank() == 0:
        print(f"Total videos remaining in this run: {len(remaining_df)} (out of {original_total})")

    df = remaining_df.iloc[rank::world_size].copy()
    print(f"[Rank {rank}] Assigned {len(df)} samples.")

    # Load the heavy pipeline only if this rank will actually run inference.
    # This makes re-runs (where most samples are already finished) start much faster.
    pipe = None
    if len(df) > 0:
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            use_usp=False,
            tokenizer_config=ModelConfig(path=umt5_tokenizer_dir),
            model_configs=[
                ModelConfig(path=umt5_encoder_path),
                ModelConfig(path=f"{model_path}"),
                ModelConfig(path=wan_vae_path),
                ModelConfig(path=f"{qwen_model_path}"),
            ],
        )
        pipe.enable_vram_management()
        if dist.get_rank() == 0:
            print(f"Using Wan component dir: {wan_component_dir}")
            print(f"Loading checkpoint: {model_path}")
            print(f"Using Qwen model: {qwen_model_path}")
            qwen_model_name = type(pipe.qwen25vl).__name__ if pipe.qwen25vl is not None else "None"
            connector_name = type(pipe.qwen25vl_connector).__name__ if pipe.qwen25vl_connector is not None else "None"
            connector_in_features = None
            if (
                pipe.qwen25vl_connector is not None
                and hasattr(pipe.qwen25vl_connector, "fc")
                and len(pipe.qwen25vl_connector.fc) > 0
            ):
                connector_in_features = getattr(pipe.qwen25vl_connector.fc[0], "in_features", None)
            print(f"Loaded Qwen model class: {qwen_model_name}")
            print(f"Loaded connector class: {connector_name}")
            print(f"Connector input dim: {connector_in_features}")
    sync_dir = _build_final_sync_dir(args.output_dir, args.output_csv_path)


    iterator = df.iterrows()
    if dist.get_rank() == 0:
        iterator = tqdm(list(iterator), total=len(df), desc="Processing Videos")
    
    for idx, row in iterator:
        prompt = row['prompt']
        video_path = row['absolute_video_path']
        
        if dist.get_rank() == 0:
            print(f"\nProcessing video {video_path}")

        video = None
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")

            num_frames_total, original_width, original_height = get_video_info(video_path)
            num_frames = adjust_num_frames(num_frames_total)
            width, height = determine_dimensions(original_width, original_height, reso=args.resolution)

            if dist.get_rank() == 0:
                print(f"Video info: Total {num_frames_total} frames, processing {num_frames} frames. Original {original_width}x{original_height}, resizing to {width}x{height}")

            def convert_video(video: List[Image.Image]) -> List[Image.Image]:
                resized_video = video[:num_frames]
                resized_video = [frame.resize((width, height)) for frame in resized_video]
                return resized_video

            video = load_video(video_path, convert_method=convert_video)[:num_frames]

        except Exception as e:
            print(f"[Rank {rank}] Error loading {video_path}: {e}")

        if video is None:
            _cleanup_cuda_memory()
            continue

        try:
            print(f"[Rank {rank}] Inference started for {video_path}")
            print(
                f"[Rank {rank}] Qwen thinking config: enable_text_save={args.enable_text_save}, "
                f"mm_type=video, qwenvideo={video_path}",
                flush=True,
            )

            output_video = pipe(
                prompt=prompt,
                negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                seed=0, tiled=True,
                height=height, width=width,
                original_video=video,
                qwenvideo=video_path,
                mm_type="video",
                num_frames=num_frames,
                inference_time_thinking_scaling=args.inference_time_thinking_scaling,
                enable_text_save=args.enable_text_save,
                text_save_dir=args.output_dir,
            )
            
            filename = os.path.basename(video_path)
            save_path = os.path.join(args.output_dir, filename)
            save_video(output_video, save_path, fps=24, quality=5)
            absolute_save_path = os.path.abspath(save_path)
            print(f"[Rank {rank}] Saved to {absolute_save_path}")

            thinking_text, clean_text = "", ""
            text_source = "disabled"
            if args.enable_text_save:
                md = getattr(pipe, "qwen25vl_last_thinking_metadata", None)
                if isinstance(md, dict):
                    t = md.get("thinking_text")
                    c = md.get("clean_text")
                    if isinstance(t, (list, tuple)) and t:
                        thinking_text = str(t[0] or "")
                    elif t is not None:
                        thinking_text = str(t)
                    if isinstance(c, (list, tuple)) and c:
                        clean_text = str(c[0] or "")
                    elif c is not None:
                        clean_text = str(c)
                    if thinking_text or clean_text:
                        text_source = "pipeline_metadata"

                if not thinking_text and not clean_text:
                    thinking_text, clean_text = _load_text_record(args.output_dir, video_path)
                    if thinking_text or clean_text:
                        text_source = "text_record"
                    else:
                        text_source = "missing"

                print(
                    f"[Rank {rank}] Qwen text capture source={text_source}, "
                    f"thinking_len={len(thinking_text)}, clean_len={len(clean_text)}",
                    flush=True,
                )

            if args.output_csv_path:
                try:
                    result_record = row.copy()
                    result_record['original_video'] = _normalize_original_video_path(video_path, args.input_root_dir)
                    result_record[EDITED_RESULT_COLUMN] = absolute_save_path
                    if args.enable_text_save:
                        _attach_text_columns(result_record, thinking_text, clean_text)
                    output_df = pd.DataFrame([result_record])
                    cols = _build_result_columns(row.index, include_text=args.enable_text_save)
                    if csv_lock_supported:
                        _append_result_to_csv_locked(args.output_csv_path, output_df, cols)
                    else:
                        shard_path = _rank_csv_path(args.output_csv_path, rank)
                        _append_result_to_csv_unlocked(shard_path, output_df, cols)
                except Exception as e:
                    print(f"[Rank {rank}] CRITICAL: Failed to append result to CSV for {video_path}. Error: {e}")

        except Exception as e:
            print(f"[Rank {rank}] Error during inference for {video_path}: {e}")
            import traceback
            traceback.print_exc()
            pipe.qwen25vl_last_thinking_metadata = None
            if getattr(pipe, "qwen25vl", None) is not None and hasattr(pipe.qwen25vl, "latest_forward_thinking_metadata"):
                pipe.qwen25vl.latest_forward_thinking_metadata = None
        
        del video
        if 'output_video' in locals():
            del output_video
        pipe.qwen25vl_last_thinking_metadata = None
        if getattr(pipe, "qwen25vl", None) is not None and hasattr(pipe.qwen25vl, "latest_forward_thinking_metadata"):
            pipe.qwen25vl.latest_forward_thinking_metadata = None
        _cleanup_cuda_memory()
        continue

    _write_rank_done_marker(sync_dir, rank, len(df))

    if dist.get_rank() == 0:
        missing_ranks = _wait_for_all_rank_done_markers(
            sync_dir=sync_dir,
            world_size=world_size,
            timeout_seconds=FINAL_SYNC_TIMEOUT_MINUTES * 60,
            poll_seconds=FINAL_SYNC_POLL_SECONDS,
        )
        if missing_ranks:
            print(
                f"Warning: final rank sync timed out after {FINAL_SYNC_TIMEOUT_MINUTES} minutes; "
                f"missing completion markers from ranks {missing_ranks}. Proceeding with best-effort finalization."
            )
        else:
            print("All rank completion markers received.")

        print("\nAll tasks in this run completed.")

        # If we used per-rank sharded CSVs, merge them into the final output CSV.
        if args.output_csv_path and not csv_lock_supported:
            try:
                import glob

                base_csv = args.output_csv_path
                shard_glob = os.path.splitext(base_csv)[0] + ".rank*.csv"
                shard_paths = sorted(glob.glob(shard_glob))
                dfs = []

                # Include existing base CSV if present and non-empty (e.g. from earlier runs).
                if os.path.exists(base_csv) and os.path.getsize(base_csv) > 0:
                    dfs.append(pd.read_csv(base_csv))

                for sp in shard_paths:
                    if os.path.getsize(sp) <= 0:
                        continue
                    dfs.append(pd.read_csv(sp))

                if dfs:
                    merged = pd.concat(dfs, ignore_index=True)
                    if 'original_video' in merged.columns:
                        merged['original_video'] = merged['original_video'].apply(
                            lambda x: _normalize_original_video_path(x, args.input_root_dir)
                        )
                        merged = merged.drop_duplicates(subset=['original_video'], keep='last')
                    merged.to_csv(base_csv, index=False)
                    print(f"Merged {len(shard_paths)} sharded CSVs into {base_csv}.")

                    # Clean up intermediate shard files after a successful merge.
                    removed = 0
                    for sp in shard_paths:
                        try:
                            os.remove(sp)
                            removed += 1
                        except FileNotFoundError:
                            pass
                        except Exception as e:
                            print(f"Warning: failed to remove shard CSV {sp}. Error: {e}")
                    if removed:
                        print(f"Removed {removed} intermediate shard CSV files.")
                else:
                    # No data written (should not happen if videos were saved), keep file as-is.
                    print(f"Warning: no sharded CSV content found to merge for {base_csv}.")
            except Exception as e:
                print(f"CRITICAL: Failed to merge sharded CSVs into {args.output_csv_path}. Error: {e}")
        
        # Reconcile the final CSV with processed videos in the output directory.
        if args.output_csv_path:
            print("\nStarting final CSV reconciliation...")
            try:
                # 1. Read the original source CSV to get all expected entries
                source_df = pd.read_csv(args.csv_path)
                source_df['absolute_video_path'] = source_df['original_video'].apply(
                    lambda x: _normalize_original_video_path(x, args.input_root_dir)
                )

                # 2. Read the output CSV as it currently stands
                output_df = pd.DataFrame()
                if os.path.exists(args.output_csv_path) and os.path.getsize(args.output_csv_path) > 0:
                    output_df = pd.read_csv(args.output_csv_path)
                    if 'original_video' in output_df.columns:
                        output_df['original_video'] = output_df['original_video'].apply(
                            lambda x: _normalize_original_video_path(x, args.input_root_dir)
                        )
                
                # 3. Find which videos are already logged
                logged_videos = set()
                if not output_df.empty and 'original_video' in output_df.columns:
                    logged_videos = set(output_df['original_video'].tolist())
                
                print(f"Found {len(logged_videos)} records in the final output CSV.")

                # 4. Find records that are in source but not in output
                missing_records_df = source_df[~source_df['absolute_video_path'].isin(logged_videos)]
                
                if missing_records_df.empty:
                    print("CSV is already complete. No backfilling needed.")
                else:
                    print(f"Found {len(missing_records_df)} potential missing records. Checking for existing output files...")
                    records_to_add = []
                    for _, row in missing_records_df.iterrows():
                        # 5. For each missing record, check if the output video file actually exists
                        video_basename = os.path.basename(row['original_video'])
                        expected_output_path = os.path.join(args.output_dir, video_basename)
                        
                        if os.path.exists(expected_output_path):
                            # 6. If it exists, prepare the full record to be appended
                            new_record = row.copy()
                            new_record['original_video'] = _normalize_original_video_path(
                                row['original_video'], args.input_root_dir
                            )
                            new_record[EDITED_RESULT_COLUMN] = os.path.abspath(expected_output_path)
                            if args.enable_text_save:
                                t, c = _load_text_record(args.output_dir, new_record['original_video'])
                                _attach_text_columns(new_record, t, c)
                            records_to_add.append(new_record)
                            print(f"  - Found existing file for missing record: {video_basename}")
                    
                    if records_to_add:
                        # 7. Append all found records to the CSV
                        backfill_df = pd.DataFrame(records_to_add)
                        
                        # Ensure column order matches the existing CSV if it exists
                        header = not os.path.exists(args.output_csv_path) or os.path.getsize(args.output_csv_path) == 0
                        
                        cols = _build_result_columns(source_df.columns.tolist(), include_text=args.enable_text_save)

                        # Avoid flock/fsync here; write the full CSV in one shot.
                        final_df = pd.concat([output_df, backfill_df], ignore_index=True) if not output_df.empty else backfill_df
                        if 'original_video' in final_df.columns:
                            final_df['original_video'] = final_df['original_video'].apply(
                                lambda x: _normalize_original_video_path(x, args.input_root_dir)
                            )
                            final_df = final_df.drop_duplicates(subset=['original_video'], keep='last')
                        final_df.to_csv(args.output_csv_path, index=False, columns=cols)
                        print(f"\nSuccessfully backfilled {len(records_to_add)} missing records into {args.output_csv_path}.")
                    else:
                        print("No existing video files found for missing records. Nothing to backfill.")

            except Exception as e:
                print(f"CRITICAL: An error occurred during CSV reconciliation: {e}")

    if dist.get_rank() == 0:
        print("\nScript finished.")

    if dist.get_rank() == 0:
        _cleanup_rank_done_markers(sync_dir, world_size)

    if dist.is_initialized():
        dist.destroy_process_group()
