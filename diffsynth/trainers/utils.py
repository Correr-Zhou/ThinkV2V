import imageio, os, torch, warnings, torchvision, argparse, json, copy, csv
from collections import OrderedDict
from peft import LoraConfig, inject_adapter_in_model
from PIL import Image
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import random
import wandb
from diffsynth import save_video, VideoData, save_frames
from accelerate.utils import gather_object
import imageio
import numpy as np
import os
import time
import re
from datetime import datetime
from safetensors.torch import load_file


def _read_metadata_csv(path):
    delimiter = ","
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(8192)
        if sample:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except Exception:
        delimiter = ","
    metadata = pd.read_csv(path, sep=delimiter, on_bad_lines='skip')
    if len(metadata.columns) == 1 and delimiter != ",":
        first_col = str(metadata.columns[0])
        if "," in first_col and ";" not in first_col:
            metadata = pd.read_csv(path, sep=",", on_bad_lines='skip')
    return metadata

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("image",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            reso = args.reso
            
        self.base_path = base_path
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.repeat = repeat
        self.reso = reso

        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in tqdm(f):
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = _read_metadata_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]


    def generate_metadata(self, folder):
        image_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            image_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["image"] = image_list
        metadata["prompt"] = prompt_list
        return metadata
    
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image, reso='720p'):
        orig_width, orig_height = image.size
        is_landscape = orig_width >= orig_height
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
            raise ValueError(f"Not supported resolution: {reso}，please use '480p' or '720p'")

        target_height = target_height // self.height_division_factor * self.height_division_factor
        target_width = target_width // self.width_division_factor * self.width_division_factor
        return target_height, target_width
    
    
    def load_image(self, file_path):
        image = Image.open(file_path).convert("RGB")
        image = self.crop_and_resize(image, *self.get_height_width(image, self.reso))
        return image
    
    
    def load_data(self, file_path):
        return self.load_image(file_path)


    def __getitem__(self, data_id):
        data = self.data[data_id % len(self.data)].copy()
        for key in self.data_file_keys:
            if key in data:
                if key == "mllm_hidden_states":
                    continue
                if isinstance(data[key], list):
                    path = [os.path.join(self.base_path, p) for p in data[key]]
                    data[key] = [self.load_data(p) for p in path]
                else:
                    path = os.path.join(self.base_path, data[key])
                    data[key] = self.load_data(path)
                if data[key] is None:
                    warnings.warn(f"cannot load file {data[key]}.")
                    return None
        return data
    

    def __len__(self):
        return len(self.data) * self.repeat



class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        num_frames=81,
        time_division_factor=4, time_division_remainder=1,
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        data_file_keys=("video",),
        image_file_extension=("jpg", "jpeg", "png", "webp"),
        video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
        repeat=1,
        args=None,
    ):
        if args is not None:
            base_path = args.dataset_base_path
            metadata_path = args.dataset_metadata_path
            height = args.height
            width = args.width
            max_pixels = args.max_pixels
            num_frames = args.num_frames
            data_file_keys = args.data_file_keys.split(",")
            repeat = args.dataset_repeat
            is_qwen25vl = args.is_qwen25vl
            reso = args.reso
        
        self.base_path = base_path
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.max_pixels = max_pixels
        self.height = height
        self.width = width
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.data_file_keys = data_file_keys
        self.image_file_extension = image_file_extension
        self.video_file_extension = video_file_extension
        self.repeat = repeat
        self.is_qwen25vl = is_qwen25vl
        self.reso = reso
        self.offline_feature_invalid_paths = set()
        
        if height is not None and width is not None:
            print("Height and width are fixed. Setting `dynamic_resolution` to False.")
            self.dynamic_resolution = False
        elif height is None and width is None:
            print("Height and width are none. Setting `dynamic_resolution` to True.")
            self.dynamic_resolution = True
            
        if metadata_path is None:
            print("No metadata. Trying to generate it.")
            metadata = self.generate_metadata(base_path)
            print(f"{len(metadata)} lines in metadata.")
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]
        else:
            if isinstance(metadata_path, str):
                metadata_paths = [path.strip() for path in metadata_path.split(",")]
            else:
                raise TypeError("metadata_path must be a comma-separated string or None")
            
            all_data = []
            for path in metadata_paths:
                if path.endswith(".json"):
                    with open(path, "r") as f:
                        data = json.load(f)
                    all_data.extend(data)
                else:
                    metadata_df = _read_metadata_csv(path)
                    data = [metadata_df.iloc[i].to_dict() for i in range(len(metadata_df))]
                    all_data.extend(data)
            
            self.data = all_data
            print(f"Loaded {len(self.data)} samples from {len(metadata_paths)} metadata files.")

        original_count = len(self.data)
        self.data = self._prefilter_invalid_prompt_rows(self.data)
        filtered_count = original_count - len(self.data)
        if filtered_count > 0:
            print(
                f"Prompt prefilter keeps {len(self.data)}/{original_count} rows "
                f"(filtered {filtered_count})"
            )

        self.use_offline_mllm_hidden_states = any(
            not pd.isna(sample.get("mllm_hidden_states", None)) for sample in self.data
        )
        if self.use_offline_mllm_hidden_states:
            print("Detected mllm_hidden_states column; offline MLLM hidden states will be used when available.")
            
    def __len__(self):
        return len(self.data) * self.repeat

    def generate_metadata(self, folder):
        video_list, prompt_list = [], []
        file_set = set(os.listdir(folder))
        for file_name in file_set:
            if "." not in file_name:
                continue
            file_ext_name = file_name.split(".")[-1].lower()
            file_base_name = file_name[:-len(file_ext_name)-1]
            if file_ext_name not in self.image_file_extension and file_ext_name not in self.video_file_extension:
                continue
            prompt_file_name = file_base_name + ".txt"
            if prompt_file_name not in file_set:
                continue
            with open(os.path.join(folder, prompt_file_name), "r", encoding="utf-8") as f:
                prompt = f.read().strip()
            video_list.append(file_name)
            prompt_list.append(prompt)
        metadata = pd.DataFrame()
        metadata["video"] = video_list
        metadata["prompt"] = prompt_list
        return metadata
        
        
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    
    def get_height_width(self, image, reso='720p'):
        orig_width, orig_height = image.size
        is_landscape = orig_width >= orig_height
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
            raise ValueError(f"Not supported resolution: {reso}，please use '480p' or '720p'")
        target_height = target_height // self.height_division_factor * self.height_division_factor
        target_width = target_width // self.width_division_factor * self.width_division_factor
        return target_height, target_width
    
    
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
    
    def load_video(self, file_path):
        try:
            reader = imageio.get_reader(file_path)
        except Exception as e:
            warnings.warn(f"Unable to open the video file {file_path}: {str(e)}")
            return None
        
        try:
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                try:
                    frame = reader.get_data(frame_id)
                except Exception as e:
                    warnings.warn(f"Reading video frames {frame_id} Failed {file_path}: {str(e)}")
                    return None
                
                try:
                    frame = Image.fromarray(frame)
                    frame = self.crop_and_resize(frame, *self.get_height_width(frame,self.reso))
                    frames.append(frame)
                except Exception as e:
                    warnings.warn(f"Processing video frames {frame_id} failed {file_path}: {str(e)}")
                    return None
            
            return frames
        
        except Exception as e:
            warnings.warn(f"Failed when processing video {file_path}: {str(e)}")
            return None
        
        finally:
            try:
                reader.close()
            except:
                pass
    
    
    def load_image(self, file_path):
        try:
            image = Image.open(file_path).convert("RGB")
            image = self.crop_and_resize(image, *self.get_height_width(image, self.reso))
            frames = [image]
            return frames
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None
        
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.image_file_extension
    
    
    def is_video(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        return file_ext_name.lower() in self.video_file_extension
    
    
    def load_data(self, file_path):
        if self.is_image(file_path):
            return self.load_image(file_path)
        elif self.is_video(file_path):
            return self.load_video(file_path)
        else:
            return None

    def _normalize_optional_path(self, path):
        if path is None or pd.isna(path):
            return None
        if not isinstance(path, str):
            path = str(path)
        path = path.strip()
        if not path:
            return None
        if self.base_path and not os.path.isabs(path):
            path = os.path.join(self.base_path, path)
        return path

    def _load_mllm_hidden_states_payload(self, path):
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            hidden_states = payload
            text = ""
        elif isinstance(payload, dict):
            if "hidden_states" in payload:
                hidden_states = payload["hidden_states"]
            elif "mllm_hidden_states" in payload:
                hidden_states = payload["mllm_hidden_states"]
            else:
                raise KeyError("mllm hidden-state payload must contain 'hidden_states' or 'mllm_hidden_states'")
            text = str(payload.get("text") or payload.get("mllm_text") or "")
        else:
            raise TypeError(f"mllm_hidden_states payload must be a tensor or dict, got {type(payload).__name__}")

        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"mllm_hidden_states is not a tensor: {type(hidden_states).__name__}")
        if hidden_states.ndim != 2:
            raise ValueError(f"mllm_hidden_states should be 2D, got shape={tuple(hidden_states.shape)}")
        if hidden_states.shape[0] == 0:
            raise ValueError("mllm_hidden_states is empty")
        return {"hidden_states": hidden_states, "text": text, "char_count": len(text)}

    def _load_offline_mllm_hidden_states(self, data, sample_id=None):
        path = self._normalize_optional_path(data.get("mllm_hidden_states", None))
        if path is None:
            return True
        if path in self.offline_feature_invalid_paths:
            return False
        if not os.path.exists(path):
            warnings.warn(f"mllm_hidden_states file does not exist for sample {sample_id}: {path}")
            self.offline_feature_invalid_paths.add(path)
            return False
        try:
            data["mllm_hidden_states"] = self._load_mllm_hidden_states_payload(path)
            return True
        except Exception as exc:
            warnings.warn(f"Failed to load mllm_hidden_states for sample {sample_id}: {type(exc).__name__}: {exc}")
            self.offline_feature_invalid_paths.add(path)
            return False

    def _normalize_prompt_inplace(self, data):
        if "prompt" not in data:
            return "missing prompt field in metadata"

        prompt = data.get("prompt", None)
        if prompt is None or pd.isna(prompt):
            return "prompt is missing"

        prompt = str(prompt).strip()
        if not prompt:
            return "prompt is empty"

        data["prompt"] = prompt
        return None

    def _prefilter_invalid_prompt_rows(self, data_list):
        filtered_data = []
        filtered_reason_counts = {}
        for sample in data_list:
            reason = self._normalize_prompt_inplace(sample)
            if reason is None:
                filtered_data.append(sample)
                continue
            filtered_reason_counts[reason] = filtered_reason_counts.get(reason, 0) + 1

        if filtered_reason_counts:
            summary = ", ".join(f"{reason}: {count}" for reason, count in sorted(filtered_reason_counts.items()))
            print(f"Prompt prefilter summary: {summary}")
        return filtered_data

    def __getitem__(self, data_id):
        current_id = data_id % len(self.data)
        data = self.data[current_id].copy()
        valid, shapes_match = self._load_and_check(data, sample_id=current_id)
        
        if valid and shapes_match:
            return data
        
        random.seed(data_id)
        
        max_attempts = len(self.data) * 2
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            random_id = random.randint(0, len(self.data) - 1)

            if random_id == current_id:
                continue
                
            data = self.data[random_id].copy()
            valid, shapes_match = self._load_and_check(data, sample_id=random_id)
            
            if valid and shapes_match:
                return data
        raise RuntimeError(f"After {max_attempts} attempts, no valid data sample was found.")
    
    def _load_and_check(self, data, sample_id=None):
        valid = True
        loaded_data = {}
        path_only_keys = []  

        prompt_reason = self._normalize_prompt_inplace(data)
        if prompt_reason is not None:
            warnings.warn(f"Invalid prompt for sample {sample_id}: {prompt_reason}")
            return False, False

        if self.use_offline_mllm_hidden_states and not self._load_offline_mllm_hidden_states(data, sample_id=sample_id):
            return False, False

        original_video_path = None
        if "original_video" in data:
            if self.base_path:
                original_video_path = os.path.join(self.base_path, data["original_video"])
            else:
                original_video_path = data["original_video"]
        
        for key in self.data_file_keys:
            if key not in data:
                warnings.warn(f"Key not in data: {key}")
                return False, False
            if self.base_path:
                path = os.path.join(self.base_path, data[key])
            else:
                path = data[key]
            if not os.path.exists(path):
                warnings.warn(f"The file does not exist.: {path}")
                return False, False
            if "mask" in key.lower():
                loaded_data[key] = path
                path_only_keys.append(key)
            else:
                loaded = self.load_data(path)
                if loaded is None:
                    warnings.warn(f"Failed to load file: {path}")
                    return False, False
                loaded_data[key] = loaded
        
        if hasattr(self, 'is_qwen25vl') and self.is_qwen25vl and original_video_path is not None:
            loaded_data["qwenvideo"] = original_video_path
        
        if "original_video" in loaded_data and "video" in loaded_data:
            original_video_frams = loaded_data["original_video"]
            video_frames = loaded_data["video"]
            if len(video_frames) != len(original_video_frams):
                warnings.warn("The original_video and video have different frame rates and cannot be aligned.")
                return False, False
        
        non_mask_keys = [k for k in self.data_file_keys if k not in path_only_keys]
        if len(non_mask_keys) < 2:
            shapes_match = True
        else:
            key1, key2 = non_mask_keys[0], non_mask_keys[1]
            shapes_match = (len(loaded_data[key1]) == len(loaded_data[key2]))
        for key, value in loaded_data.items():
            data[key] = value
        return True, shapes_match


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self
        
        
    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules
    
    
    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names
    
    
    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    def mapping_lora_state_dict(self, state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "lora_A.weight" in key or "lora_B.weight" in key:
                new_key = key.replace("lora_A.weight", "lora_A.default.weight").replace("lora_B.weight", "lora_B.default.weight")
                new_state_dict[new_key] = value
            elif "lora_A.default.weight" in key or "lora_B.default.weight" in key:
                new_state_dict[key] = value
        return new_state_dict
    
    
    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x: x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0

    def on_step_end(self, accelerator, model, save_steps=None):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def on_epoch_end(self, accelerator, model, epoch_id):
        self.save_model(accelerator, model, f"epoch-{epoch_id}.safetensors")

    def on_training_end(self, accelerator, model, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def save_model(self, accelerator, model, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                state_dict,
                remove_prefix=self.remove_prefix_in_ckpt,
            )
            state_dict = self.state_dict_converter(state_dict)
            save_path = os.path.join(self.output_path, "models")
            os.makedirs(save_path, exist_ok=True)
            path = os.path.join(save_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    wandb_project="wan21",
    configs=None,
):
    def _format_epoch_tag(x: float) -> str:
        # Keep filenames readable and stable: 1 -> "1", 1.7 -> "1.7", 0.0500 -> "0.05"
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _parse_save_epochs(v):
        """Parse save epoch points from CLI.

        Supported forms:
        - "1.7"
        - "0.5,1.7,2"
        - "[0.5, 1.7, 2]" (JSON list)
        """
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            vals = list(v)
        else:
            s = str(v).strip()
            if s == "":
                return []
            if s.startswith("["):
                try:
                    vals = json.loads(s)
                except Exception as e:
                    raise ValueError(f"Invalid --save_epochs JSON list: {s}") from e
            else:
                vals = [p.strip() for p in s.split(",") if p.strip() != ""]
        out = []
        for x in vals:
            fx = float(x)
            if fx <= 0:
                raise ValueError(f"--save_epochs must be > 0, got {fx}")
            out.append(fx)
        # de-dup + sort
        out = sorted(set(out))
        return out

    def _parse_save_at_steps(v):
        """Parse save step points from CLI.

        Supported forms:
        - "55296"
        - "55296,11060"
        - "[55296, 11060]" (JSON list)
        """
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            vals = list(v)
        else:
            s = str(v).strip()
            if s == "":
                return []
            if s.startswith("["):
                try:
                    vals = json.loads(s)
                except Exception as e:
                    raise ValueError(f"Invalid --save_at_steps JSON list: {s}") from e
            else:
                vals = [p.strip() for p in s.split(",") if p.strip() != ""]

        out = []
        for x in vals:
            ix = int(x)
            if ix <= 0:
                raise ValueError(f"--save_at_steps must be > 0, got {ix}")
            out.append(ix)
        out = sorted(set(out))
        return out

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers, drop_last=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    use_wandb = bool(wandb_project)
    if use_wandb and accelerator.is_main_process:
        run_name = os.path.basename(configs.output_path) if configs and hasattr(configs, 'output_path') else "default_run"
        wandb.init(project=wandb_project, name=run_name)
        wandb.config.update(configs)
    
    # Resume training
    if configs.resume_training and os.path.exists(os.path.join(configs.output_path, "models")):
        model_dir = os.path.join(configs.output_path, "models")
        files = os.listdir(model_dir)
        step_files = [f for f in files if f.startswith("step-") and f.endswith(".safetensors")]
        if len(step_files) > 0:
            steps = [int(re.findall(r"step-(\d+).safetensors", f)[0]) for f in step_files]
            # Try the latest checkpoints first; if the newest file is corrupted / incomplete,
            # fall back to previous ones.
            state_dict = None
            resumed_step = None
            latest_ckpt = None
            for s in sorted(steps, reverse=True):
                ckpt_path = os.path.join(model_dir, f"step-{s}.safetensors")
                try:
                    print(f"Resuming training from {ckpt_path}...")
                    state_dict = load_file(ckpt_path)
                    resumed_step = s
                    latest_ckpt = ckpt_path
                    break
                except Exception as e:
                    print(
                        f"[WARN] Failed to load checkpoint {ckpt_path}, will try an earlier one. "
                        f"Error: {type(e).__name__}: {e}"
                    )

            if state_dict is None:
                print(f"[WARN] No valid step-*.safetensors found under {model_dir}, skip resuming.")
                resumed_step = None
            
            # Add prefix if needed
            if state_dict is not None and configs.remove_prefix_in_ckpt:
                prefix = configs.remove_prefix_in_ckpt
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_state_dict[prefix + key] = value
                state_dict = new_state_dict
            
            # Load state dict
            if state_dict is not None:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if len(missing) > 0:
                    print(f"Missing keys: {missing}")
                if len(unexpected) > 0:
                    print(f"Unexpected keys: {unexpected}")

                model_logger.num_steps = int(resumed_step)
                print(f"Resumed at step {resumed_step}")

    # When enabled, only save at --save_at_steps, skip epoch saves and stop once the last save_at_steps is reached.
    save_at_steps_only = bool(getattr(configs, "save_at_steps_only", False))

    # Optional: save checkpoints at specific epoch points (supports fractional epoch)
    save_epoch_points = []
    if (not save_at_steps_only) and configs is not None and hasattr(configs, "save_epochs"):
        try:
            save_epoch_points = _parse_save_epochs(configs.save_epochs)
        except Exception as e:
            raise ValueError(f"Failed to parse --save_epochs={getattr(configs, 'save_epochs', None)}") from e

    steps_per_epoch = len(dataloader)
    if steps_per_epoch <= 0:
        raise RuntimeError("Empty dataloader: cannot compute steps_per_epoch for --save_epochs")

    # Ignore save points beyond total epochs, but keep behavior predictable.
    if num_epochs is not None:
        save_epoch_points = [x for x in save_epoch_points if x <= float(num_epochs) + 1e-12]

    # Optional: save checkpoints at specific step points
    save_step_points = []
    if configs is not None and hasattr(configs, "save_at_steps"):
        try:
            save_step_points = _parse_save_at_steps(configs.save_at_steps)
        except Exception as e:
            raise ValueError(f"Failed to parse --save_at_steps={getattr(configs, 'save_at_steps', None)}") from e

    if save_at_steps_only and not save_step_points:
        raise ValueError("--save_at_steps_only must be used with --save_at_steps; otherwise no checkpoint will be saved.")

    last_save_step_point = max(save_step_points) if (save_at_steps_only and save_step_points) else None

    # If resuming, skip targets already passed.
    next_save_epoch_idx = 0
    if save_epoch_points:
        current_epoch_progress = model_logger.num_steps / steps_per_epoch
        while next_save_epoch_idx < len(save_epoch_points) and current_epoch_progress >= save_epoch_points[next_save_epoch_idx] - 1e-12:
            next_save_epoch_idx += 1

    next_save_step_idx = 0
    if save_step_points:
        while next_save_step_idx < len(save_step_points) and model_logger.num_steps >= save_step_points[next_save_step_idx]:
            next_save_step_idx += 1

    should_stop_early = False

    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            step_start_time = time.time()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps)

                # Save at specified step points
                if next_save_step_idx < len(save_step_points):
                    while next_save_step_idx < len(save_step_points) and model_logger.num_steps >= save_step_points[next_save_step_idx]:
                        sp = save_step_points[next_save_step_idx]
                        # Expected: model_logger.num_steps == sp
                        if model_logger.num_steps == sp:
                            model_logger.save_model(accelerator, model, f"step-{sp}.safetensors")
                        else:
                            model_logger.save_model(
                                accelerator,
                                model,
                                f"step-at-{sp}_step-{model_logger.num_steps}.safetensors",
                            )
                        next_save_step_idx += 1

                        if last_save_step_point is not None and sp == last_save_step_point:
                            should_stop_early = True
                            break

                    if should_stop_early:
                        break

                # Save at specified epoch points (fractional epoch supported)
                if next_save_epoch_idx < len(save_epoch_points):
                    current_epoch_progress = model_logger.num_steps / steps_per_epoch
                    while next_save_epoch_idx < len(save_epoch_points) and current_epoch_progress >= save_epoch_points[next_save_epoch_idx] - 1e-12:
                        ep = save_epoch_points[next_save_epoch_idx]
                        model_logger.save_model(
                            accelerator,
                            model,
                            f"epoch-at-{_format_epoch_tag(ep)}_step-{model_logger.num_steps}.safetensors",
                        )
                        next_save_epoch_idx += 1
                if use_wandb and accelerator.is_main_process:
                    wandb.log(
                        {
                            "loss": loss.item(),
                            "learning_rate": scheduler.get_last_lr()[0],
                            "steps": model_logger.num_steps,
                            "step_time": time.time() - step_start_time,
                        },
                    )
                scheduler.step()
        if should_stop_early:
            if accelerator.is_main_process and last_save_step_point is not None:
                print(f"Reached last save_at_steps={last_save_step_point}, stopping training early.")
            accelerator.wait_for_everyone()
            return

        if not save_at_steps_only:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    if not save_at_steps_only:
        model_logger.on_training_end(accelerator, model, save_steps)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the dataset.")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--max_pixels", type=int, default=1280*720, help="Maximum number of pixels per frame, used for dynamic resolution..")
    parser.add_argument("--height", type=int, default=None, help="Height of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--width", type=int, default=None, help="Width of images or videos. Leave `height` and `width` empty to enable dynamic resolution.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video. Frames are sampled from the video prefix.")
    parser.add_argument("--data_file_keys", type=str, default="image,video", help="Data file keys in the metadata. Comma-separated.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None, help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--trainable_models", type=str, default="qwen25vl_connector,dit", help="Models to train, e.g., dit, vae, text_encoder.")
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which model LoRA is added to.")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Which layers LoRA is added to.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated.")
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Whether to use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Whether to offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--save_steps", type=int, default=None, help="Number of checkpoint saving invervals. If None, checkpoints will be saved every epoch.")
    parser.add_argument(
        "--save_at_steps",
        type=str,
        default=None,
        help=(
            "Save checkpoints at specific step numbers. "
            "Examples: --save_at_steps 55296 ; --save_at_steps 55296,11060 ; --save_at_steps '[55296,11060]'"
        ),
    )
    parser.add_argument(
        "--save_epochs",
        type=str,
        default=None,
        help=(
            "Save checkpoints at specific epoch points (supports fractional epochs). "
            "Examples: --save_epochs 1.7 ; --save_epochs 0.5,1.7,2 ; --save_epochs '[0.5,1.7,2]'"
        ),
    )
    parser.add_argument(
        "--save_at_steps_only",
        action="store_true",
        help=(
            "When set, skip all epoch checkpoint saves (epoch-* / epoch-at-*) and stop training right after "
            "saving the last checkpoint specified by --save_at_steps."
        ),
    )
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    parser.add_argument("--wandb_project", type=str, default="", help="Weights & Biases project name. Leave empty to disable W&B logging.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--reso", type=str, default='480p', help="The resolution used during training.('480p' or '720p)")
    parser.add_argument("--is_qwen25vl", type=str2bool, default=True, help="Whether to QwenVL.")
    parser.add_argument("--resume_training", type=str2bool, default=False, help="Whether to resume training from the latest checkpoint.")
    return parser
