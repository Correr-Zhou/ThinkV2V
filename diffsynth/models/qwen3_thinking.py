import gc
import torch
import json
import os
import hashlib
import imageio
from qwen_vl_utils import process_vision_info
import random
import re
from diffusers.utils import load_video
from PIL import Image
from typing import List, Optional, Dict, Any
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3ThinkingVideoEmbedder(torch.nn.Module):
    def __init__(self, model_path, dtype=torch.bfloat16, device="cuda"):
        super(Qwen3ThinkingVideoEmbedder, self).__init__()
        self.dtype = dtype
        self.device = device
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(device)

        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
        )

        self.txt_sys_prompt = "First, thoroughly analyze the user's text description to deconstruct its key components: the primary subject(s), the background and environment, the specified composition (e.g., close-up, wide shot), the lighting conditions (e.g., golden hour, neon lights), the color palette, and the desired artistic style (e.g., photorealistic, oil painting, anime). Based on this comprehensive analysis, construct a detailed internal 'scene blueprint'. Finally, generate a high-fidelity and artistically coherent image or video that accurately brings this blueprint to life, ensuring all aspects of the user's request are visually represented."
        self.img_sys_prompt = "First, analyze the input image's key visual features, including its color palette, lighting, composition, key objects, and background. Based on this analysis, interpret the user's text instruction to create a clear plan for modification. This plan should specify exactly what visual elements to alter (e.g., color grading, object removal, style transfer, background replacement, adding visual effects), and how the edits should be executed. Finally, generate a new image that precisely implements the requested modifications."
        self.vid_sys_prompt = """
You are a video editing instruction refiner. Your task is to take a user's potentially vague or underspecified editing instruction along with the input video, and produce a single, precise, and unambiguous editing prompt that a video editing model can directly execute.

## Output Format (STRICTLY follow this)
Your final answer must contain ONLY the refined video editing instruction.
- Do NOT include any analysis, reasoning, planning, or explanation.
- Do NOT use headers, bullet points, or section labels.
- Do NOT mention timestamps, timecodes, durations, frame numbers, or numeric temporal ranges in either the <think> block or the final answer.
- Output a single cohesive paragraph that:
  (a) Precisely identifies the target element(s) using detailed visual attributes (appearance, clothing, color, texture, position, motion, spatial relationships, etc.) observed from the actual video.
  (b) Clearly states the editing operation (remove, replace, modify, add, restyle, etc.).
  (c) Describes the expected result for the edited region (e.g., background reconstruction, replacement content).
  (d) Specifies preservation constraints — what must remain unchanged.
  (e) Includes temporal consistency requirements where applicable (e.g., "across the entire video sequence", "from frame X onward", "maintain coherent motion throughout").

## Thinking Instructions
Before answering, use the <think> block to internally:
1. **Video Analysis**: Examine the input video's visual content — identify all key subjects, their detailed appearance (hair, clothing, accessories, pose, expression), spatial layout, background elements, lighting, camera motion, and temporal dynamics (subject movement patterns, interactions).
2. **Instruction Interpretation**: Map the user's vague/ambiguous references to specific visual elements observed in the video. Resolve ambiguities (e.g., "the younger individual" → identify which person is younger based on visual cues, then describe them exhaustively).
3. **Edit Planning**: Determine the precise scope of the edit — spatial extent (which pixels/regions), temporal extent (which frames), and what reconstruction or generation is needed (e.g., inpainting occluded background, filling gaps in motion).
4. **Constraint Identification**: Identify what must be explicitly preserved (other subjects, background, audio sync, lighting continuity) and what temporal consistency challenges exist (camera movement, lighting changes, occlusions).
5. **Self-Check**: Verify that your refined instruction is self-contained — a video editing model reading ONLY your output (with the video) should have zero ambiguity about what to edit, how, and what to preserve.

All of the above must stay inside <think>...</think>.
Your visible answer is ONLY the final refined editing instruction.
"""
        # Token offsets used by the Qwen-VL embedding extraction helpers.
        self.text_drop_idx = 130
        self.image_drop_idx = 109
        self.video_drop_idx = 108

        # Additional Instruction for Thinking Mode
        self.thinking_instruction = ""
        self.latest_forward_thinking_metadata = None
        self._forward_thinking_call_count = 0

    @staticmethod
    def from_pretrained(path, torch_dtype=torch.bfloat16, device="cuda"):
        return Qwen3ThinkingVideoEmbedder(path, dtype=torch_dtype, device=device)

    @property
    def current_device(self):
        return next(self.model.parameters()).device

    @staticmethod
    def _cleanup_cuda_memory():
        gc.collect()
        if torch.cuda.is_available() and hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    @staticmethod
    def _adjust_num_frames(num_frames):
        if num_frames % 4 == 1:
            return num_frames
        for i in range(num_frames, 0, -1):
            if i % 4 == 1:
                return i
        return 1

    @staticmethod
    def _get_video_info(video_path):
        reader = imageio.get_reader(video_path)
        try:
            num_frames = reader.count_frames()
            first_frame = reader.get_data(0)
            height, width = first_frame.shape[:2]
        finally:
            reader.close()
        return num_frames, width, height

    @staticmethod
    def _determine_dimensions(original_width, original_height, reso='720p'):
        is_landscape = original_width >= original_height
        if reso == '480p':
            target_height, target_width = (480, 832) if is_landscape else (832, 480)
        elif reso == '720p':
            target_height, target_width = (704, 1280) if is_landscape else (1280, 704)
        else:
            raise ValueError(f"Unsupported resolution parameter: {reso}, please use '480p' or '720p'.")
        return target_width, target_height

    def _preprocess_video_like_offline_extract(self, video_item, height=None, width=None):
        if not isinstance(video_item, str):
            return video_item

        num_frames, original_width, original_height = self._get_video_info(video_item)
        num_frames = self._adjust_num_frames(num_frames)

        if height is None or width is None:
            reso = '720p'
            if {int(original_width), int(original_height)} == {832, 480}:
                reso = '480p'
            width, height = self._determine_dimensions(original_width, original_height, reso=reso)

        def convert_video(video: List[Image.Image]) -> List[Image.Image]:
            resized_video = video[:num_frames]
            resized_video = [frame.resize((width, height)) for frame in resized_video]
            return resized_video

        return load_video(video_item, convert_method=convert_video)[:num_frames]

    @staticmethod
    def _preview_text(text, max_len=240):
        text = "" if text is None else str(text).replace("\n", " ").strip()
        if len(text) <= max_len:
            return text
        return text[:max_len] + " ..."

    def _log_forward_thinking_issue(self, sample_id, message, prompt=None):
        rank = int(os.environ.get("RANK", "0"))
        prompt_preview = self._preview_text(prompt, 120) if prompt is not None else ""
        suffix = f" prompt={prompt_preview}" if prompt_preview else ""
        print(f"[Qwen3Thinking][rank {rank}][sample {sample_id}] {message}{suffix}", flush=True)

    def _normalize_loaded_offline_feature(self, payload, label):
        if not isinstance(payload, dict):
            raise TypeError(f"{label} payload is not a dict: {type(payload).__name__}")

        required_keys = {"hidden_states", "text", "char_count"}
        missing_keys = [key for key in required_keys if key not in payload]
        if missing_keys:
            raise KeyError(f"{label} payload missing keys: {missing_keys}")

        hidden_states = payload["hidden_states"]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"{label}.hidden_states is not a tensor: {type(hidden_states).__name__}")
        if hidden_states.ndim != 2:
            raise ValueError(f"{label}.hidden_states should be 2D, got shape={tuple(hidden_states.shape)}")
        if hidden_states.shape[0] == 0:
            raise ValueError(f"{label}.hidden_states is empty")

        char_count = payload["char_count"]
        if not isinstance(char_count, int):
            try:
                char_count = int(char_count)
            except Exception as exc:
                raise TypeError(f"{label}.char_count is invalid: {char_count!r}") from exc

        return {
            "hidden_states": hidden_states.to(self.current_device),
            "text": str(payload["text"] or ""),
            "char_count": char_count,
        }

    def _extract_single_offline_feature(self, payload, label):
        if payload is None:
            return None
        if isinstance(payload, (list, tuple)):
            if len(payload) != 1:
                raise ValueError(f"{label} batch size mismatch: expect 1, got {len(payload)}")
            payload = payload[0]
        if not isinstance(payload, dict):
            raise TypeError(f"{label} should be a loaded payload dict, got {type(payload).__name__}")

        unpacked_payload = dict(payload)
        hidden_states = unpacked_payload.get("hidden_states", None)
        if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3:
            if hidden_states.shape[0] != 1:
                raise ValueError(f"{label}.hidden_states batch size mismatch: expect 1, got {hidden_states.shape[0]}")
            unpacked_payload["hidden_states"] = hidden_states[0]

        text = unpacked_payload.get("text", None)
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError(f"{label}.text batch size mismatch: expect 1, got {len(text)}")
            unpacked_payload["text"] = text[0]

        char_count = unpacked_payload.get("char_count", None)
        if isinstance(char_count, torch.Tensor):
            if char_count.numel() != 1:
                raise ValueError(f"{label}.char_count batch size mismatch: expect 1 value, got {char_count.numel()}")
            unpacked_payload["char_count"] = int(char_count.item())
        elif isinstance(char_count, (list, tuple)):
            if len(char_count) != 1:
                raise ValueError(f"{label}.char_count batch size mismatch: expect 1, got {len(char_count)}")
            unpacked_payload["char_count"] = char_count[0]

        return self._normalize_loaded_offline_feature(unpacked_payload, label)

    def _pack_clean_hidden_states(self, clean_hidden_state_list):
        max_seq_len = max(hidden.size(0) for hidden in clean_hidden_state_list)
        prompt_embeds = torch.stack([
            torch.cat([hidden, hidden.new_zeros(max_seq_len - hidden.size(0), hidden.size(1))])
            for hidden in clean_hidden_state_list
        ]).to(dtype=self.dtype, device=self.current_device)
        encoder_attention_mask = torch.stack([
            torch.cat([
                torch.ones(hidden.size(0), dtype=torch.long, device=hidden.device),
                torch.zeros(max_seq_len - hidden.size(0), dtype=torch.long, device=hidden.device)
            ])
            for hidden in clean_hidden_state_list
        ])
        return prompt_embeds, encoder_attention_mask

    def _log_forward_thinking_texts(self, prompt, video_path, parsed_results, mm_type, enable_text_save=False, text_save_dir=None):
        rank = int(os.environ.get("RANK", "0"))
        if len(parsed_results) == 0:
            return

        self._forward_thinking_call_count += 1
        log_every_n = max(1, int(os.environ.get("QWEN3_THINKING_LOG_EVERY_N", "20")))
        should_print = rank == 0 and (
            self._forward_thinking_call_count == 1 or self._forward_thinking_call_count % log_every_n == 0
        )

        if enable_text_save and text_save_dir:
            # Save one JSON per sample under a sub-folder.
            # This replaces the previous JSONL behavior.
            record_dir = os.path.join(text_save_dir, "text_records")
            os.makedirs(record_dir, exist_ok=True)

            for sample_id, (sample_prompt, sample_video_path, sample_result) in enumerate(zip(prompt, video_path, parsed_results)):
                video_path_str = sample_video_path if isinstance(sample_video_path, str) else repr(sample_video_path)
                video_basename = os.path.basename(video_path_str) if isinstance(sample_video_path, str) else f"sample{sample_id}"
                stem, _ = os.path.splitext(video_basename)
                h = hashlib.sha1(video_path_str.encode("utf-8")).hexdigest()[:10]
                record_path = os.path.join(record_dir, f"{stem}__{h}.json")
                tmp_path = record_path + ".tmp"

                record = {
                    "call_id": self._forward_thinking_call_count,
                    "sample_id": sample_id,
                    "rank": rank,
                    "mm_type": mm_type,
                    "prompt": str(sample_prompt),
                    "video_path": video_path_str,
                    "thinking_text": sample_result.get("thinking_text"),
                    "clean_text": sample_result.get("clean_text"),
                    "infer_time_scaling": sample_result.get("infer_time_scaling"),
                }
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, record_path)

        if not should_print:
            return

        sample_prompt = self._preview_text(prompt[0], 120)
        sample_thinking = self._preview_text(parsed_results[0]["thinking_text"], 240)
        sample_clean = self._preview_text(parsed_results[0]["clean_text"], 240)
        print(f"[Qwen3Thinking][call {self._forward_thinking_call_count}] prompt={sample_prompt}", flush=True)
        print(f"[Qwen3Thinking][call {self._forward_thinking_call_count}] thinking={sample_thinking}", flush=True)
        print(f"[Qwen3Thinking][call {self._forward_thinking_call_count}] clean={sample_clean}", flush=True)


    def forward(self, prompt, video_path, mm_type="video", **kwargs):
        return self.forward_thinking(prompt, video_path, mm_type, **kwargs)

    def forward_thinking(
        self,
        prompt,
        video_path,
        mm_type="video",
        enable_text_save=False,
        text_save_dir=None,
        height=None,
        width=None,
        mllm_hidden_states=None,
        inference_time_thinking_scaling: bool = False,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(video_path, str):
            video_path = [video_path]

        if len(prompt) != 1 or len(video_path) != 1:
            raise ValueError(f"forward_thinking currently expects batch size 1, got prompt={len(prompt)}, video_path={len(video_path)}")

        sample_prompt = prompt[0]
        sample_video_path = video_path[0]
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(
                f"[Qwen3Thinking][start] mm_type={mm_type} enable_text_save={enable_text_save} "
                f"video={self._preview_text(sample_video_path, 160)} prompt={self._preview_text(sample_prompt, 160)}",
                flush=True,
            )
        offline_mllm_payload = self._extract_single_offline_feature(mllm_hidden_states, "mllm_hidden_states")

        if mm_type == "video":
            sys_prompt = self.vid_sys_prompt
        elif mm_type == "image":
            sys_prompt = self.img_sys_prompt
        else:
            sys_prompt = self.txt_sys_prompt
        sys_prompt_thinking = sys_prompt + self.thinking_instruction

        infer_time_scaling_info: Optional[Dict[str, Any]] = None
        use_thinking_scaling = bool(inference_time_thinking_scaling)
        thinking_scaling_n = 8

        if offline_mllm_payload is not None:
            clean_hidden_states = offline_mllm_payload["hidden_states"]
            clean_text = offline_mllm_payload.get("text", "")
            clean_char_count = offline_mllm_payload.get("char_count", len(clean_text))
            thinking_text = ""
            thinking_char_count = 0
            parsed_results = [{
                "thinking_text": "",
                "clean_text": f"[OFFLINE] {clean_text}",
                "thinking_char_count": thinking_char_count,
                "clean_char_count": clean_char_count,
            }]
        else:
            if mm_type == "video":
                sample_video_input = self._preprocess_video_like_offline_extract(sample_video_path, height=height, width=width)
            else:
                sample_video_input = sample_video_path

            def _build_inputs_from_messages(messages_obj):
                texts_obj = [
                    self.processor.apply_chat_template(
                        messages_obj, tokenize=False, add_generation_prompt=True, add_vision_id=True
                    )
                ]
                if mm_type == "video":
                    image_inputs_, video_inputs_, video_kwargs_ = process_vision_info(
                        [messages_obj],
                        image_patch_size=16,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )
                    if video_inputs_ is not None:
                        video_inputs_, video_metadatas_ = zip(*video_inputs_)
                        video_inputs_, video_metadatas_ = list(video_inputs_), list(video_metadatas_)
                    else:
                        video_metadatas_ = None
                    inputs_ = self.processor(
                        text=texts_obj,
                        images=image_inputs_,
                        videos=video_inputs_,
                        video_metadata=video_metadatas_,
                        padding=True,
                        return_tensors="pt",
                        do_resize=False,
                        **video_kwargs_,
                    )
                else:
                    image_inputs_, video_inputs_, video_kwargs_ = process_vision_info([messages_obj], return_video_kwargs=True)
                    inputs_ = self.processor(
                        text=texts_obj,
                        images=image_inputs_,
                        videos=video_inputs_,
                        padding=True,
                        return_tensors="pt",
                        **video_kwargs_,
                    )
                return inputs_.to(self.current_device)

            eos_token_ids = [self.processor.tokenizer.eos_token_id]
            if hasattr(self.processor.tokenizer, "eod_id"):
                eos_token_ids.append(self.processor.tokenizer.eod_id)
            think_start_token_id = 151667
            think_end_token_id = 151668

            def _run_generation_once(prompt_text: str, generator_seed: Optional[int] = None):
                messages_local = [
                    {"role": "system", "content": [{"type": "text", "text": sys_prompt_thinking}]},
                    {"role": "user", "content": []},
                ]
                content_local = messages_local[1]["content"]
                if mm_type == "video":
                    content_local.append({"type": "video", "video": sample_video_input})
                elif mm_type == "image":
                    content_local.append({"type": "image", "image": sample_video_input})
                content_local.append({"type": "text", "text": prompt_text})

                inputs_local = _build_inputs_from_messages(messages_local)

                max_generate_attempts = 3
                chosen = None
                for attempt_idx in range(max_generate_attempts):
                    generate_kwargs = dict(
                        max_new_tokens=8192,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                    )
                    # For inference-time thinking scaling, sample to explore candidates.
                    if use_thinking_scaling or attempt_idx > 0:
                        generate_kwargs.update(
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                    fork_devices = []
                    if self.current_device.type == "cuda" and torch.cuda.is_available():
                        device_index = self.current_device.index
                        if device_index is None:
                            device_index = torch.cuda.current_device()
                        fork_devices = [device_index]

                    with torch.random.fork_rng(devices=fork_devices, enabled=generator_seed is not None):
                        if generator_seed is not None:
                            torch.manual_seed(int(generator_seed))
                            if self.current_device.type == "cuda" and torch.cuda.is_available():
                                torch.cuda.manual_seed(int(generator_seed))

                        with torch.no_grad():
                            outputs_local = self.model.generate(
                                **inputs_local,
                                **generate_kwargs,
                            )

                    generated_ids = outputs_local.sequences.to(self.current_device)
                    input_len = inputs_local.input_ids.shape[1]
                    generated_ids_trimmed = generated_ids[:, input_len:]
                    last_layer_hidden_states = torch.stack(
                        [step_hidden[-1][:, -1, :].to(self.current_device) for step_hidden in outputs_local.hidden_states],
                        dim=1,
                    )

                    sample_ids = generated_ids_trimmed[0]
                    valid_len = len(sample_ids)
                    for eos_id in eos_token_ids:
                        eos_indices = (sample_ids == eos_id).nonzero(as_tuple=True)[0]
                        if len(eos_indices) > 0:
                            valid_len = min(valid_len, eos_indices[0].item())

                    valid_sample_ids = sample_ids[:valid_len]
                    valid_sample_hidden = last_layer_hidden_states[0, :valid_len, :]
                    think_start_indices = (valid_sample_ids == think_start_token_id).nonzero(as_tuple=True)[0]
                    think_end_indices = (valid_sample_ids == think_end_token_id).nonzero(as_tuple=True)[0]

                    if len(think_end_indices) > 0:
                        idx_end = think_end_indices[-1].item()
                        idx_start = think_start_indices[0].item() if len(think_start_indices) > 0 else -1
                        thinking_ids = valid_sample_ids[idx_start + 1:idx_end]
                        clean_ids = valid_sample_ids[idx_end + 1:]
                        thinking_hidden = valid_sample_hidden[idx_start + 1:idx_end, :]
                        clean_hidden = valid_sample_hidden[idx_end + 1:, :]
                    else:
                        thinking_ids = valid_sample_ids[:0]
                        clean_ids = valid_sample_ids
                        thinking_hidden = valid_sample_hidden[:0, :]
                        clean_hidden = valid_sample_hidden

                    thinking_txt = self.processor.tokenizer.decode(
                        thinking_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).replace("<think>", "").replace("</think>", "").strip()
                    clean_txt = self.processor.tokenizer.decode(
                        clean_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).replace("<think>", "").replace("</think>", "").strip()

                    need_retry_or_fallback = (not thinking_txt) or thinking_hidden.size(0) == 0
                    if not need_retry_or_fallback:
                        thinking_hidden_cpu = thinking_hidden.detach().to("cpu")
                        clean_hidden_cpu = clean_hidden.detach().to("cpu")
                        chosen = {
                            "thinking_text": thinking_txt,
                            "clean_text": clean_txt,
                            "thinking_hidden_states": thinking_hidden_cpu,
                            "clean_hidden_states": clean_hidden_cpu,
                            "attempt": attempt_idx + 1,
                        }
                        del thinking_hidden_cpu, clean_hidden_cpu
                        del outputs_local, generated_ids, generated_ids_trimmed, last_layer_hidden_states
                        del sample_ids, valid_sample_ids, valid_sample_hidden
                        del thinking_ids, clean_ids, thinking_hidden, clean_hidden
                        self._cleanup_cuda_memory()
                        break
                    if attempt_idx < max_generate_attempts - 1:
                        del outputs_local, generated_ids, generated_ids_trimmed, last_layer_hidden_states
                        del sample_ids, valid_sample_ids, valid_sample_hidden
                        del thinking_ids, clean_ids, thinking_hidden, clean_hidden
                        self._cleanup_cuda_memory()
                        print(
                            f"[Qwen3Thinking] Retry generation because empty thinking text was detected "
                            f"(attempt {attempt_idx + 1}/{max_generate_attempts}, do_sample=True, temperature=0.7, top_p=0.9)",
                            flush=True,
                        )
                        continue

                    self._log_forward_thinking_issue(
                        0,
                        "online generation produced empty thinking text; reuse clean text/hidden states as final fallback",
                        prompt=prompt_text,
                    )
                    clean_hidden_cpu = clean_hidden.detach().to("cpu")
                    chosen = {
                        "thinking_text": clean_txt,
                        "clean_text": clean_txt,
                        "thinking_hidden_states": clean_hidden_cpu,
                        "clean_hidden_states": clean_hidden_cpu,
                        "attempt": attempt_idx + 1,
                        "fallback": True,
                    }
                    del clean_hidden_cpu
                    del outputs_local, generated_ids, generated_ids_trimmed, last_layer_hidden_states
                    del sample_ids, valid_sample_ids, valid_sample_hidden
                    del thinking_ids, clean_ids, thinking_hidden, clean_hidden
                    self._cleanup_cuda_memory()
                    break

                if chosen is None:
                    raise RuntimeError("Qwen3Thinking generation failed unexpectedly.")
                del inputs_local
                self._cleanup_cuda_memory()
                chosen["prompt_used"] = prompt_text
                chosen["seed"] = generator_seed
                return chosen

            def _judge_pick_index(clean_texts: List[str]) -> Dict[str, Any]:
                judge_sys_base = (
                    "You are a strict judge for selecting the best refined video editing instruction. "
                    "You will be given the input video, the original user instruction, and multiple candidate refined instructions. "
                    "Your goal is NOT to simply describe or match the video content; it is to decide which candidate instruction is "
                    "most likely to successfully realize the original editing request when applied to THIS video by a video editing model. "
                    "Criteria (in priority order): "
                    "(1) Faithfulness: satisfies the original instruction without adding unwanted edits; "
                    "(2) Groundedness & feasibility: refers to objects/events that are actually present and identifiable in the video, "
                    "and the edit is plausible; "
                    "(3) Clarity: unambiguous, specific, and executable; "
                    "(4) Preservation: explicitly preserves non-target regions/attributes and maintains temporal consistency. "
                    "Before answering, use the <think> block to internally compare candidates concisely. "
                    "The <think> block must stay compact: at most 10 brief lines, under 300 words total. "
                    "Do NOT restate or quote the full candidate texts. Refer to candidates by number only. "
                    "Do NOT repeat the criteria verbatim. Focus only on the top differences needed to choose a winner. "
                    "All reasoning must stay inside <think>...</think>. "
                    "After the </think> block, your visible final answer must be ONLY one token group in the format \\box{K} where K is the winning candidate number (1..N)."
                )
                cand_lines = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(clean_texts)])
                user_text = (
                    f"Original instruction:\n{sample_prompt}\n\n"
                    f"Candidates:\n{cand_lines}\n\n"
                    "Think inside a compact <think>...</think> block using candidate numbers only, then after </think> return ONLY one token group in the format \\box{K}."
                )

                last_text = ""
                last_picked = 0
                max_retries = 2
                for retry in range(max_retries):
                    judge_sys = judge_sys_base
                    if retry > 0:
                        judge_sys = judge_sys_base + " If you did not follow the format, retry and make sure the answer is structured as a compact <think>...</think> block followed by ONLY \\box{K}."

                    judge_messages = [
                        {"role": "system", "content": [{"type": "text", "text": judge_sys}]},
                        {"role": "user", "content": []},
                    ]
                    judge_content = judge_messages[1]["content"]
                    if mm_type == "video":
                        judge_content.append({"type": "video", "video": sample_video_input})
                    elif mm_type == "image":
                        judge_content.append({"type": "image", "image": sample_video_input})
                    judge_content.append({"type": "text", "text": user_text})

                    judge_inputs = _build_inputs_from_messages(judge_messages)
                    with torch.no_grad():
                        judge_out = self.model.generate(
                            **judge_inputs,
                            max_new_tokens=4096,
                            do_sample=False,
                            return_dict_in_generate=False,
                        )
                    judge_ids = judge_out.to(self.current_device)
                    input_len = judge_inputs.input_ids.shape[1]
                    judge_trim = judge_ids[:, input_len:]
                    judge_text = self.processor.tokenizer.decode(
                        judge_trim[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
                    ).strip()
                    last_text = judge_text

                    picked_1based = None
                    m_box = re.search(r"\\box\{\s*(\d{1,2})\s*\}", judge_text)
                    if m_box:
                        picked_1based = int(m_box.group(1))
                    else:
                        visible_answer = judge_text
                        if "</think>" in visible_answer:
                            visible_answer = visible_answer.split("</think>", 1)[1].strip()
                        visible_answer = re.sub(r"<think>[\s\S]*?</think>", "", visible_answer).strip()
                        m_num = re.search(r"\b(\d{1,2})\b", visible_answer)
                        if m_num:
                            picked_1based = int(m_num.group(1))

                    if picked_1based is not None and 1 <= picked_1based <= len(clean_texts):
                        result = {"picked": picked_1based - 1, "raw": judge_text, "retries": retry}
                        del judge_out, judge_ids, judge_trim, judge_inputs
                        self._cleanup_cuda_memory()
                        return result

                    # parse failed or out-of-range, retry judging
                    last_picked = 0
                    del judge_out, judge_ids, judge_trim, judge_inputs
                    self._cleanup_cuda_memory()

                return {"picked": last_picked, "raw": last_text, "retries": max_retries}

            # Run inference-time thinking scaling (online only).
            if not use_thinking_scaling:
                chosen = _run_generation_once(sample_prompt, generator_seed=None)
                thinking_text = chosen["thinking_text"]
                clean_text = chosen["clean_text"]
                clean_hidden_states = chosen["clean_hidden_states"]
                infer_time_scaling_info = None
            else:
                base_seed = int(hashlib.sha1((str(sample_video_path) + "|" + str(sample_prompt)).encode("utf-8")).hexdigest()[:8], 16)
                candidates = []
                cur_prompt = sample_prompt
                steps = []
                for i in range(thinking_scaling_n):
                    cand = _run_generation_once(cur_prompt, generator_seed=base_seed + i)
                    steps.append({"idx": i, "prompt_in": cur_prompt, "clean_text": cand["clean_text"]})
                    candidates.append(cand)
                    cur_prompt = cand["clean_text"]
                    self._cleanup_cuda_memory()

                judge = _judge_pick_index([c["clean_text"] for c in candidates])
                best_idx = int(judge["picked"])
                chosen = candidates[best_idx]
                thinking_text = chosen["thinking_text"]
                clean_text = chosen["clean_text"]
                clean_hidden_states = chosen["clean_hidden_states"]
                infer_time_scaling_info = {
                    "mode": "sequential_refinement_best_of_n",
                    "n": thinking_scaling_n,
                    "steps": steps,
                    "judge": {"picked_idx": best_idx, "raw": judge.get("raw")},
                }

            thinking_char_count = len(thinking_text)
            clean_char_count = len(clean_text)
            parsed_results = [{
                "thinking_text": thinking_text,
                "clean_text": clean_text,
                "thinking_char_count": thinking_char_count,
                "clean_char_count": clean_char_count,
                "infer_time_scaling": infer_time_scaling_info,
            }]

        prompt_embeds, encoder_attention_mask = self._pack_clean_hidden_states([clean_hidden_states])
        thinking_metadata = {
            "mllm_hidden_states": [clean_hidden_states],
            "thinking_text": [thinking_text],
            "clean_text": [clean_text],
            "thinking_char_count": [thinking_char_count],
            "clean_char_count": [clean_char_count],
            "infer_time_scaling": [infer_time_scaling_info],
        }
        self.latest_forward_thinking_metadata = thinking_metadata
        self._log_forward_thinking_texts(
            [sample_prompt],
            [sample_video_path],
            parsed_results,
            mm_type,
            enable_text_save=enable_text_save,
            text_save_dir=text_save_dir,
        )
        return prompt_embeds, encoder_attention_mask, thinking_metadata
        
