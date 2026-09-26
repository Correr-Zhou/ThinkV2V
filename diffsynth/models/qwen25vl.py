import torch
from qwen_vl_utils import process_vision_info
import random


class Qwen25VL_7b_Video_Embedder(torch.nn.Module):
    def __init__(self, model_path, dtype=torch.bfloat16, device="cuda"):
        super(Qwen25VL_7b_Video_Embedder, self).__init__()
        self.dtype = dtype
        self.device = device
        
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )

        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
        )

        self.txt_sys_prompt = "First, thoroughly analyze the user's text description to deconstruct its key components: the primary subject(s), the background and environment, the specified composition (e.g., close-up, wide shot), the lighting conditions (e.g., golden hour, neon lights), the color palette, and the desired artistic style (e.g., photorealistic, oil painting, anime). Based on this comprehensive analysis, construct a detailed internal 'scene blueprint'. Finally, generate a high-fidelity and artistically coherent image or video that accurately brings this blueprint to life, ensuring all aspects of the user's request are visually represented."
        self.img_sys_prompt = "First, analyze the input image's key visual features, including its color palette, lighting, composition, key objects, and background. Based on this analysis, interpret the user's text instruction to create a clear plan for modification. This plan should specify exactly what visual elements to alter (e.g., color grading, object removal, style transfer, background replacement, adding visual effects), and how the edits should be executed. Finally, generate a new image that precisely implements the requested modifications."
        self.vid_sys_prompt = "First, analyze the input video's key visual and temporal features, including its color palette, lighting, composition, key objects, background, and subject motion. Based on this analysis, interpret the user's text instruction to create a clear plan for modification. This plan should specify exactly what visual elements to alter (e.g., color grading, object removal, adding visual effects), and how the edits should be executed. Finally, generate a new video that precisely implements the requested modifications."
        self.text_drop_idx = 130
        self.image_drop_idx = 109
        self.video_drop_idx = 108

    @staticmethod
    def from_pretrained(path, torch_dtype=torch.bfloat16, device="cuda"):
        return Qwen25VL_7b_Video_Embedder(path, dtype=torch_dtype, device=device)

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def forward(self, prompt, video_path, mm_type="video"):

        if mm_type == "video":
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.vid_sys_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

            if 'fps' in video_kwargs and isinstance(video_kwargs['fps'], list):
                if len(video_kwargs['fps']) > 0:
                    video_kwargs['fps'] = video_kwargs['fps'][0]
                else:
                    video_kwargs['fps'] = None 

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(self.device)

            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values_videos=inputs.pixel_values_videos,
                video_grid_thw=inputs.video_grid_thw,
                output_hidden_states=True,
            )

            hidden_states = outputs["hidden_states"][-1]

            split_hidden_states = self.extract_masked_hidden(hidden_states, inputs.attention_mask)
            split_hidden_states = [e[self.video_drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
            return prompt_embeds, encoder_attention_mask
        elif mm_type == "image":
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.img_sys_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": video_path},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.pixel_values,
                image_grid_thw=inputs.image_grid_thw,
                output_hidden_states=True,
            )

            hidden_states = outputs["hidden_states"][-1]

            split_hidden_states = self.extract_masked_hidden(hidden_states, inputs.attention_mask)
            split_hidden_states = [e[self.image_drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
            return prompt_embeds, encoder_attention_mask
        else:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.txt_sys_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )

            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )

            hidden_states = outputs["hidden_states"][-1]

            split_hidden_states = self.extract_masked_hidden(hidden_states, inputs.attention_mask)
            split_hidden_states = [e[self.text_drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)
            return prompt_embeds, encoder_attention_mask
