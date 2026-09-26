import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
import copy

class Qwen25VLConnectorStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        # Infer constructor kwargs from checkpoint shapes to avoid size-mismatch
        # when loading connectors trained with different Qwen(VL) hidden sizes
        # (e.g. 2048 vs 4096).
        extra_kwargs = {}
        w = state_dict.get("fc.0.weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            # nn.Linear(out_features, in_features)
            extra_kwargs["hidden_size"] = int(w.shape[0])
            extra_kwargs["llm_hidden_size"] = int(w.shape[1])
        return (state_dict, extra_kwargs) if extra_kwargs else state_dict

    def from_civitai(self, state_dict):
        state_dict_ = {}
        for name, param in state_dict.items():
            if name.startswith("pipe.qwen25vl_connector."):
                name_ = name[len("pipe.qwen25vl_connector."):]
                state_dict_[name_] = param
        extra_kwargs = {}
        w = state_dict_.get("fc.0.weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            extra_kwargs["hidden_size"] = int(w.shape[0])
            extra_kwargs["llm_hidden_size"] = int(w.shape[1])
        return (state_dict_, extra_kwargs) if extra_kwargs else state_dict_


class SparseMoE(nn.Module):
    def __init__(self,
                 d_model,
                 d_ffn,
                 num_experts,
                 num_selected,
                 dtype=torch.bfloat16,
                 load_balance_alpha=1e-1,
                 router_z_alpha=1e-2,
                ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_experts = num_experts
        self.num_selected = num_selected
        self.load_balance_alpha = load_balance_alpha
        self.router_z_alpha = router_z_alpha

        self.gate = nn.Linear(d_model, num_experts, bias=False, dtype=dtype)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ffn, bias=True, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(d_ffn, d_model, bias=True, dtype=dtype)
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, d_model = x.shape
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.size(0)

        gate_logits = self.gate(x_flat)
        router_z_loss = torch.logsumexp(gate_logits.to(torch.float32), dim=-1).mean().square()
        router_z_loss = self.router_z_alpha * router_z_loss

        gate_probs = F.softmax(gate_logits.to(torch.float32), dim=-1)
        f_i = gate_probs.mean(dim=0)
        _, selected_indices = torch.topk(gate_probs, self.num_selected, dim=-1)
        expert_mask = F.one_hot(selected_indices, self.num_experts).sum(dim=1)
        p_i = expert_mask.float().mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(p_i * f_i)
        load_balance_loss = self.load_balance_alpha * load_balance_loss

        dummy_loss = 0.0
        for expert in self.experts:
            for p in expert.parameters():
                dummy_loss += p.sum() * 0.0

        total_aux_loss = router_z_loss + load_balance_loss + dummy_loss
        weights, selected_experts = torch.topk(gate_probs.to(x.dtype), self.num_selected, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)

        final_output = torch.zeros_like(x_flat)
        flat_expert_indices = selected_experts.flatten()
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.num_selected)
        flat_x_for_experts = x_flat[flat_token_indices]

        for i, expert in enumerate(self.experts):
            mask = (flat_expert_indices == i)
            if mask.sum() == 0:
                continue
            tokens_for_expert = flat_x_for_experts[mask]
            expert_output = expert(tokens_for_expert)
            weights_for_expert = weights.flatten()[mask].unsqueeze(1)
            final_output.scatter_add_(0, flat_token_indices[mask].unsqueeze(1).expand_as(expert_output), expert_output * weights_for_expert)

        final_output = final_output.view(batch_size, sequence_length, d_model)
        return final_output, total_aux_loss

class TransformerMoEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, num_experts, num_selected, dropout=0.1, norm_first=True, dtype=torch.bfloat16):
        super().__init__()
        assert norm_first, "This implementation requires norm_first=True"
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, dtype=dtype)
        self.moe_ffn = SparseMoE(d_model, d_ffn, num_experts, num_selected, dtype=dtype)
        self.norm1 = nn.LayerNorm(d_model, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = self.norm1(src)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        x = self.norm2(src)
        ffn_output, aux_loss = self.moe_ffn(x)
        src = src + self.dropout2(ffn_output)
        return src, aux_loss

class TransformerMoEDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn, num_experts, num_selected, dropout=0.1, norm_first=True, dtype=torch.bfloat16):
        super().__init__()
        assert norm_first, "This implementation requires norm_first=True"
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, dtype=dtype)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, dtype=dtype)
        self.moe_ffn = SparseMoE(d_model, d_ffn, num_experts, num_selected, dtype=dtype)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        x = self.norm1(tgt)
        self_attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(self_attn_output)

        x = self.norm2(tgt)
        cross_attn_output, _ = self.multihead_attn(x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(cross_attn_output)

        x = self.norm3(tgt)
        ffn_output, aux_loss = self.moe_ffn(x)
        tgt = tgt + self.dropout3(ffn_output)

        return tgt, aux_loss

class TransformerMoEEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        total_aux_loss = 0.0
        for mod in self.layers:
            output, aux_loss = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            total_aux_loss += aux_loss
        if self.norm is not None:
            output = self.norm(output)
        return output, total_aux_loss

class TransformerMoEDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        total_aux_loss = 0.0
        for mod in self.layers:
            output, aux_loss = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            total_aux_loss += aux_loss
        if self.norm is not None:
            output = self.norm(output)
        return output, total_aux_loss

class Qwen25VLConnector(nn.Module):
    def __init__(self, llm_hidden_size=2048, hidden_size=3072,
                 learnable_query_length=512,
                 nhead=24, num_encoder_layers=2, num_decoder_layers=2,
                 num_experts=6, num_selected=2, dropout=0.0, dtype=torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.learnable_query_length = learnable_query_length
        d_ffn = hidden_size * 4

        self.fc = nn.Sequential(
                    nn.Linear(llm_hidden_size, self.hidden_size, dtype=dtype),
                    nn.GELU(),
                    nn.Linear(self.hidden_size, self.hidden_size, dtype=dtype),
                )

        self.decoder_query = nn.Parameter(
                                torch.randn((1, self.learnable_query_length, self.hidden_size),
                                dtype=dtype
                                ),
                                requires_grad=True
                            )

        moe_encoder_layer = TransformerMoEEncoderLayer(
            d_model=self.hidden_size, nhead=nhead, d_ffn=d_ffn,
            num_experts=num_experts, num_selected=num_selected, dropout=dropout, norm_first=True, dtype=dtype
        )
        encoder_norm = nn.LayerNorm(self.hidden_size, dtype=dtype)
        self.encoder = TransformerMoEEncoder(moe_encoder_layer, num_encoder_layers, norm=encoder_norm)

        moe_decoder_layer = TransformerMoEDecoderLayer(
            d_model=self.hidden_size, nhead=nhead, d_ffn=d_ffn,
            num_experts=num_experts, num_selected=num_selected, dropout=dropout, norm_first=True, dtype=dtype
        )
        decoder_norm = nn.LayerNorm(self.hidden_size, dtype=dtype)
        self.decoder = TransformerMoEDecoder(moe_decoder_layer, num_decoder_layers, norm=decoder_norm)

        self.output_zero_linear = nn.Linear(self.hidden_size, self.hidden_size, bias=True, dtype=dtype)
        nn.init.zeros_(self.output_zero_linear.weight)
        nn.init.zeros_(self.output_zero_linear.bias)

        self.to(dtype)

    def forward(self, vlm_last_hidden_state):
        src = self.fc(vlm_last_hidden_state)
        batch_size = src.shape[0]
        memory, encoder_aux_loss = self.encoder(src)
        tgt = self.decoder_query.repeat(batch_size, 1, 1)
        vision_tokens, decoder_aux_loss = self.decoder(tgt=tgt, memory=memory)
        vision_tokens = self.output_zero_linear(vision_tokens)

        total_aux_loss = encoder_aux_loss + decoder_aux_loss
        return vision_tokens, total_aux_loss

    @staticmethod
    def state_dict_converter():
        return Qwen25VLConnectorStateDictConverter()
