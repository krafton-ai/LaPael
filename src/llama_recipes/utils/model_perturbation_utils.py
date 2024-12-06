from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from math import sqrt

import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat, reduce, rearrange
import logging

from transformers import LlamaForCausalLM, LlamaModel, LlamaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.cache_utils import StaticCache, Cache, DynamicCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

class GaussianMSELoss(nn.Module):
    def __init__(self):
        super(GaussianMSELoss, self).__init__()
        self.min_sigma = 1.0

    def forward(self, y_pred,  mu, sigma):
        # Compute the MSE loss for each dimension
        sigma = torch.maximum(sigma, torch.full(sigma.size(), self.min_sigma, device=sigma.device))
        per_feature_loss = (y_pred - mu) ** 2 / (2 * sigma ** 2)
        # Average the loss over all features and then over the batch
        loss = per_feature_loss.mean()  # This averages across all dimensions and samples
        return loss

class SymmetricKLDivLoss(nn.Module):
    """
    A nn.Module for calculating the symmetric KL divergence between two Gaussian distributions.
    """
    def __init__(self):
        super(SymmetricKLDivLoss, self).__init__()
        self.kl_divergence = KLDivergence()
        self.min_sigma = 0.1

    # sigma2 somtimes become too small
    def forward(self, mu1, sigma1, mu2, sigma2):
        """
        Forward pass to calculate symmetric KL divergence.

        Parameters:
        - mu1: Mean of the first Gaussian distribution (e.g., predicted).
        - sigma1: Standard deviation of the first Gaussian distribution (e.g., predicted).
        - mu2: Mean of the second Gaussian distribution (e.g., ground truth).
        - sigma2: Standard deviation of the second Gaussian distribution (e.g., ground truth).

        Returns:
        - sym_kl_div: The symmetric KL divergence between the two distributions.
        """
        # Ensure standard deviations are positive
        sigma2 = sigma2.to(dtype=sigma1.dtype)
        sigma1 = torch.clamp(sigma1, min=self.min_sigma)
        mask = sigma2 < self.min_sigma 
        sigma2[mask] = sigma1[mask].clone().detach()

        kl_div_1_to_2 = self.kl_divergence(mu1, sigma1, mu2, sigma2)
        kl_div_2_to_1 = self.kl_divergence(mu2, sigma2, mu1, sigma1)

        sym_kl_div = 0.5 * (kl_div_1_to_2 + kl_div_2_to_1)
        return sym_kl_div.mean()

# Assuming the KLDivergence class defined previously is available
class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu1, sigma1, mu2, sigma2):
        sigma1 = torch.abs(sigma1)
        sigma2 = torch.abs(sigma2)
        term1 = torch.log(sigma2 / sigma1)
        term2 = (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
        term3 = 0.5
        return term1 + term2 - term3

#### Llama
class CustomLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        self.layer_to_train = 1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        suffix_idx: Optional[torch.LongTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        noise_train_mode: Optional[bool] = False,
        n_augments: Optional[int] = 1,
        adversarial_embedding: Optional[torch.FloatTensor] = None,
        entity_mask: Optional[torch.FloatTensor] = None,
        valid_augs: Optional[torch.FloatTensor] = None,
        original: Optional[torch.LongTensor] = None,
        noise_mask: Optional[torch.FloatTensor] = None,
        use_neftune: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not noise_train_mode:
            n = n_augments
            batch_size = input_ids.shape[0]
            input_ids = repeat(input_ids, 'b l -> (b n) l', n=n)
            attention_mask = repeat(attention_mask, 'b l -> (b n) l', n=n)
            if labels is not None:
                labels = repeat(labels, 'b l -> (b n) l', n=n)

            if entity_mask is not None:
                entity_mask = repeat(entity_mask, 'b l -> (b n) l', n=n)

            if original is not None:
                original = repeat(original, 'b 1 -> (b n)', n=n)

            if noise_mask is not None:
                noise_mask = repeat(noise_mask, 'b l -> (b n) l', n=n)

            # Create an indicator tensor on the same device
            if n == 1:
                apply_perturb = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.float)
            else:            
                apply_perturb = torch.cat([
                    torch.zeros(1, dtype=torch.float, device=input_ids.device),
                    torch.ones((n-1), dtype=torch.float, device=input_ids.device)
                ])
                apply_perturb = torch.cat([apply_perturb.clone() for _ in range(batch_size)], dim=0)
            aug_labels = torch.randint(0, 4, (batch_size * n,)).to(device=input_ids.device)
        else:
            n = 4
            batch_size = input_ids.shape[0]
            input_ids = repeat(input_ids, 'b l -> (b n) l', n=n)
            attention_mask = repeat(attention_mask, 'b l -> (b n) l', n=n)
            labels = repeat(labels, 'b l -> (b n) l', n=n)
            suffix_idx = repeat(suffix_idx, 'b l -> (b n) l', n=n)
            apply_perturb = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.float)
            if entity_mask is not None:
                entity_mask = repeat(entity_mask, 'b l -> (b n) l', n=n)
            if valid_augs is not None:
                valid_augs = repeat(valid_augs, 'b l -> (b n) l', n=n)
            if noise_mask is not None:
                noise_mask = repeat(noise_mask, 'b l -> (b n) l', n=n)
            
            aug_labels = torch.randint(0, 4, (batch_size * n,)).to(device=input_ids.device)

        if use_neftune:
            noise_alpha = 5
            embed_init = self.model.embed_tokens(input_ids)
            dims = attention_mask.sum(-1) * embed_init.size(2)
            mag_norm = 5.0 / torch.sqrt(dims)
            z = torch.zeros_like(embed_init).uniform_(-1, 1) * mag_norm.unsqueeze(-1).unsqueeze(-1)
            inputs_embeds = embed_init + z.to(dtype=embed_init.dtype)
            input_ids = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            apply_perturb=apply_perturb,
            entity_mask=entity_mask,
            aug_labels=aug_labels,
            noise_mask=noise_mask,
        )

        hidden_states = outputs[0]

        if noise_train_mode:
            loss, logging_losses = self.train_perturbation(
                outputs, 
                suffix_idx, 
                target_embedding, 
                adversarial_embedding, 
                n=n,
                aug_labels=aug_labels,
            )
        else:
            logging_losses = None
        
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)

        
        if noise_train_mode:
            loss = (loss, logging_losses)
        else:
            lm_loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)
            loss = lm_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def train_perturbation(self, outputs, suffix_idx, target_embedding, adversarial_embedding, n=1, aug_labels=None):
        #### Loss for Perturbation Training Implementation ####
        suffix_idx = suffix_idx.squeeze()
        internal_hidden_states_list = outputs[1]
        last_hidden_states = outputs[0]

        layer_losses = []

        skip_marginalize = getattr(self.config.perturb_config, "skip_marginalize", False)
        embedding_type = getattr(self.config.perturb_config, "embedding_type", "medium")
        aug_labels = aug_labels.reshape(-1, 1, 1).expand(-1, 1, last_hidden_states.shape[-1])

        loss_type = getattr(self.config.perturb_config, "loss_type", "mse")
        if self.config.perturb_config.use_mse_loss or loss_type == "mse":
            loss_fct = nn.MSELoss()
        elif loss_type == "kl":
            loss_fct = SymmetricKLDivLoss()
        else:
            loss_fct = GaussianMSELoss()
        perturb_train_first_target_layer = getattr(
            self.config.perturb_config, "perturb_train_first_target_layer", 0
        )
        for layer_idx in range(self.layer_to_train, len(internal_hidden_states_list)):
            if layer_idx < perturb_train_first_target_layer:
                continue
            internal_hidden_states = internal_hidden_states_list[layer_idx]
            
            B = internal_hidden_states.shape[0]
            # Ensure suffix_idx is broadcastable over the sequence length dimension
            selected_hidden_states = internal_hidden_states[torch.arange(B), suffix_idx]
            if not skip_marginalize and loss_type != "kl":
                selected_hidden_states = reduce(selected_hidden_states, '(b n) d -> b d', 'mean', n=n)
                        
            internal_target_embedding = target_embedding[:, layer_idx]

            # For MSE Loss
            if self.config.perturb_config.use_mse_loss:                
                valid_batch = (internal_target_embedding.sum(-1) != 0.0)
                subset_size = internal_target_embedding.shape[1]
                selected_hidden_states = repeat(selected_hidden_states, 'b d -> b n d', n=subset_size)
                layer_loss = loss_fct(selected_hidden_states[valid_batch], internal_target_embedding[valid_batch].to(selected_hidden_states.dtype))
            else:
                if loss_type == "kl":
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()
                    internal_target_std = internal_target_std.squeeze()
                    internal_target_std[internal_target_std.isnan()] = 0.0 # Remove nan to small variance (It will be clipped)

                    selected_hidden_states = rearrange(selected_hidden_states, '(b n) d -> b n d', n=n)
                    selected_hidden_states_mean = torch.mean(selected_hidden_states, dim=1)
                    selected_hidden_states_std = torch.std(selected_hidden_states, dim=1)

                    layer_loss = loss_fct(
                        selected_hidden_states_mean,
                        selected_hidden_states_std,
                        internal_target_mean, 
                        internal_target_std, 
                    )
                elif loss_type == "mse":
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()

                    if skip_marginalize:
                        internal_target_mean = repeat(internal_target_mean, 'b d -> (b n) d', n=n)

                    layer_loss = loss_fct(
                        selected_hidden_states,
                        internal_target_mean.to(selected_hidden_states.dtype),
                    )
                else:
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()
                    internal_target_std = internal_target_std.squeeze()
                    internal_target_std[internal_target_std.isnan()] = 0.0 # Remove nan to small variance (It will be clipped)

                    if skip_marginalize:
                        internal_target_mean = repeat(internal_target_mean, 'b d -> (b n) d', n=n)
                        internal_target_std = repeat(internal_target_std, 'b d -> (b n) d', n=n)

                    layer_loss = loss_fct(
                        selected_hidden_states, 
                        internal_target_mean, 
                        internal_target_std, 
                    )
            layer_losses.append(layer_loss)
        
        total_loss = torch.sum(torch.stack(layer_losses, dim=0), dim=0)
        if total_loss.isnan():
            print("OMG") # For debug purpose
        perturb_loss = torch.mean(torch.stack(outputs.perturb_loss, dim=0), dim=0)
        loss = total_loss + perturb_loss
        return loss, (layer_losses, outputs.perturb_loss)

class CustomLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        apply_perturb: Optional[bool] = None,
        entity_mask: Optional[torch.FloatTensor] = None,
        aug_labels: Optional[torch.LongTensor] = None,
        noise_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_perturb_losses = ()
        next_decoder_cache = None

        # Training from the lower layers
        perturb_position = getattr(self.config.perturb_config, "perturb_position", 0)
        perturb_start_layer = self.config.perturb_config.perturb_start_layer
        perturb_start_idx = perturb_start_layer
        perturb_end_idx = perturb_start_layer + len(self.perturb_model.layers)
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Perturb last-n hidden states (but not the latest)
            if layer_idx >= perturb_start_idx and layer_idx < perturb_end_idx:
                perturb_layer = self.perturb_model.layers[layer_idx - perturb_start_idx]
            else:
                perturb_layer = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    perturb_layer=perturb_layer,
                    apply_perturb=apply_perturb,
                    attention_mask_original=attention_mask,
                    entity_mask=entity_mask,
                    aug_labels=aug_labels,
                    perturb_position=perturb_position,
                    noise_mask=noise_mask,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if perturb_layer is not None:
                all_perturb_losses += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        outputs.perturb_loss = all_perturb_losses
        return outputs

from transformers.models.llama.modeling_llama import (
    LLAMA_ATTENTION_CLASSES,
    LlamaMLP,
    LlamaRMSNorm
)

class CustomLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        perturb_layer = None,
        apply_perturb = None,
        attention_mask_original = None,
        entity_mask = None,
        aug_labels = None,
        perturb_position = 0,
        noise_mask = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # Position 4 (word embedding)
        if perturb_layer is not None and perturb_position == 4:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Position 1
        if perturb_layer is not None and perturb_position == 1:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        # Position 3
        if perturb_layer is not None and perturb_position == 3:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Position 0
        if perturb_layer is not None and perturb_position == 0:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )
        hidden_states = self.mlp(hidden_states)
        # Position 2
        if perturb_layer is not None and perturb_position == 2:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if perturb_layer is not None:
            outputs += (perturb_loss,)

        return outputs


from llama_recipes.utils.phi3.modeling_phi3 import (
    Phi3ForCausalLM, 
    Phi3Model,
    PHI3_ATTENTION_CLASSES,
    Phi3MLP,
    Phi3RMSNorm
)
from llama_recipes.utils.phi3.configuration_phi3 import Phi3Config

#### Phi3
class CustomPhi3ForCausalLM(Phi3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomPhi3Model(config)
        self.layer_to_train = 1

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        suffix_idx: Optional[torch.LongTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        noise_train_mode: Optional[bool] = False,
        n_augments: Optional[int] = 1,
        adversarial_embedding: Optional[torch.FloatTensor] = None,
        entity_mask: Optional[torch.FloatTensor] = None,
        valid_augs: Optional[torch.FloatTensor] = None,
        original: Optional[torch.LongTensor] = None,
        noise_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not noise_train_mode:
            n = n_augments
            batch_size = input_ids.shape[0]
            input_ids = repeat(input_ids, 'b l -> (b n) l', n=n)
            attention_mask = repeat(attention_mask, 'b l -> (b n) l', n=n)
            if labels is not None:
                labels = repeat(labels, 'b l -> (b n) l', n=n)

            if entity_mask is not None:
                entity_mask = repeat(entity_mask, 'b l -> (b n) l', n=n)

            if original is not None:
                original = repeat(original, 'b 1 -> (b n)', n=n)

            if noise_mask is not None:
                noise_mask = repeat(noise_mask, 'b l -> (b n) l', n=n)

            # Create an indicator tensor on the same device
            if n == 1:
                apply_perturb = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.float)
            else:
                apply_perturb = torch.cat([
                    torch.zeros(1, dtype=torch.float, device=input_ids.device),
                    torch.ones((n-1), dtype=torch.float, device=input_ids.device)
                ])
                apply_perturb = torch.cat([apply_perturb.clone() for _ in range(batch_size)], dim=0)
            aug_labels = torch.randint(0, 4, (batch_size * n,)).to(device=input_ids.device)
        else:
            n = 4
            batch_size = input_ids.shape[0]
            input_ids = repeat(input_ids, 'b l -> (b n) l', n=n)
            attention_mask = repeat(attention_mask, 'b l -> (b n) l', n=n)
            labels = repeat(labels, 'b l -> (b n) l', n=n)
            suffix_idx = repeat(suffix_idx, 'b l -> (b n) l', n=n)
            apply_perturb = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.float)
            if entity_mask is not None:
                entity_mask = repeat(entity_mask, 'b l -> (b n) l', n=n)
            if valid_augs is not None:
                valid_augs = repeat(valid_augs, 'b l -> (b n) l', n=n)
            if noise_mask is not None:
                noise_mask = repeat(noise_mask, 'b l -> (b n) l', n=n)
            aug_labels = torch.randint(0, 4, (batch_size * n,)).to(device=input_ids.device)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            apply_perturb=apply_perturb,
            entity_mask=entity_mask,
            aug_labels=aug_labels,
            noise_mask=noise_mask,
        )

        hidden_states = outputs[0]
        if noise_train_mode:
            loss, logging_losses = self.train_perturbation(
                outputs, 
                suffix_idx, 
                target_embedding, 
                adversarial_embedding, 
                n=n,
                aug_labels=aug_labels,
            )
        else:
            logging_losses = None
        
        logits = self.lm_head(hidden_states)

        if noise_train_mode:
            loss = (loss, logging_losses)
        else:
            lm_loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)
            loss = lm_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def train_perturbation(self, outputs, suffix_idx, target_embedding, adversarial_embedding, n=1, aug_labels=None):
        #### Loss for Perturbation Training Implementation ####
        suffix_idx = suffix_idx.squeeze()
        internal_hidden_states_list = outputs[1]
        last_hidden_states = outputs[0]
        # B, L, D = internal_hidden_states_list[0].size()

        layer_losses = []

        skip_marginalize = getattr(self.config.perturb_config, "skip_marginalize", False)
        embedding_type = getattr(self.config.perturb_config, "embedding_type", "medium")
        aug_labels = aug_labels.reshape(-1, 1, 1).expand(-1, 1, last_hidden_states.shape[-1])

        loss_type = getattr(self.config.perturb_config, "loss_type", "mse")
        if self.config.perturb_config.use_mse_loss or loss_type == "mse":
            loss_fct = nn.MSELoss()
        elif loss_type == "kl":
            loss_fct = SymmetricKLDivLoss()
        else:
            loss_fct = GaussianMSELoss()
        perturb_train_first_target_layer = getattr(
            self.config.perturb_config, "perturb_train_first_target_layer", 0
        )
        for layer_idx in range(self.layer_to_train, len(internal_hidden_states_list)):
            if layer_idx < perturb_train_first_target_layer:
                continue
            internal_hidden_states = internal_hidden_states_list[layer_idx]
            
            B = internal_hidden_states.shape[0]
            # Ensure suffix_idx is broadcastable over the sequence length dimension
            selected_hidden_states = internal_hidden_states[torch.arange(B), suffix_idx]
            if not skip_marginalize and loss_type != "kl":
                selected_hidden_states = reduce(selected_hidden_states, '(b n) d -> b d', 'mean', n=n)
                        
            internal_target_embedding = target_embedding[:, layer_idx]
            # print(internal_target_embedding.shape)

            # For MSE Loss
            if self.config.perturb_config.use_mse_loss:                
                valid_batch = (internal_target_embedding.sum(-1) != 0.0)
                subset_size = internal_target_embedding.shape[1]
                selected_hidden_states = repeat(selected_hidden_states, 'b d -> b n d', n=subset_size)
                layer_loss = loss_fct(selected_hidden_states[valid_batch], internal_target_embedding[valid_batch].to(selected_hidden_states.dtype))
            else:
                if loss_type == "kl":
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()
                    internal_target_std = internal_target_std.squeeze()
                    internal_target_std[internal_target_std.isnan()] = 0.0 # Remove nan to small variance (It will be clipped)

                    selected_hidden_states = rearrange(selected_hidden_states, '(b n) d -> b n d', n=n)
                    selected_hidden_states_mean = torch.mean(selected_hidden_states, dim=1)
                    selected_hidden_states_std = torch.std(selected_hidden_states, dim=1)

                    layer_loss = loss_fct(
                        selected_hidden_states_mean,
                        selected_hidden_states_std,
                        internal_target_mean, 
                        internal_target_std, 
                    )
                elif loss_type == "mse":
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()

                    if skip_marginalize:
                        internal_target_mean = repeat(internal_target_mean, 'b d -> (b n) d', n=n)

                    layer_loss = loss_fct(
                        selected_hidden_states,
                        internal_target_mean.to(selected_hidden_states.dtype),
                    )
                else:
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()
                    internal_target_std = internal_target_std.squeeze()
                    internal_target_std[internal_target_std.isnan()] = 0.0 # Remove nan to small variance (It will be clipped)

                    if skip_marginalize:
                        internal_target_mean = repeat(internal_target_mean, 'b d -> (b n) d', n=n)
                        internal_target_std = repeat(internal_target_std, 'b d -> (b n) d', n=n)

                    layer_loss = loss_fct(
                        selected_hidden_states, 
                        internal_target_mean, 
                        internal_target_std, 
                    )
            layer_losses.append(layer_loss)
        
        total_loss = torch.sum(torch.stack(layer_losses, dim=0), dim=0)
        if total_loss.isnan():
            print("OMG") # For debug purpose
       
        perturb_loss = torch.mean(torch.stack(outputs.perturb_loss, dim=0), dim=0)
        loss = total_loss + perturb_loss.to(total_loss.dtype)
        return loss, (layer_losses, outputs.perturb_loss)

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
class CustomPhi3Model(Phi3Model):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomPhi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        apply_perturb: Optional[bool] = None,
        entity_mask: Optional[torch.FloatTensor] = None,
        aug_labels: Optional[torch.LongTensor] = None,
        noise_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            causal_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            causal_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_perturb_losses = ()
        next_decoder_cache = None

        # Training from the lower layers
        perturb_position = getattr(self.config.perturb_config, "perturb_position", 0)
        perturb_start_layer = self.config.perturb_config.perturb_start_layer
        perturb_start_idx = perturb_start_layer
        perturb_end_idx = perturb_start_layer + len(self.perturb_model.layers)
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Perturb last-n hidden states (but not the latest)
            if layer_idx >= perturb_start_idx and layer_idx < perturb_end_idx:
                perturb_layer = self.perturb_model.layers[layer_idx - perturb_start_idx]
            else:
                perturb_layer = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    perturb_layer=perturb_layer,
                    apply_perturb=apply_perturb,
                    attention_mask_original=attention_mask,
                    entity_mask=entity_mask,
                    aug_labels=aug_labels,
                    perturb_position=perturb_position,
                    noise_mask=noise_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if perturb_layer is not None:
                all_perturb_losses += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        outputs.perturb_loss = all_perturb_losses
        return outputs

class CustomPhi3DecoderLayer(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.self_attn = PHI3_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx=layer_idx)

        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        perturb_layer = None,
        apply_perturb = None,
        attention_mask_original = None,
        entity_mask = None,
        aug_labels = None,
        perturb_position = 0,
        noise_mask = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        # Position 4 (word embedding)
        if perturb_layer is not None and perturb_position == 4:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Position 1
        if perturb_layer is not None and perturb_position == 1:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Position 3
        if perturb_layer is not None and perturb_position == 3:
            attn_outputs, perturb_loss = perturb_layer(
                attn_outputs, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # Position 0
        if perturb_layer is not None and perturb_position == 0:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )
        hidden_states = self.mlp(hidden_states)
        # Position 2
        if perturb_layer is not None and perturb_position == 2:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if perturb_layer is not None:
            outputs += (perturb_loss,)

        return outputs

from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM, 
    MistralModel,
    MISTRAL_ATTENTION_CLASSES,
    MistralMLP,
    MistralRMSNorm
)
from transformers.models.mistral.configuration_mistral import MistralConfig

#### Mistral
class CustomMistralForCausalLM(MistralForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomMistralModel(config)
        self.layer_to_train = 1
       
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        suffix_idx: Optional[torch.LongTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        noise_train_mode: Optional[bool] = False,
        n_augments: Optional[int] = 1,
        adversarial_embedding: Optional[torch.FloatTensor] = None,
        entity_mask: Optional[torch.FloatTensor] = None,
        valid_augs: Optional[torch.FloatTensor] = None,
        original: Optional[torch.LongTensor] = None,
        noise_mask: Optional[torch.FloatTensor] = None,
        use_neftune: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MistralForCausalLM

        >>> model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not noise_train_mode:
            n = n_augments
            batch_size = input_ids.shape[0]
            input_ids = repeat(input_ids, 'b l -> (b n) l', n=n)
            attention_mask = repeat(attention_mask, 'b l -> (b n) l', n=n)
            if labels is not None:
                labels = repeat(labels, 'b l -> (b n) l', n=n)

            if entity_mask is not None:
                entity_mask = repeat(entity_mask, 'b l -> (b n) l', n=n)

            if original is not None:
                original = repeat(original, 'b 1 -> (b n)', n=n)

            if noise_mask is not None:
                noise_mask = repeat(noise_mask, 'b l -> (b n) l', n=n)

            # Create an indicator tensor on the same device
            if n == 1:
                apply_perturb = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.float)
            else:
                apply_perturb = torch.cat([
                    torch.zeros(1, dtype=torch.float, device=input_ids.device),
                    torch.ones((n-1), dtype=torch.float, device=input_ids.device)
                ])
                apply_perturb = torch.cat([apply_perturb.clone() for _ in range(batch_size)], dim=0)
            aug_labels = torch.randint(0, 4, (batch_size * n,)).to(device=input_ids.device)
        else:
            n = 4
            batch_size = input_ids.shape[0]
            input_ids = repeat(input_ids, 'b l -> (b n) l', n=n)
            attention_mask = repeat(attention_mask, 'b l -> (b n) l', n=n)
            labels = repeat(labels, 'b l -> (b n) l', n=n)
            suffix_idx = repeat(suffix_idx, 'b l -> (b n) l', n=n)
            apply_perturb = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.float)
            if entity_mask is not None:
                entity_mask = repeat(entity_mask, 'b l -> (b n) l', n=n)
            if valid_augs is not None:
                valid_augs = repeat(valid_augs, 'b l -> (b n) l', n=n)
            if noise_mask is not None:
                noise_mask = repeat(noise_mask, 'b l -> (b n) l', n=n)
            aug_labels = torch.randint(0, 4, (batch_size * n,)).to(device=input_ids.device)

        if use_neftune:
            ### NEFTune
            noise_alpha = 5
            embed_init = self.model.embed_tokens(input_ids)
            dims = attention_mask.sum(-1) * embed_init.size(2)
            mag_norm = 5.0 / torch.sqrt(dims)
            z = torch.zeros_like(embed_init).uniform_(-1, 1) * mag_norm.unsqueeze(-1).unsqueeze(-1)
            inputs_embeds = embed_init + z.to(dtype=embed_init.dtype)
            input_ids = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
            apply_perturb=apply_perturb,
            entity_mask=entity_mask,
            aug_labels=aug_labels,
            noise_mask=noise_mask,
        )

        hidden_states = outputs[0]

        if noise_train_mode:
            loss, logging_losses = self.train_perturbation(
                outputs, 
                suffix_idx, 
                target_embedding, 
                adversarial_embedding, 
                n=n,
                aug_labels=aug_labels,
            )
        else:
            logging_losses = None

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if noise_train_mode:
            loss = (loss, logging_losses)
        else:
            lm_loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Ensure tensors are on the same device
                shift_labels = shift_labels.to(shift_logits.device)
                loss_fct = CrossEntropyLoss()
                lm_loss = loss_fct(shift_logits, shift_labels)
            loss = lm_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def train_perturbation(self, outputs, suffix_idx, target_embedding, adversarial_embedding, n=1, aug_labels=None):
        #### Loss for Perturbation Training Implementation ####
        suffix_idx = suffix_idx.squeeze()
        internal_hidden_states_list = outputs[1]
        last_hidden_states = outputs[0]
        # B, L, D = internal_hidden_states_list[0].size()

        layer_losses = []

        skip_marginalize = getattr(self.config.perturb_config, "skip_marginalize", False)
        embedding_type = getattr(self.config.perturb_config, "embedding_type", "medium")
        aug_labels = aug_labels.reshape(-1, 1, 1).expand(-1, 1, last_hidden_states.shape[-1])

        loss_type = getattr(self.config.perturb_config, "loss_type", "mse")
        if self.config.perturb_config.use_mse_loss or loss_type == "mse":
            loss_fct = nn.MSELoss()
        elif loss_type == "kl":
            loss_fct = SymmetricKLDivLoss()
        else:
            loss_fct = GaussianMSELoss()
        perturb_train_first_target_layer = getattr(
            self.config.perturb_config, "perturb_train_first_target_layer", 0
        )
        for layer_idx in range(self.layer_to_train, len(internal_hidden_states_list)):
            if layer_idx < perturb_train_first_target_layer:
                continue
            internal_hidden_states = internal_hidden_states_list[layer_idx]
            
            B = internal_hidden_states.shape[0]
            # Ensure suffix_idx is broadcastable over the sequence length dimension
            selected_hidden_states = internal_hidden_states[torch.arange(B), suffix_idx]
            if not skip_marginalize and loss_type != "kl":
                selected_hidden_states = reduce(selected_hidden_states, '(b n) d -> b d', 'mean', n=n)
                        
            internal_target_embedding = target_embedding[:, layer_idx]

            # For MSE Loss
            if self.config.perturb_config.use_mse_loss:                
                valid_batch = (internal_target_embedding.sum(-1) != 0.0)
                subset_size = internal_target_embedding.shape[1]
                selected_hidden_states = repeat(selected_hidden_states, 'b d -> b n d', n=subset_size)
                layer_loss = loss_fct(selected_hidden_states[valid_batch], internal_target_embedding[valid_batch].to(selected_hidden_states.dtype))
            else:
                if loss_type == "kl":
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()
                    internal_target_std = internal_target_std.squeeze()
                    internal_target_std[internal_target_std.isnan()] = 0.0 # Remove nan to small variance (It will be clipped)

                    selected_hidden_states = rearrange(selected_hidden_states, '(b n) d -> b n d', n=n)
                    selected_hidden_states_mean = torch.mean(selected_hidden_states, dim=1)
                    selected_hidden_states_std = torch.std(selected_hidden_states, dim=1)

                    layer_loss = loss_fct(
                        selected_hidden_states_mean,
                        selected_hidden_states_std,
                        internal_target_mean, 
                        internal_target_std, 
                    )
                elif loss_type == "mse":
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()

                    if skip_marginalize:
                        internal_target_mean = repeat(internal_target_mean, 'b d -> (b n) d', n=n)

                    layer_loss = loss_fct(
                        selected_hidden_states,
                        internal_target_mean.to(selected_hidden_states.dtype),
                    )
                else:
                    internal_target_mean, internal_target_std = internal_target_embedding.split(dim=1, split_size=1)
                    internal_target_mean = internal_target_mean.squeeze()
                    internal_target_std = internal_target_std.squeeze()
                    internal_target_std[internal_target_std.isnan()] = 0.0 # Remove nan to small variance (It will be clipped)

                    if skip_marginalize:
                        internal_target_mean = repeat(internal_target_mean, 'b d -> (b n) d', n=n)
                        internal_target_std = repeat(internal_target_std, 'b d -> (b n) d', n=n)

                    layer_loss = loss_fct(
                        selected_hidden_states, 
                        internal_target_mean, 
                        internal_target_std, 
                    )
            layer_losses.append(layer_loss)
        
        total_loss = torch.sum(torch.stack(layer_losses, dim=0), dim=0)
        if total_loss.isnan():
            print("OMG") # For debug purpose

        perturb_loss = torch.mean(torch.stack(outputs.perturb_loss, dim=0), dim=0)
        loss = total_loss + perturb_loss

        return loss, (layer_losses, outputs.perturb_loss)

class CustomMistralModel(MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomMistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        apply_perturb: Optional[bool] = None,
        entity_mask: Optional[torch.FloatTensor] = None,
        aug_labels: Optional[torch.LongTensor] = None,
        noise_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            return_legacy_cache = True
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, use_cache, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_perturb_losses = ()
        next_decoder_cache = None

        perturb_position = getattr(self.config.perturb_config, "perturb_position", 0)
        perturb_start_layer = self.config.perturb_config.perturb_start_layer
        perturb_start_idx = perturb_start_layer
        perturb_end_idx = perturb_start_layer + len(self.perturb_model.layers)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Perturb last-n hidden states (but not the latest)
            if layer_idx >= perturb_start_idx and layer_idx < perturb_end_idx:
                perturb_layer = self.perturb_model.layers[layer_idx - perturb_start_idx]
            else:
                perturb_layer = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    perturb_layer=perturb_layer,
                    apply_perturb=apply_perturb,
                    attention_mask_original=attention_mask,
                    entity_mask=entity_mask,
                    aug_labels=aug_labels,
                    perturb_position=perturb_position,
                    noise_mask=noise_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if perturb_layer is not None:
                all_perturb_losses += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        outputs.perturb_loss = all_perturb_losses
        return outputs

class CustomMistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MISTRAL_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        perturb_layer = None,
        apply_perturb = None,
        attention_mask_original = None,
        entity_mask = None,
        aug_labels = None,
        perturb_position = 0,
        noise_mask = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        # Position 4 (word embedding)
        if perturb_layer is not None and perturb_position == 4:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Position 1
        if perturb_layer is not None and perturb_position == 1:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # Position 3
        if perturb_layer is not None and perturb_position == 3:
            attn_outputs, perturb_loss = perturb_layer(
                attn_outputs, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Position 0
        if perturb_layer is not None and perturb_position == 0:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        hidden_states = self.mlp(hidden_states)

        # Position 2
        if perturb_layer is not None and perturb_position == 2:
            hidden_states, perturb_loss = perturb_layer(
                hidden_states, 
                apply_perturb, 
                attention_mask=attention_mask_original,
                entity_mask=entity_mask,
                aug_labels=aug_labels,
                noise_mask=noise_mask,
            )

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if perturb_layer is not None:
            outputs += (perturb_loss,)

        return outputs