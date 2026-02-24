# -*- coding: utf-8 -*-
#
# @author: Yang Liu <yangliu.real@gmail.com>
# @date: 2024/04/03

import re
import logging
from typing import Any, List, Union, Optional

import torch
from transformers import AutoModelForCausalLM
from mergoo.models.modeling_llama import LlamaForCausalLM as MoELlamaForCausalLM
from mergoo.models.modeling_mistral import MistralForCausalLM as MoEMistralForCausalLM


logger = logging.getLogger(__name__)

LMLexiconModel = Union[AutoModelForCausalLM, MoELlamaForCausalLM, MoEMistralForCausalLM]


def unfreeze_all_layers(
    model: LMLexiconModel,
) -> LMLexiconModel:
    """
    Unfreeze all the layers of the model

    Args:
        model: the model to be trained

    Returns:
        model: the model with unfrozen setting
    """
    for _, param in model.named_parameters():
        param.requires_grad = True

    return model


def optimize_specific_layers(
    model: LMLexiconModel,
    optim_layer_names: Optional[List[str]] = None,
) -> LMLexiconModel:
    """
    Optimize specific layers of the model according to the given `optim_layer_names`

    Args:
        model: the model to be trained
        optim_layer_names: the layers that need to be trained

    Returns:
        model: the model with frozen setting

    Example:
        - If you want to freeze all gate module of experts
        You need to set the optim_layers as: ["gate"]

        - If you want to freeze all expert modules
        You need to set the optim_layers as: ["all"]

    May the force be with you!
    """

    def display_all_model_layers(
        model: Union[AutoModelForCausalLM, MoELlamaForCausalLM, MoEMistralForCausalLM],
        only_display_trainable: bool = False,
    ) -> None:
        for name, param in model.named_parameters():
            if only_display_trainable and param.requires_grad:
                print(f"Weight name: {name}\tRequire Gradient: {param.requires_grad}")
            elif not only_display_trainable:
                print(f"Weight name: {name}\tRequire Gradient: {param.requires_grad}")
            # logger.log(
            # level=logging.INFO,
            # msg=f"Weight name: {name}\tRequire Gradient: {param.requires_grad}",
            # )

    def hook_dense_model_layers(
        hook_mode: str,
        model: AutoModelForCausalLM,
    ) -> List[torch.nn.parameter.Parameter]:
        raise NotImplementedError(
            "The model class `AutoModelForCausalLM` is not supported for layers freezing."
        )

    def hook_sparse_model_layers(
        model: MoELlamaForCausalLM,
        hook_mode: str,
    ) -> List[torch.nn.parameter.Parameter]:
        """
        Hook the layers of the sparse model based on the hook_mode
        One or many options can be selected from the following:
        - lm_head: the last layer of the model
        - gate: all the gate module of the experts
        - expert: all the expert modules
        - attention: all the attention modules
        - norm: all the norm modules
        - token_embedding: the token embedding module
        - all: all the layers

        Args:
            hook_mode: the mode to freeze the layers
            model: the model to be trained

        Returns:
            layers_hooked: the layers that have been hooked
        """
        if isinstance(model, MoEMistralForCausalLM):
            raise NotImplementedError(
                "The model class `MoEMistralForCausalLM` is not supported for freezing layers"
            )

        if hook_mode == "lm_head":
            pattern = r"lm_head\.weight"
        elif hook_mode == "gate":
            pattern = (
                r"model.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.gate\.weight"
            )
        elif hook_mode == "ffn":
            pattern = r"model.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.experts\.\d+\.weight"
        elif hook_mode == "attention":
            pattern = (
                r"model.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight"
            )
        elif hook_mode == "norm":
            pattern = r"model\.(layers\.\d+\.(input_layernorm|post_attention_layernorm)|norm)\.weight"
        elif hook_mode == "token_embedding":
            pattern = r"embed_tokens\.weight"
        elif hook_mode == "all":
            pattern = r"(model\..+|lm_head)\.weight"
        else:
            raise ValueError(f"Unimplemented freeze mode: {hook_mode}")

        layers_hooked = []
        for name, param in model.named_parameters():
            if re.match(pattern, name):
                layers_hooked.append(name)

        return layers_hooked

    if optim_layer_names is None or "all" in optim_layer_names:
        return model

    if isinstance(model, AutoModelForCausalLM):
        raise NotImplementedError(
            "The model class `AutoModelForCausalLM` is not supported for freezing layers"
        )
    elif isinstance(model, MoELlamaForCausalLM):
        # 0. LM head module 🔥
        #
        # lm_head.weight
        # 1. Gate modules 🔥
        # layers.*.mlp.gate_proj.gate.weight
        # layers.*.mlp.up_proj.gate.weight
        # layers.*.mlp.down_proj.gate.weight
        #
        # 2. Expert modules 🔥
        # layers.*.mlp.gate_proj.experts.*.weight
        # layers.*.mlp.up_proj.experts.*.weight
        # layers.*.mlp.down_proj.experts.*.weight
        #
        # 3. Attention modules 🔥
        #
        # layers.*.self_attn.q_proj.weight
        # layers.*.self_attn.k_proj.weight
        # layers.*.self_attn.v_proj.weight
        # layers.*.self_attn.o_proj.weight
        #
        # 4. LayerNorm / Norm modules ❄️
        # layers.*.input_layernorm.weight
        # layers.*.post_attention_layernorm.weight
        # norm.weight
        #
        # 5. Token Embedding module 🔥
        # embed_tokens.weight
        hooked_layers_optim = []
        for layer_name in optim_layer_names:
            logger.info(f"Optimizing module: {layer_name}")
            hooked_layers_optim += hook_sparse_model_layers(model, layer_name)

        # freeze all layers
        for _, param in model.named_parameters():
            param.requires_grad = False

        # unfreeze the layers that need to be optimized
        for name, param in model.named_parameters():
            if name in hooked_layers_optim:
                param.requires_grad = True
    elif isinstance(model, MoEMistralForCausalLM):
        raise NotImplementedError(
            "The model class `MoEMistralForCausalLM` is not supported for freezing layers"
        )

    display_all_model_layers(model, only_display_trainable=True)

    return model
