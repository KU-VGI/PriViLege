# PriViLege prompts on the official open_clip CLIP vision encoder (paper Figure 3).
#
# The authors' unreleased models/CLIP_model.py was a copy of OpenAI's CLIP model.py with the prompts added.
# Here the official open_clip modules and LAION-2B weights are used unchanged (patch embedding, the 12
# residual blocks, ln_pre / ln_post, projection); only the method is added on top, following the authors'
# compiled code (models/__pycache__/CLIP_model.cpython-39.pyc):
#   * VL-Prompt: 2 tokens inserted after [CLS] (their outputs are the Vision and Language features)
#   * B-Prompt in blocks 0-1: one vector per block, prepended to the keys and to the values, each scaled by a
#     1x1 conv (key_comp / val_comp) applied to the block's frozen key / value projection weights
#   * ln_post and the projection are applied to every token, so callers can read tokens 0, 1 and 2.
# Difference from the authors' copy: open_clip's ViT-B-16 uses GELU, the activation the LAION-2B weights were
# trained with; the OpenAI copy used QuickGELU.
import torch
from torch import nn


class PromptedVisionTransformer(nn.Module):
    def __init__(self, visual, prompt_length=2, adapt_layer=(0, 1)):
        super().__init__()
        assert visual.transformer.batch_first, 'expects open_clip >= 2.24 (batch-first transformer)'
        width = visual.class_embedding.shape[0]
        # official open_clip modules (weights included)
        self.conv1 = visual.conv1
        self.class_embedding = visual.class_embedding
        self.positional_embedding = visual.positional_embedding
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        # PriViLege additions (names as in the authors' code, which the trainer relies on)
        self.transformer.adapt_layer = list(adapt_layer)
        self.transformer.key_comp = nn.Conv1d(width, 1, kernel_size=1)
        self.transformer.val_comp = nn.Conv1d(width, 1, kernel_size=1)
        self.prompt_length = prompt_length
        self.expert_length = len(adapt_layer)
        self.prompt = nn.Parameter(torch.randn(self.prompt_length, width))          # VL-Prompt
        self.expert_prompt = nn.Parameter(torch.randn(self.expert_length, width))   # B-Prompt
        nn.init.uniform_(self.prompt, -1, 1)
        nn.init.uniform_(self.expert_prompt, -1, 1)

    def _prompted_block(self, block, x, b_prompt):
        # x: (B, N, C). Same computation as the authors' ResidualAttentionBlock.attention, batch-first.
        B, N, C = x.shape
        t = self.transformer
        q = block.ln_1(x)
        w = block.attn.in_proj_weight.data
        k_var = t.key_comp(w[C:2 * C].detach()).expand(B, 1, C)
        v_var = t.val_comp(w[2 * C:].detach()).expand(B, 1, C)
        prompt = b_prompt.view(1, 1, C).expand(B, 1, C)   # one vector per block, used for key and value
        k = torch.cat([prompt * k_var, q], dim=1)
        v = torch.cat([prompt * v_var, q], dim=1)
        x = x + block.ls_1(block.attn(q, k, v, need_weights=False)[0])
        x = x + block.ls_2(block.mlp(block.ln_2(x)))
        return x

    def forward(self, x, VL_tuning=False, B_tuning=False):
        x = self.conv1(x)                                    # (B, width, grid, grid)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + self.positional_embedding.to(x.dtype)
        if VL_tuning:
            pos_tkn = self.prompt + self.positional_embedding[0, :].to(x.dtype).expand(self.prompt_length, -1)
            x = torch.cat([x[:, :1], pos_tkn.expand(x.shape[0], -1, -1), x[:, 1:]], dim=1)   # [CLS, Vision, Language, patches]
        x = self.ln_pre(x)
        for i, block in enumerate(self.transformer.resblocks):
            if B_tuning and i in self.transformer.adapt_layer:
                x = self._prompted_block(block, x, self.expert_prompt[self.transformer.adapt_layer.index(i)])
            else:
                x = block(x)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x
