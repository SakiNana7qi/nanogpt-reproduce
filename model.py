import torch
import torch.nn as nn
import math
import inspect
import os
import time


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 768 / 12 = 64

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输入经过QKV

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 输出层

        self.c_proj.WEIGHT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # self.register_buffer(
        #     "bias",
        #     torch.tril(torch.ones(config.block_size, config.block_size)).view(
        #         1, 1, config.block_size, config.block_size
        #     ),
        # )
        # 用 register_buffer 方法创建下三角矩阵，避免进入梯度计算
        # 但是 scaled_dot_product_attention is_causal=True 这样就不用了 causal 因果注意力

    def forward(self, x):
        # Attention(Q,K,V) = softmax( (Q@K^T)/sqrt(d_k) )@V

        B, T, C = x.size()
        # batch_size, sequence length, embedding dimensionality (n_embd)

        qkv = self.c_attn(x)
        # [B,T,C] -> [B,T,3*C]
        """
        Linear 特性：
        设计输入形状：[*, in_features]
        设计输出形状：[*, out_features]
        其中 * 表示任意数量的额外维度
        """

        q, k, v = qkv.split(self.n_embd, dim=-1)
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        # [B,T,C] -> [B,T,n_head,head_size] -> [B,n_head,T,head_size]

        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = torch.nn.functional.softmax(att, dim=-1)
        # y = att @ v

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )  # Flash Attention

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # transpose 后用 view 前要先 contiguous

        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        self.c_proj.WEIGHT_SCALE_INIT = 1

    def forward(self, x):
        """
        FFN(x) = max(0,xW1+b1)@W2+b2
        max(0, ) = GeLU
        """

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # [gpt2] Layer normalization was moved to the input of each sub-block,

        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                # word token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),
                # position embeddings
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                # gpt blocks
                ln_f=nn.LayerNorm(config.n_embd),
                # [gpt2] and an additional layer normalization was added after the final selfattention block.
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # embeddings to final output
        # Embedding(X,Y) => [X,Y]
        # Linear(X,Y) => [Y,X]

        self.transformer.wte.weight = self.lm_head.weight
        # weight sharing [feature]

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "WEIGHT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # 初始化原地操作即可

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Input_Embded + Position_Embd -> x
        x -- block() * N_layer --> x

        """

        B, T = idx.size()
        # batch_size, length

        assert T <= self.config.block_size, f"length {T} > blocksize {T}"
        # max_length <= max_sequence_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        # [T,n_embd] 每一个 batch 的 pos_emb 都是一样的

        tok_emb = self.transformer.wte(idx)
        # [B,T,n_embd]

        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def configure_optimizers(
        self, weight_decay, learning_rate, master_process, device_type
    ):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 获取所有参数

        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        # Linear.weight 要权重衰减, Linear.bias 和 LayerNorm 不用

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
            )
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        # AdamW 是否支持融合模式(fused)
        used_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {used_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=used_fused,
        )
        return optimizer


def get_human_time():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def save_checkpoint(model, step, val_loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "config": model.config,
        "step": step,
        "val_loss": val_loss,
    }

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"checkpoint{get_human_time()}_step_{step:07d}.pth",
    )

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint at step {step} to {checkpoint_path}")
    return checkpoint_path
