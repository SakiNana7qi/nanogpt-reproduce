from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os
from model import *
from model_config import GPTconfig
from dataloaderlite import *
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import tiktoken

ddp = int(os.environ.get("RANK", -1) != -1)
if ddp:
    assert torch.cuda.is_available(), "CUDA must be available"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"

seed = 114514
torch.manual_seed(seed)
if device_type == "cuda":
    torch.cuda.manual_seed(seed)
enc = tiktoken.get_encoding("gpt2")

total_batch_size = 2**19
B = 16
T = 1024
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "total_batch_size % B * T * ddp_world_size != 0"

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total batch size: {total_batch_size}")
    print(f"=> calcucated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    master_process=master_process,
    split="train",
)

val_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    master_process=master_process,
    split="val",
)

torch.set_float32_matmul_precision("high")  # TF32

model = GPT(GPTconfig(vocab_size=50304))  # 50257 -> 50304
model.to(device)
cpt0 = time.time()
model = torch.compile(model, dynamic=True)
print(f"compile time: {(time.time()-cpt0)*1000:.2f}ms")

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

max_steps = 10**10 // total_batch_size
warmup_steps = int(max_steps * 0.0375)
max_lr = 6e-4
min_lr = max_lr * 0.1


def get_lr(it):
    # 1) warm_up (linear)
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) > lr_decay_iters, return min_lr
    if it > max_steps:
        return min_lr
    # 3) cos ratio
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


from hellaswag import render_example, iterate_examples


def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = torch.nn.functional.cross_entropy(
        flat_shift_logits, flat_shift_tokens, reduction="none"
    )
    shift_losses = shift_losses.view(tokens.size(0), -1)

    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask

    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    pred_norm = avg_loss.argmin().item()
    return pred_norm


sumdt, sumtps = 0.0, 0.0
raw_model = model.module if ddp else model
# DDP 导致的

optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=6e-4,
    master_process=master_process,
    device_type=device_type,
)

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log{get_human_time()}.txt")
with open(log_file, "w") as f:
    pass

print(f"max_steps {max_steps}")

evaluation_steps = 250

# 开训
for step in range(max_steps):
    t0 = time.time()
    is_laststep = step == max_steps - 1

    # Val
    if step % evaluation_steps == 0 or is_laststep:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    # 混合精度
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach()
            val_loss_accum /= val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            # all_reduce 将所有卡的结果归约, AVG 取平均

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or is_laststep):
                save_checkpoint(raw_model, step, val_loss_accum.item())

    # Test
    if (step > 0 and step % evaluation_steps == 0) or is_laststep:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        # [N,] -> [1,N,] -> [1*num_return_sequences,N*1,] -> [n_r_s,N]

        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(seed + ddp_rank)

        # generate
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # 概率值和在 vocal 中的索引, 都是 [B,50]

                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                # 每个 batch 采样 1 个, 返回在 topk 中的位置

                xcol = torch.gather(topk_indices, -1, ix)
                # 由 ix 转化为 vocal 索引

                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # hellaswag
    if step % evaluation_steps == 0 or is_laststep:
        model.eval()
        num_correct = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = render_example(example)
            tokens, mask = tokens.to(device), mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                predict = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            if predict == label:
                num_correct += 1
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct = torch.tensor(num_correct, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct = num_correct.item()
        accuracy = num_correct / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct}/{num_total}={accuracy:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {accuracy:.4f}\n")

    # train
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for ministep in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = ministep == grad_accum_steps - 1
            """
            避免每个 ministep 都同步梯度
            只在最后一次微步时同步
            显著减少通信开销
            """
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss /= grad_accum_steps
        """
        梯度累积（分4个微步，每步32样本）
        微步1: [样本1-32] → loss1/4 → backward → 梯度累积
        微步2: [样本33-64] → loss2/4 → backward → 梯度累积
        微步3: [样本65-96] → loss3/4 → backward → 梯度累积
        微步4: [样本97-128] → loss4/4 → backward → 梯度累积
        总梯度 = (梯度1 + 梯度2 + 梯度3 + 梯度4) 
        = 1/4(原始梯度1) + 1/4(原始梯度2) + ... 
        = 1/4 * (原始梯度总和)
        = 原始大批次的平均梯度
        """

        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 梯度裁剪 防止梯度爆炸

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_processed = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / (t1 - t0)
    if step > 0:
        sumdt += dt
        sumtps += tokens_per_sec
    if master_process:
        print(
            f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e}|norm: {norm:.4f}| dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

        if step == 5:
            print(f"Estimate hours: {max_steps*dt/1000/60/60:.1f}")

print(f"Average | dt {sumdt/(max_steps-1):.2f} | tps: {sumtps/(max_steps-1):.2f}")
print("train finished")

if ddp:
    destroy_process_group()
