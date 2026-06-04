# ============================================================
# E2E Nonlinear-ECCT for BER Table Sweep
# - Nonlinear systematic code: c(m) = [p_theta(m), m]
# - No linear G/H/syndrome
# - 100k-step snapshot / plot / checkpoint
# - Colab/Drive-friendly resume
# ============================================================

import os, json, shutil, random, time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm

from IPython.display import display

try:
    from google.colab import files, drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False

torch.backends.cuda.matmul.allow_tf32 = True


# ============================================================
# Basic utils
# ============================================================
def format_seconds(seconds):
    if seconds is None or np.isinf(seconds) or np.isnan(seconds):
        return "unknown"

    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bits_to_bpsk(b):
    # bit 0 -> +1, bit 1 -> -1
    return 1.0 - 2.0 * b.float()


def ste_binary(logits, tau=1.0):
    """
    Straight-through binary estimator.
    Forward: hard {0,1}
    Backward: sigmoid gradient
    """
    soft = torch.sigmoid(logits / tau)
    hard = (soft >= 0.5).float()
    return hard.detach() - soft.detach() + soft


def ebn0_to_std(ebn0_db, rate):
    """
    Match existing ECCT-style convention:
    snr_db = Eb/N0 + 10log10(2R)
    """
    snr_db = ebn0_db + 10.0 * np.log10(2.0 * rate)
    return float(np.sqrt(1.0 / (10.0 ** (snr_db / 10.0))))


def sample_messages(bs, k, device):
    # Nonlinear code는 all-one message로만 학습하면 안 됨.
    # p_theta(m)가 전체 message space에 대해 parity를 배워야 하므로 random message 유지.
    return torch.randint(0, 2, (bs, k), device=device).float()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_name(s):
    return (
        s.replace("(", "_")
         .replace(")", "")
         .replace(",", "_")
         .replace("/", "_")
         .replace(" ", "")
    )


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def cfg_to_dict(cfg):
    return asdict(cfg)


# ============================================================
# Nonlinear Encoder
# ============================================================
class NonlinearEncoder(nn.Module):
    """
    Nonlinear systematic encoder.

    m: [B, k], binary
    p_theta(m): [B, n-k], nonlinear parity
    c(m): [p_theta(m), m]
    """
    def __init__(self, n, k, hidden=128, layers=2, tau=1.0):
        super().__init__()
        assert n > k

        self.n = n
        self.k = k
        self.r = n - k
        self.tau = tau

        net = []
        d = k

        for _ in range(layers):
            net += [
                nn.Linear(d, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            ]
            d = hidden

        net += [nn.Linear(d, self.r)]
        self.net = nn.Sequential(*net)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.7)
                nn.init.zeros_(m.bias)

    def forward(self, m, hard=True):
        # {0,1} -> {-1,+1}
        z = 2.0 * m.float() - 1.0

        parity_logits = self.net(z)

        if hard:
            parity = ste_binary(parity_logits, self.tau)
        else:
            parity = torch.sigmoid(parity_logits / self.tau)

        codeword = torch.cat([parity, m.float()], dim=1)
        return codeword


# ============================================================
# Syndrome-free Transformer Decoder
# ============================================================
class NonlinearECCDecoder(nn.Module):
    """
    No H.
    No G.
    No syndrome.

    Input:
        noisy BPSK y, shape [B, n]

    Output:
        codeword logits [B, n]
        message logits  [B, k]
    """
    def __init__(self, n, k, d_model=64, heads=8, layers=3, dropout=0.0):
        super().__init__()

        assert d_model % heads == 0

        self.n = n
        self.k = k
        self.r = n - k

        # per-symbol features:
        # y, |y|, y^2, hard channel decision
        self.input_proj = nn.Linear(4, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

        self.code_head = nn.Linear(d_model, 1)
        self.msg_head = nn.Linear(d_model, 1)

    def forward(self, y):
        hard = (y < 0).float() * 2.0 - 1.0

        feat = torch.stack(
            [
                y,
                y.abs(),
                y * y,
                hard,
            ],
            dim=-1,
        )

        h = self.input_proj(feat) + self.pos_embed
        h = self.transformer(h)
        h = self.norm(h)

        code_logits = self.code_head(h).squeeze(-1)

        # systematic message positions are the last k bits
        msg_logits = self.msg_head(h[:, self.r :, :]).squeeze(-1)

        return code_logits, msg_logits


# ============================================================
# Full Nonlinear E2E ECC Model
# ============================================================
class E2E_DC_ECC_Transformer(nn.Module):
    """
    Notebook 호환을 위해 class name은 유지.

    기존 E2E DC-ECCT:
        learn linear P -> construct G/H -> syndrome decoder

    현재 E2E Nonlinear-ECCT:
        learn nonlinear p_theta(m)
        c(m) = [p_theta(m), m]
        syndrome-free Transformer decoder
    """
    def __init__(
        self,
        n=31,
        k=16,
        d_model=64,
        num_heads=8,
        num_decoder_layers=3,
        encoder_hidden=128,
        encoder_layers=2,
        tau=1.0,
        code_w=0.25,
        dist_w=0.05,
        bal_w=0.01,
        target_rel_dist=0.25,
        dropout=0.0,
    ):
        super().__init__()

        self.n = n
        self.k = k
        self.r = n - k

        self.code_w = code_w
        self.dist_w = dist_w
        self.bal_w = bal_w
        self.target_rel_dist = target_rel_dist

        self.encoder = NonlinearEncoder(
            n=n,
            k=k,
            hidden=encoder_hidden,
            layers=encoder_layers,
            tau=tau,
        )

        self.decoder = NonlinearECCDecoder(
            n=n,
            k=k,
            d_model=d_model,
            heads=num_heads,
            layers=num_decoder_layers,
            dropout=dropout,
        )

    def encode(self, m, hard=True):
        return self.encoder(m, hard=hard)

    def forward(self, m, noise):
        c = self.encode(m, hard=True)
        x = bits_to_bpsk(c)

        y = x + noise

        code_logits, msg_logits = self.decoder(y)

        # main objective: recover original message
        msg_loss = F.binary_cross_entropy_with_logits(msg_logits, m.float())

        # auxiliary objective: recover full codeword
        # detach target so this term trains decoder, not encoder target itself
        code_loss = F.binary_cross_entropy_with_logits(code_logits, c.detach())

        # nonlinear codebook distance regularization
        m2 = torch.randint(0, 2, (m.size(0), self.k), device=m.device).float()
        c2 = self.encode(m2, hard=True)

        rel_dist = (c - c2).abs().mean(dim=1)
        dist_loss = F.relu(self.target_rel_dist - rel_dist).mean()

        # parity collapse prevention
        parity_mean = c[:, : self.r].mean(dim=0)
        bal_loss = ((parity_mean - 0.5) ** 2).mean()

        loss = (
            msg_loss
            + self.code_w * code_loss
            + self.dist_w * dist_loss
            + self.bal_w * bal_loss
        )

        code_pred = (torch.sigmoid(code_logits) >= 0.5).float()
        msg_pred = (torch.sigmoid(msg_logits) >= 0.5).float()

        metrics = {
            "loss": loss.detach(),
            "msg_loss": msg_loss.detach(),
            "code_loss": code_loss.detach(),
            "dist_loss": dist_loss.detach(),
            "bal_loss": bal_loss.detach(),
        }

        return loss, code_pred, c.detach(), msg_pred, m.detach(), metrics

    @torch.no_grad()
    def nonlinear_score(self, num_pairs=2048):
        """
        Check nonlinearity of parity map.

        For a linear parity map:
            p(a xor b) == p(a) xor p(b)

        Positive score means nonlinear behavior.
        """
        dev = next(self.parameters()).device

        a = sample_messages(num_pairs, self.k, dev)
        b = sample_messages(num_pairs, self.k, dev)
        axb = torch.remainder(a + b, 2.0)

        pa = self.encode(a)[:, : self.r]
        pb = self.encode(b)[:, : self.r]
        pab = self.encode(axb)[:, : self.r]

        expected = torch.remainder(pa + pb, 2.0)

        return float((pab != expected).float().mean().item())

    @torch.no_grad()
    def distance_estimate(self, pairs=4096):
        dev = next(self.parameters()).device

        a = sample_messages(pairs, self.k, dev)
        b = sample_messages(pairs, self.k, dev)

        same = (a == b).all(dim=1)
        if same.any():
            b[same, 0] = 1.0 - b[same, 0]

        ca = self.encode(a)
        cb = self.encode(b)

        d = (ca != cb).float().sum(dim=1)

        return {
            "sample_min_d": int(d.min().item()),
            "sample_avg_d": float(d.mean().item()),
        }

    def get_pc_matrix(self):
        raise RuntimeError("Nonlinear code: true parity-check matrix H does not exist.")

    def get_generator_matrix(self):
        raise RuntimeError("Nonlinear code: generator matrix G does not exist.")


# ============================================================
# Config
# ============================================================
@dataclass
class TrainConfig:
    n: int = 31
    k: int = 16

    d_model: int = 32
    num_heads: int = 8
    num_decoder_layers: int = 2

    encoder_hidden: int = 128
    encoder_layers: int = 2

    # 논문식 최종 세팅
    batch_size: int = 1024
    train_iters: int = 1_000_000

    lr: float = 1e-4
    lr_min: float = 1e-6
    grad_clip: float = 1.0

    # ECCT 논문에서 명시된 training Eb/N0 range
    train_ebn0_min: int = 1
    train_ebn0_max: int = 8

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # sweep metadata
    code_name: str = ""
    arch_name: str = ""


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate_ber(model, ebn0_grid, batch_size=1024, num_batches=5):
    """
    Quick evaluation. 중간 확인용.
    최종 표에는 evaluate_for_table() 사용.
    """
    dev = next(model.parameters()).device
    model.eval()

    out = []

    for ebn0 in ebn0_grid:
        std = ebn0_to_std(ebn0, model.k / model.n)

        msg_err = 0
        msg_tot = 0
        code_err = 0
        code_tot = 0
        fer = 0
        frames = 0

        for _ in range(num_batches):
            m = sample_messages(batch_size, model.k, dev)
            noise = torch.randn(batch_size, model.n, device=dev) * std

            _, code_pred, code_true, msg_pred, msg_true, _ = model(m, noise)

            me = msg_pred != msg_true
            ce = code_pred != code_true

            msg_err += int(me.sum().item())
            msg_tot += int(me.numel())

            code_err += int(ce.sum().item())
            code_tot += int(ce.numel())

            fer += int(me.any(dim=1).sum().item())
            frames += m.size(0)

        out.append(
            {
                "EbN0_dB": float(ebn0),
                "msg_BER": msg_err / max(msg_tot, 1),
                "code_BER": code_err / max(code_tot, 1),
                "FER": fer / max(frames, 1),
            }
        )

    return out


@torch.no_grad()
def evaluate_for_table(
    model,
    ebn0_grid=(4, 5, 6),
    batch_size=4096,
    min_frames=100_000,
    min_frame_errors=50,
    max_batches=5000,
):
    """
    논문 표 채우기용 평가.
    출력값 중 neg_ln_code_BER를 표에 넣으면 됨.
    """
    dev = next(model.parameters()).device
    model.eval()

    out = []

    for ebn0 in ebn0_grid:
        std = ebn0_to_std(ebn0, model.k / model.n)

        msg_err = 0
        msg_tot = 0
        code_err = 0
        code_tot = 0
        fer = 0
        frames = 0
        batches = 0

        while batches < max_batches:
            m = sample_messages(batch_size, model.k, dev)
            noise = torch.randn(batch_size, model.n, device=dev) * std

            _, code_pred, code_true, msg_pred, msg_true, _ = model(m, noise)

            me = msg_pred != msg_true
            ce = code_pred != code_true

            msg_err += int(me.sum().item())
            msg_tot += int(me.numel())

            code_err += int(ce.sum().item())
            code_tot += int(ce.numel())

            fer += int(me.any(dim=1).sum().item())
            frames += m.size(0)
            batches += 1

            if frames >= min_frames and fer >= min_frame_errors:
                break

        msg_ber = msg_err / max(msg_tot, 1)
        code_ber = code_err / max(code_tot, 1)
        frame_er = fer / max(frames, 1)

        # zero BER 방지
        code_ber_safe = max(code_ber, 1.0 / max(code_tot, 1))

        out.append({
            "EbN0_dB": float(ebn0),
            "msg_BER": msg_ber,
            "code_BER": code_ber,
            "FER": frame_er,
            "neg_ln_code_BER": -np.log(code_ber_safe),
            "frames": frames,
            "frame_errors": fer,
        })

    return pd.DataFrame(out)


# ============================================================
# Plot / save / download helpers
# ============================================================
def make_progress_plots(hist_df, quick_eval_df, out_dir, exp_name, step):
    ensure_dir(out_dir)

    # 1. Training loss
    plt.figure(figsize=(8, 4))
    plt.plot(hist_df["iter"], hist_df["loss"], marker="o", label="total")
    plt.plot(hist_df["iter"], hist_df["msg_loss"], marker="o", label="message")
    plt.plot(hist_df["iter"], hist_df["code_loss"], marker="o", label="code auxiliary")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(f"{exp_name} | loss up to {step}")
    plt.grid(True)
    plt.legend()
    loss_png = os.path.join(out_dir, f"{exp_name}_step_{step}_loss.png")
    plt.savefig(loss_png, dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Online BER
    plt.figure(figsize=(8, 4))
    plt.semilogy(hist_df["iter"], hist_df["msg_BER"], marker="o", label="online message BER")
    plt.semilogy(hist_df["iter"], hist_df["code_BER"], marker="s", label="online code BER")
    plt.xlabel("iteration")
    plt.ylabel("BER")
    plt.title(f"{exp_name} | online BER up to {step}")
    plt.grid(True, which="both")
    plt.legend()
    ber_png = os.path.join(out_dir, f"{exp_name}_step_{step}_online_ber.png")
    plt.savefig(ber_png, dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Nonlinear score
    plt.figure(figsize=(8, 4))
    plt.plot(hist_df["iter"], hist_df["nl_score"], marker="o", label="nonlinear score")
    plt.xlabel("iteration")
    plt.ylabel("nonlinear score")
    plt.title(f"{exp_name} | nonlinear score up to {step}")
    plt.grid(True)
    plt.legend()
    nl_png = os.path.join(out_dir, f"{exp_name}_step_{step}_nl_score.png")
    plt.savefig(nl_png, dpi=150, bbox_inches="tight")
    plt.close()

    # 4. Quick BER evaluation
    if quick_eval_df is not None and len(quick_eval_df) > 0:
        plt.figure(figsize=(8, 4))
        plt.semilogy(quick_eval_df["EbN0_dB"], quick_eval_df["msg_BER"], marker="o", label="message BER")
        plt.semilogy(quick_eval_df["EbN0_dB"], quick_eval_df["code_BER"], marker="s", label="code BER")
        plt.semilogy(quick_eval_df["EbN0_dB"], quick_eval_df["FER"], marker="^", label="FER")
        plt.xlabel("Eb/N0 (dB)")
        plt.ylabel("error rate")
        plt.title(f"{exp_name} | quick eval at step {step}")
        plt.grid(True, which="both")
        plt.legend()
        eval_png = os.path.join(out_dir, f"{exp_name}_step_{step}_quick_eval.png")
        plt.savefig(eval_png, dpi=150, bbox_inches="tight")
        plt.close()


def create_lightweight_snapshot_zip(exp_dir, exp_name, step, include_checkpoint=False):
    """
    로컬 다운로드용 zip 생성.
    기본값 include_checkpoint=False:
        .pt checkpoint는 제외해서 다운로드 용량 줄임.
    checkpoint까지 로컬로 받고 싶으면 include_checkpoint=True.
    """
    tmp_dir = f"/content/{exp_name}_step_{step}_download"
    zip_base = f"/content/{exp_name}_step_{step}"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    ensure_dir(tmp_dir)

    for root, _, files_in_dir in os.walk(exp_dir):
        rel_root = os.path.relpath(root, exp_dir)
        dst_root = os.path.join(tmp_dir, rel_root) if rel_root != "." else tmp_dir
        ensure_dir(dst_root)

        for fn in files_in_dir:
            src = os.path.join(root, fn)

            if (not include_checkpoint) and fn.endswith(".pt"):
                continue

            dst = os.path.join(dst_root, fn)
            shutil.copy2(src, dst)

    if os.path.exists(zip_base + ".zip"):
        os.remove(zip_base + ".zip")

    shutil.make_archive(zip_base, "zip", tmp_dir)
    shutil.rmtree(tmp_dir)

    return zip_base + ".zip"


def maybe_download_snapshot(exp_dir, exp_name, step, auto_download=True, include_checkpoint=False):
    if not auto_download:
        return

    zip_path = create_lightweight_snapshot_zip(
        exp_dir=exp_dir,
        exp_name=exp_name,
        step=step,
        include_checkpoint=include_checkpoint,
    )

    print("Prepared download:", zip_path)

    if IN_COLAB:
        files.download(zip_path)
    else:
        print("Not in Colab. File saved:", zip_path)


# ============================================================
# Training with snapshots / resume
# ============================================================
def train_model_with_snapshots(
    cfg,
    exp_name,
    exp_dir,
    resume=True,
    snapshot_every=100_000,
    quick_eval_every=100_000,
    quick_eval_batches=10,
    quick_eval_ebn0_grid=None,
    auto_download=True,
    download_checkpoint=False,

    # 추가: 진행률 출력 관련
    progress_every=10_000,
    pbar_update_every=100,
):
    seed_everything(cfg.seed)
    ensure_dir(exp_dir)

    dev = torch.device(cfg.device)

    ckpt_latest = os.path.join(exp_dir, "latest.pt")
    hist_csv = os.path.join(exp_dir, "history.csv")
    quick_eval_csv = os.path.join(exp_dir, "quick_eval_history.csv")

    model = E2E_DC_ECC_Transformer(
        n=cfg.n,
        k=cfg.k,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_decoder_layers=cfg.num_decoder_layers,
        encoder_hidden=cfg.encoder_hidden,
        encoder_layers=cfg.encoder_layers,
    ).to(dev)

    # 논문식: Adam + cosine decay
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=cfg.train_iters,
        eta_min=cfg.lr_min,
    )

    history = {
        "iter": [],
        "loss": [],
        "msg_loss": [],
        "code_loss": [],
        "dist_loss": [],
        "bal_loss": [],
        "msg_BER": [],
        "code_BER": [],
        "nl_score": [],
        "lr": [],
    }

    quick_eval_rows = []
    start_step = 0

    if resume and os.path.exists(ckpt_latest):
        print(f"Resuming from {ckpt_latest}")
        ckpt = torch.load(ckpt_latest, map_location=dev)

        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        history = ckpt.get("history", history)
        quick_eval_rows = ckpt.get("quick_eval_rows", quick_eval_rows)
        start_step = ckpt.get("step", 0)

        print(f"Resume step: {start_step}")

    stds = torch.tensor(
        [
            ebn0_to_std(e, cfg.k / cfg.n)
            for e in range(cfg.train_ebn0_min, cfg.train_ebn0_max + 1)
        ],
        device=dev,
    )

    model.train()
    t0 = time.time()

    if quick_eval_ebn0_grid is None:
        quick_eval_ebn0_grid = list(range(1, 9))

    print("-" * 100)
    print(f"[RUN START] {exp_name}")
    print(f"start_step       : {start_step}")
    print(f"target_train_iter: {cfg.train_iters}")
    print(f"remaining_steps  : {cfg.train_iters - start_step}")
    print(f"snapshot_every   : {snapshot_every}")
    print(f"progress_every   : {progress_every}")
    print("-" * 100)

    pbar = tqdm(
        total=cfg.train_iters,
        initial=start_step,
        desc=f"{exp_name}",
        unit="step",
        dynamic_ncols=True,
    )

    last_pbar_step = start_step

    try:
        for step in range(start_step + 1, cfg.train_iters + 1):
            m = sample_messages(cfg.batch_size, cfg.k, dev)

            s = stds[torch.randint(0, len(stds), (cfg.batch_size,), device=dev)]
            noise = torch.randn(cfg.batch_size, cfg.n, device=dev) * s[:, None]

            opt.zero_grad(set_to_none=True)

            loss, code_pred, code_true, msg_pred, msg_true, met = model(m, noise)

            loss.backward()

            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()
            scheduler.step()

            # tqdm progress bar update
            if step % pbar_update_every == 0 or step == cfg.train_iters:
                pbar.update(step - last_pbar_step)
                last_pbar_step = step

            # lightweight progress log: 매 1만 step마다 ETA 출력
            if step % progress_every == 0 or step == cfg.train_iters:
                elapsed_s = time.time() - t0
                done_this_session = step - start_step
                speed = done_this_session / max(elapsed_s, 1e-9)
                remain_steps = cfg.train_iters - step
                eta_s = remain_steps / max(speed, 1e-9)
                percent = 100.0 * step / cfg.train_iters

                cur_lr = scheduler.get_last_lr()[0]

                print(
                    f"[PROGRESS] {exp_name} | "
                    f"step {step:,}/{cfg.train_iters:,} "
                    f"({percent:.2f}%) | "
                    f"speed={speed:.2f} step/s | "
                    f"elapsed={format_seconds(elapsed_s)} | "
                    f"ETA={format_seconds(eta_s)} | "
                    f"lr={cur_lr:.2e}"
                )

            need_snapshot = (
                step % snapshot_every == 0
                or step == cfg.train_iters
            )

            if need_snapshot:
                with torch.no_grad():
                    msg_ber = (msg_pred != msg_true).float().mean().item()
                    code_ber = (code_pred != code_true).float().mean().item()
                    nl_score = model.nonlinear_score(2048)
                    cur_lr = scheduler.get_last_lr()[0]

                history["iter"].append(step)
                history["loss"].append(float(met["loss"]))
                history["msg_loss"].append(float(met["msg_loss"]))
                history["code_loss"].append(float(met["code_loss"]))
                history["dist_loss"].append(float(met["dist_loss"]))
                history["bal_loss"].append(float(met["bal_loss"]))
                history["msg_BER"].append(msg_ber)
                history["code_BER"].append(code_ber)
                history["nl_score"].append(nl_score)
                history["lr"].append(cur_lr)

                elapsed_min = (time.time() - t0) / 60
                percent = 100.0 * step / cfg.train_iters

                print(
                    f"[SNAPSHOT] {exp_name} | "
                    f"step {step:7d}/{cfg.train_iters} ({percent:.2f}%) | "
                    f"loss={float(met['loss']):.5f} | "
                    f"msg_BER={msg_ber:.6f} | "
                    f"code_BER={code_ber:.6f} | "
                    f"nl={nl_score:.3f} | "
                    f"lr={cur_lr:.2e} | "
                    f"elapsed={elapsed_min:.1f} min"
                )

                hist_df = pd.DataFrame(history)
                hist_df.to_csv(hist_csv, index=False)

                quick_eval_df = None

                if step % quick_eval_every == 0 or step == cfg.train_iters:
                    quick_eval = evaluate_ber(
                        model,
                        ebn0_grid=quick_eval_ebn0_grid,
                        batch_size=1024,
                        num_batches=quick_eval_batches,
                    )

                    for row in quick_eval:
                        row["step"] = step
                        row["exp_name"] = exp_name
                        row["code"] = cfg.code_name
                        row["arch"] = cfg.arch_name
                        row["n"] = cfg.n
                        row["k"] = cfg.k
                        row["d_model"] = cfg.d_model
                        row["num_layers"] = cfg.num_decoder_layers
                        row["neg_ln_code_BER"] = -np.log(max(row["code_BER"], 1e-12))

                    quick_eval_rows.extend(quick_eval)
                    quick_eval_df = pd.DataFrame(quick_eval)

                    # ----------------------------------------------------
                    # 1) Long format 저장
                    #    기존 quick_eval_history.csv
                    #    step, EbN0_dB, msg_BER, code_BER, FER 형태로 길게 저장됨
                    # ----------------------------------------------------
                    quick_eval_long_df = pd.DataFrame(quick_eval_rows)
                    quick_eval_long_df.to_csv(quick_eval_csv, index=False)

                    # ----------------------------------------------------
                    # 2) Wide format 저장
                    #    엑셀에서 보기 좋은 형태
                    #    한 row = 특정 step
                    #    columns = msg_BER_EbN0_1, ..., code_BER_EbN0_8, ...
                    # ----------------------------------------------------
                    quick_eval_wide_csv = os.path.join(exp_dir, "quick_eval_history_wide.csv")
                    quick_eval_wide_xlsx = os.path.join(exp_dir, "quick_eval_history_wide.xlsx")

                    quick_eval_long_df["EbN0_int"] = quick_eval_long_df["EbN0_dB"].astype(int)

                    index_cols = [
                        "step",
                        "exp_name",
                        "code",
                        "arch",
                        "n",
                        "k",
                        "d_model",
                        "num_layers",
                    ]

                    wide_parts = []

                    for metric in ["msg_BER", "code_BER", "FER", "neg_ln_code_BER"]:
                        wide_metric = quick_eval_long_df.pivot_table(
                            index=index_cols,
                            columns="EbN0_int",
                            values=metric,
                            aggfunc="last",
                        )

                        wide_metric.columns = [
                            f"{metric}_EbN0_{int(c)}"
                            for c in wide_metric.columns
                        ]

                        wide_parts.append(wide_metric)

                    quick_eval_wide_df = pd.concat(wide_parts, axis=1).reset_index()

                    quick_eval_wide_df.to_csv(quick_eval_wide_csv, index=False)

                    try:
                        quick_eval_wide_df.to_excel(quick_eval_wide_xlsx, index=False)
                    except Exception as e:
                        print("Could not save xlsx. CSV was saved instead.")
                        print("xlsx error:", e)

                    print("Quick eval long format:")
                    # display(quick_eval_df)

                    print("Quick eval wide format for Excel:")
                    # display(quick_eval_wide_df.tail())

                    print("Saved quick eval files:")
                    print(" -", quick_eval_csv)
                    print(" -", quick_eval_wide_csv)
                    print(" -", quick_eval_wide_xlsx)

                    model.train()

                make_progress_plots(
                    hist_df=hist_df,
                    quick_eval_df=quick_eval_df,
                    out_dir=exp_dir,
                    exp_name=exp_name,
                    step=step,
                )

                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": cfg_to_dict(cfg),
                        "history": history,
                        "quick_eval_rows": quick_eval_rows,
                    },
                    ckpt_latest,
                )

                save_json(
                    {
                        "exp_name": exp_name,
                        "step": step,
                        "train_iters": cfg.train_iters,
                        "updated_at": datetime.now().isoformat(),
                        "config": cfg_to_dict(cfg),
                    },
                    os.path.join(exp_dir, "status.json"),
                )

                maybe_download_snapshot(
                    exp_dir=exp_dir,
                    exp_name=exp_name,
                    step=step,
                    auto_download=auto_download,
                    include_checkpoint=download_checkpoint,
                )

                model.train()

    finally:
        pbar.close()

    elapsed_s = time.time() - t0
    print("-" * 100)
    print(f"[RUN END] {exp_name}")
    print(f"session_elapsed: {format_seconds(elapsed_s)}")
    print("-" * 100)

    return model, history


print("Cell 1 loaded: model, training, evaluation, snapshot utilities are ready.")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))



# ============================================================
# Full sweep for Table 1: E2E Nonlinear-ECCT
# ============================================================

# ------------------------------------------------------------
# Storage option: local PC
# ------------------------------------------------------------
# 로컬 PC에서 실행할 때 결과 저장 폴더
# 현재 notebook이 있는 폴더 아래에 e2e_nonlinear_ecct_sweep 폴더가 생성됨.

USE_GOOGLE_DRIVE = False

ROOT_DIR = str(Path.cwd() / "e2e_nonlinear_ecct_sweep")

ensure_dir(ROOT_DIR)

print("Local result directory:", ROOT_DIR)

# ------------------------------------------------------------
# Snapshot / download options
# ------------------------------------------------------------
AUTO_DOWNLOAD = False

# 기본값 False 추천:
# - False: CSV/JSON/PNG만 zip 다운로드. 가벼움.
# - True: latest.pt checkpoint까지 다운로드. 매우 큼.
DOWNLOAD_CHECKPOINT = False

SNAPSHOT_EVERY = 100_000
QUICK_EVAL_EVERY = 100_000
QUICK_EVAL_BATCHES = 10

# 10만 step마다 SNR 1~8 전체 quick evaluation
QUICK_EVAL_EBN0_GRID = list(range(1, 9))

FINAL_RESULTS_CSV = os.path.join(ROOT_DIR, "final_table_results_long.csv")
FINAL_WIDE_CSV = os.path.join(ROOT_DIR, "final_table_results_wide.csv")


# ------------------------------------------------------------
# Debug / final mode
# ------------------------------------------------------------
# 처음엔 반드시 DEBUG_MODE=True로 shape/check만 확인 추천.
# 확인 끝나면 False로 바꾸면 최종 1M step sweep.
DEBUG_MODE = False

if DEBUG_MODE:
    TRAIN_ITERS = 10_000
    BATCH_SIZE = 512
    AUTO_DOWNLOAD = False
    QUICK_EVAL_BATCHES = 2
else:
    TRAIN_ITERS = 1_000_000
    BATCH_SIZE = 1024


# ------------------------------------------------------------
# Table rows
# ------------------------------------------------------------
code_specs = [
    ("POLAR(32,11)", 32, 11),
    ("POLAR(64,32)", 64, 32),
    ("BCH(31,16)", 31, 16),
    ("BCH(63,45)", 63, 45),
    ("LDPC(49,24)", 49, 24),
    ("RS(60,52)", 60, 52),
]

arch_specs = [
    # 표의 첫 번째 줄: N=2, d=32
    ("small_N2_d32", 2, 32),

    # 표의 두 번째 줄: N=6, d=128
    ("large_N6_d128", 6, 128),
]


# ------------------------------------------------------------
# Load previous final results if any
# ------------------------------------------------------------
if os.path.exists(FINAL_RESULTS_CSV):
    final_long_df = pd.read_csv(FINAL_RESULTS_CSV)
    all_final_rows = final_long_df.to_dict("records")
    print("Loaded existing final results:", FINAL_RESULTS_CSV)
else:
    final_long_df = pd.DataFrame()
    all_final_rows = []

# ------------------------------------------------------------
# Build experiment list and sweep progress helper
# ------------------------------------------------------------
experiment_specs = []

for code_name, n, k in code_specs:
    for arch_name, num_layers, d_model in arch_specs:
        exp_name = f"{safe_name(code_name)}_{arch_name}"
        exp_dir = os.path.join(ROOT_DIR, exp_name)
        final_marker = os.path.join(exp_dir, "DONE.json")

        experiment_specs.append({
            "code_name": code_name,
            "n": n,
            "k": k,
            "arch_name": arch_name,
            "num_layers": num_layers,
            "d_model": d_model,
            "exp_name": exp_name,
            "exp_dir": exp_dir,
            "done": os.path.exists(final_marker),
        })

TOTAL_EXPERIMENTS = len(experiment_specs)
ALREADY_DONE = sum(1 for x in experiment_specs if x["done"])
REMAINING_EXPERIMENTS = TOTAL_EXPERIMENTS - ALREADY_DONE

print("=" * 100)
print("[SWEEP PLAN]")
print(f"total experiments     : {TOTAL_EXPERIMENTS}")
print(f"already done          : {ALREADY_DONE}")
print(f"remaining experiments : {REMAINING_EXPERIMENTS}")
print(f"train iters per run   : {TRAIN_ITERS:,}")
print(f"batch size            : {BATCH_SIZE}")
print(f"root dir              : {ROOT_DIR}")
print("=" * 100)

sweep_start_time = time.time()
finished_this_session = 0

# ------------------------------------------------------------
# Run sweep
# ------------------------------------------------------------
for exp_idx, spec in enumerate(experiment_specs, start=1):
    code_name = spec["code_name"]
    n = spec["n"]
    k = spec["k"]
    arch_name = spec["arch_name"]
    num_layers = spec["num_layers"]
    d_model = spec["d_model"]
    exp_name = spec["exp_name"]
    exp_dir = spec["exp_dir"]

    ensure_dir(exp_dir)

    final_marker = os.path.join(exp_dir, "DONE.json")

    if os.path.exists(final_marker):
        print(f"[SWEEP] {exp_idx}/{TOTAL_EXPERIMENTS} | already done, skipping: {exp_name}")
        continue

    print("=" * 100)
    print(f"[SWEEP] Experiment {exp_idx}/{TOTAL_EXPERIMENTS}")
    print(f"Starting experiment: {exp_name}")
    print(f"Remaining including this: {TOTAL_EXPERIMENTS - exp_idx + 1}")
    print("=" * 100)

    cfg = TrainConfig(
        n=n,
        k=k,

        d_model=d_model,
        num_heads=8,
        num_decoder_layers=num_layers,

        encoder_hidden=128,
        encoder_layers=2,

        batch_size=BATCH_SIZE,
        train_iters=TRAIN_ITERS,

        # 논문식 optimizer setting
        lr=1e-4,
        lr_min=1e-6,

        # ECCT 논문에서 명시된 training Eb/N0 range
        train_ebn0_min=1,
        train_ebn0_max=8,

        seed=42,
        device="cuda" if torch.cuda.is_available() else "cpu",

        code_name=code_name,
        arch_name=arch_name,
    )

    print("device:", cfg.device)
    print("config:", cfg_to_dict(cfg))

    t_exp = time.time()

    model, history = train_model_with_snapshots(
        cfg=cfg,
        exp_name=exp_name,
        exp_dir=exp_dir,
        resume=True,
        snapshot_every=SNAPSHOT_EVERY,
        quick_eval_every=QUICK_EVAL_EVERY,
        quick_eval_batches=QUICK_EVAL_BATCHES,
        quick_eval_ebn0_grid=QUICK_EVAL_EBN0_GRID,
        auto_download=AUTO_DOWNLOAD,
        download_checkpoint=DOWNLOAD_CHECKPOINT,

        # 진행률 출력 주기
        progress_every=10_000,
        pbar_update_every=100,
    )

    train_minutes = (time.time() - t_exp) / 60
    print(f"Finished training {exp_name} in {train_minutes:.2f} min")

    # ----------------------------------------------------
    # Final table evaluation: Eb/N0 = 4,5,6
    # ----------------------------------------------------
    final_eval_df = evaluate_for_table(
        model,
        ebn0_grid=(4, 5, 6),
        batch_size=4096,
        min_frames=100_000,
        min_frame_errors=50,
        max_batches=5000,
    )

    final_eval_df.insert(0, "code", code_name)
    final_eval_df.insert(1, "arch", arch_name)
    final_eval_df.insert(2, "n", n)
    final_eval_df.insert(3, "k", k)
    final_eval_df.insert(4, "d_model", d_model)
    final_eval_df.insert(5, "num_layers", num_layers)
    final_eval_df.insert(6, "train_iters", cfg.train_iters)
    final_eval_df.insert(7, "train_minutes", train_minutes)

    print("Final eval:")
    display(final_eval_df)

    final_eval_path = os.path.join(exp_dir, "final_eval.csv")
    final_eval_df.to_csv(final_eval_path, index=False)

    # ----------------------------------------------------
    # Save cumulative long/wide results
    # ----------------------------------------------------
    all_final_rows.extend(final_eval_df.to_dict("records"))
    final_long_df = pd.DataFrame(all_final_rows)

    # 중복 방지: 같은 code/arch/EbN0가 여러 번 있으면 마지막 값 사용
    final_long_df = final_long_df.drop_duplicates(
        subset=["code", "arch", "EbN0_dB"],
        keep="last",
    )

    final_long_df.to_csv(FINAL_RESULTS_CSV, index=False)

    final_wide_df = final_long_df.pivot_table(
        index=["code", "arch", "n", "k", "num_layers", "d_model"],
        columns="EbN0_dB",
        values="neg_ln_code_BER",
        aggfunc="last",
    ).reset_index()

    final_wide_df.to_csv(FINAL_WIDE_CSV, index=False)

    print("Current wide table:")
    display(final_wide_df)

    # ----------------------------------------------------
    # Mark done
    # ----------------------------------------------------
    save_json(
        {
            "exp_name": exp_name,
            "code": code_name,
            "arch": arch_name,
            "n": n,
            "k": k,
            "num_layers": num_layers,
            "d_model": d_model,
            "train_minutes": train_minutes,
            "done_at": datetime.now().isoformat(),
            "config": cfg_to_dict(cfg),
        },
        final_marker,
    )

    # 실험 하나 완료될 때마다 전체 결과 lightweight zip 다운로드
    if AUTO_DOWNLOAD:
        zip_path = create_lightweight_snapshot_zip(
            exp_dir=ROOT_DIR,
            exp_name=f"ALL_RESULTS_AFTER_{exp_name}",
            step=cfg.train_iters,
            include_checkpoint=DOWNLOAD_CHECKPOINT,
        )
        print("Prepared full results download:", zip_path)

        if IN_COLAB:
            files.download(zip_path)

    # GPU memory cleanup
    del model
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Sweep-level ETA
    # ----------------------------------------------------
    finished_this_session += 1

    sweep_elapsed_s = time.time() - sweep_start_time
    avg_exp_s = sweep_elapsed_s / max(finished_this_session, 1)

    remaining_not_done = 0
    for s in experiment_specs:
        marker = os.path.join(s["exp_dir"], "DONE.json")
        if not os.path.exists(marker):
            remaining_not_done += 1

    sweep_eta_s = remaining_not_done * avg_exp_s

    print("=" * 100)
    print("[SWEEP PROGRESS]")
    print(f"finished this session : {finished_this_session}")
    print(f"remaining experiments : {remaining_not_done}")
    print(f"avg time / experiment : {format_seconds(avg_exp_s)}")
    print(f"sweep elapsed         : {format_seconds(sweep_elapsed_s)}")
    print(f"estimated sweep ETA   : {format_seconds(sweep_eta_s)}")
    print("=" * 100)

print("All sweep experiments finished or skipped.")

if os.path.exists(FINAL_RESULTS_CSV):
    final_long_df = pd.read_csv(FINAL_RESULTS_CSV)
    print("Final long results:")
    display(final_long_df)

if os.path.exists(FINAL_WIDE_CSV):
    final_wide_df = pd.read_csv(FINAL_WIDE_CSV)
    print("Final wide results for table:")
    display(final_wide_df)