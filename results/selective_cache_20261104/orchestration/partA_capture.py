"""Part A capture: dense decode, per-call/per-layer drift of every row's K and V against snapshots s calls earlier, plus the candidate staleness signals.

Everything is computed on the fly inside the attention call of each layer (the model is NOT changed: the block's _scaled_dot_product_attention is wrapped, the
original is still called). Nothing is cached or skipped; this is the dense Conf-0.9 decode.

Definitions (all at call t, layer l, snapshot age s in AGES, row r):
  drift_kv(r)   = || [K_t[r], V_t[r]] - [K_{t-s}[r], V_{t-s}[r]] || / || [K_{t-s}[r], V_{t-s}[r]] ||   (row = all heads concatenated, post-RoPE K)
  valid pair    = the row holds the same input token at calls t-s and t (a row whose mask was just replaced by a word has no comparable snapshot; Elastic always
                  recomputes rows written on the previous call, so those pairs are not a caching question).
  N_{t,s}       = positions that were blank at call t-s and are written at call t ("written since the snapshot").
  exposure E_fresh(r,s)  = sum over calls u in (t-s, t] of  mass_u(r -> N_{u,u-(t-s)}),  mass from the head-mean attention row of r at call u (fresh query).
  exposure E_cq(r,s)     = the same sum, but the attention row uses r's CACHED query (the query of the snapshot call) against the current keys: what a cache that
                           stores q_cache (as Elastic does) can compute without recomputing r.
  window mass WM_inst(r) = sum over the 16 window rows q of head-mean attention[q -> r] at call t; WM_acc(r,s) its sum over calls in (t-s, t].
  dist(r,s)     = |r - nearest position in N_{t,s}| (T if N is empty).
  Elastic trigger cosines: tracked rows = union over layers of the row each layer's window attention mass is largest on (non-blank rows only, as coded);
                  cos_heads = cosine across the HEAD axis of attention[tracked rows -> keys] at call t-s vs now, averaged over (rows, keys) (what the code computes:
                  F.cosine_similarity(a, b, dim=1) on a (batch, heads, tracked, keys) tensor);  cos_keys = cosine across the KEY axis, averaged over (heads, rows).
Classes at call t: 0 prompt, 1 written answer row, 2 blank in the window (first 16 blanks), 3 blank beyond the window inside the current 32-block, 4 blank in a later block.
"""
from __future__ import annotations

import math

import numpy as np
import torch

AGES = (1, 2, 4, 8, 16)
NA = len(AGES)
RING = 17
NCLS = 5
CLASS_NAMES = ("prompt", "written", "win", "far", "later")
WINDOW = 16
SAMPLES_PER_CLASS = 8
HIST_BINS = np.concatenate([[-np.inf], np.linspace(-5, 1, 49), [np.inf]])   # log10 relative L2; 50 bins
NB = len(HIST_BINS) - 1
THRESHOLDS = (0.05, 0.10)
SIG_NAMES = ("dkv", "dk", "dv", "E_fresh", "E_cq", "WM_acc")


class Capture:
    def __init__(self, blocks, prompt_len: int, total_len: int, mask_id: int, eos_ids: tuple[int, ...], block_length: int = 32, seed: int = 23,
                 with_cq: bool = True):
        self.blocks = list(blocks)
        self.L = len(self.blocks)
        self.P, self.T, self.mask_id, self.eos_ids, self.B = prompt_len, total_len, mask_id, tuple(eos_ids), block_length
        self.with_cq = with_cq
        cfg = self.blocks[0].config
        self.nh, self.nkv = cfg.n_heads, cfg.effective_n_kv_heads
        self.device = next(self.blocks[0].parameters()).device
        self.gen = torch.Generator().manual_seed(seed)
        dev, T = self.device, self.T
        self.t = -1
        self.xs = torch.zeros(RING, T, dtype=torch.long, device=dev)
        self.pos = torch.arange(T, device=dev)
        self.ringK = self.ringV = self.ringQ = None   # allocated at first call (needs head dim)
        self.accE = [torch.zeros(RING, T, device=dev) for _ in range(self.L)]
        self.accEcq = [torch.zeros(RING, T, device=dev) for _ in range(self.L)]
        self.accWM = [torch.zeros(RING, T, device=dev) for _ in range(self.L)]
        self.trk_row = torch.zeros(self.L, RING, dtype=torch.long, device=dev)
        self.union_tracks: dict[int, torch.Tensor] = {}
        self.write_call = torch.full((T,), -1, dtype=torch.long, device=dev)
        self.hist = torch.zeros(self.L, NA, NCLS, NB, dtype=torch.long, device=dev)
        self.calls: list[dict] = []          # per-call arrays moved to CPU
        self._cur: dict = {}
        self._orig = []
        self.noise_mode = False
        self.noise_store: dict = {}

    # ------------------------------------------------------------------ install / remove
    def install(self, model):
        self._handle = model.register_forward_pre_hook(self._pre_call, with_kwargs=True)
        self._handle2 = model.register_forward_hook(self._post_call)
        for l, block in enumerate(self.blocks):
            orig = block._scaled_dot_product_attention
            self._orig.append(orig)

            def wrapped(q, k, v, *args, _l=l, _orig=orig, **kwargs):
                if not self.noise_mode:
                    self._layer(_l, q, k, v)
                else:
                    self.noise_store.setdefault(_l, []).append((q.detach().clone(), k.detach().clone(), v.detach().clone()))
                return _orig(q, k, v, *args, **kwargs)
            block._scaled_dot_product_attention = wrapped

    def remove(self):
        self._handle.remove()
        self._handle2.remove()
        for block, orig in zip(self.blocks, self._orig):
            block._scaled_dot_product_attention = orig
        self._orig = []

    # ------------------------------------------------------------------ per call (layer independent)
    def _pre_call(self, _module, args, kwargs):
        if self.noise_mode:
            return
        x = (args[0] if args else kwargs["input_ids"])[0]
        self.t += 1
        t, R, T, P = self.t, RING, self.T, self.P
        dev = self.device
        self.xs[t % R] = x
        masked = x == self.mask_id
        pos = self.pos
        newly = (x != self.mask_id) & (self.write_call < 0) & (pos >= P)
        self.write_call[newly] = t
        cls = torch.zeros(T, dtype=torch.long, device=dev)
        written = (pos >= P) & ~masked
        cls[written] = 1
        mpos = torch.nonzero(masked, as_tuple=False).flatten()
        win_mask = torch.zeros(T, dtype=torch.bool, device=dev)
        if mpos.numel():
            win = mpos[:WINDOW]
            win_mask[win] = True
            blk = (pos - P) // self.B
            cur_blk = int(blk[int(mpos[0])])
            far = masked & ~win_mask & (blk == cur_blk)
            later = masked & (blk > cur_blk)
            cls[win_mask] = 2
            cls[far] = 3
            cls[later] = 4
        valid = torch.zeros(NA, T, dtype=torch.bool, device=dev)
        for si, s in enumerate(AGES):
            if t - s >= 0:
                valid[si] = self.xs[(t - s) % R] == x
        ind = torch.zeros(T, WINDOW, dtype=torch.float32, device=dev)   # column j-1: written between call t-j and call t
        for j in range(1, WINDOW + 1):
            if t - j >= 0:
                ind[:, j - 1] = ((self.xs[(t - j) % R] == self.mask_id) & ~masked).float()
        dist = torch.full((NA, T), float(T), device=dev)
        for si, s in enumerate(AGES):
            idx = torch.nonzero(ind[:, s - 1] > 0, as_tuple=False).flatten()
            if idx.numel() and t - s >= 0:
                dist[si] = (pos[:, None] - idx[None, :]).abs().min(1).values.float()
        # sampled rows: up to SAMPLES_PER_CLASS per class among rows with at least one valid pair (CPU generator, deterministic)
        anyvalid = valid.any(0)
        sel = []
        for c in range(NCLS):
            cand = torch.nonzero((cls == c) & anyvalid, as_tuple=False).flatten().cpu()
            if cand.numel() > SAMPLES_PER_CLASS:
                cand = cand[torch.randperm(cand.numel(), generator=self.gen)[:SAMPLES_PER_CLASS]]
            sel.append(cand)
        samp = torch.cat(sel).to(dev)
        onehot = torch.nn.functional.one_hot(cls, NCLS).float()   # (T, NCLS)
        ind_has = (ind.sum(0) > 0).cpu().tolist()
        self._cur = dict(t=t, ind_has=ind_has, masked=masked, cls=cls, win_mask=win_mask.float(), valid=valid, ind=ind, dist=dist, samp=samp, onehot=onehot,
                         agg=torch.zeros(self.L, NA, NCLS, 4, device=dev),
                         sv=torch.zeros(self.L, samp.numel(), NA, len(SIG_NAMES), device=dev),
                         wm_inst=torch.zeros(self.L, samp.numel(), device=dev),
                         cosh=torch.full((self.L, NA), float("nan"), device=dev), cosk=torch.full((self.L, NA), float("nan"), device=dev),
                         glob=torch.zeros(self.L, NA, NCLS, 2, device=dev))

    def _post_call(self, _module, _args, _output):
        if self.noise_mode:
            return
        c, t = self._cur, self.t
        self.union_tracks[t] = torch.unique(self.trk_row[:, t % RING])
        samp = c["samp"]
        rec = dict(t=t, samp=samp.cpu().numpy(), cls=c["cls"][samp].cpu().numpy(),
                   valid=c["valid"][:, samp].T.cpu().numpy(), dist=c["dist"][:, samp].T.cpu().numpy(),
                   is_eos=torch.isin(self.xs[t % RING][samp], torch.tensor(self.eos_ids, device=self.device)).cpu().numpy(),
                   since_written=torch.where(self.write_call[samp] >= 0, t - self.write_call[samp], -1).cpu().numpy(),
                   agg=c["agg"].cpu().numpy(), sv=c["sv"].cpu().numpy(), wm_inst=c["wm_inst"].cpu().numpy(),
                   cosh=c["cosh"].cpu().numpy(), cosk=c["cosk"].cpu().numpy(),
                   n_blank=int(c["masked"].sum()), n_new=int(c["ind"][:, 0].sum()), frac_valid=c["valid"].float().mean(1).cpu().numpy())
        self.calls.append(rec)
        self._cur = {}
        old = t - RING + 1
        self.union_tracks.pop(old - 1, None)

    # ------------------------------------------------------------------ per layer
    def _alloc(self, q, k):
        D = k.shape[1] * k.shape[-1]
        self.ringK = [torch.zeros(RING, self.T, D, dtype=k.dtype, device=self.device) for _ in range(self.L)]
        self.ringV = [torch.zeros(RING, self.T, D, dtype=k.dtype, device=self.device) for _ in range(self.L)]
        self.ringQ = [torch.zeros(RING, q.shape[1], self.T, q.shape[-1], dtype=q.dtype, device=self.device) for _ in range(self.L)]

    @torch.no_grad()
    def _layer(self, l, q, k, v):
        c = self._cur
        t, R, T = c["t"], RING, self.T
        if self.ringK is None:
            assert q.shape[1] == k.shape[1], "capture assumes as many K/V heads as query heads (LLaDA-8B)"
            self._alloc(q, k)
        nh, hd = q.shape[1], q.shape[-1]
        Kf = k[0].transpose(0, 1).reshape(T, -1)
        Vf = v[0].transpose(0, 1).reshape(T, -1)
        self.ringK[l][t % R] = Kf
        self.ringV[l][t % R] = Vf
        Kf32, Vf32 = Kf.float(), Vf.float()
        # ---- drift against each snapshot
        D3 = torch.zeros(NA, 3, T, device=self.device)
        for si, s in enumerate(AGES):
            if t - s < 0:
                continue
            Ko, Vo = self.ringK[l][(t - s) % R].float(), self.ringV[l][(t - s) % R].float()
            dk2, dv2 = ((Kf32 - Ko) ** 2).sum(1), ((Vf32 - Vo) ** 2).sum(1)
            nk2, nv2 = (Ko ** 2).sum(1), (Vo ** 2).sum(1)
            D3[si, 0] = ((dk2 + dv2) / (nk2 + nv2 + 1e-30)).sqrt()
            D3[si, 1] = (dk2 / (nk2 + 1e-30)).sqrt()
            D3[si, 2] = (dv2 / (nv2 + 1e-30)).sqrt()
        # ---- attention (fresh query), head-mean
        scale = 1.0 / math.sqrt(hd)
        qf, kf = q[0].float(), k[0].float()
        S = torch.softmax(torch.matmul(qf, kf.transpose(-1, -2)) * scale, dim=-1)   # (nh, T, T)
        Am = S.mean(0)
        wm = c["win_mask"] @ Am                                       # (T,): mass the 16 window rows put on each row
        wm_track = wm * (~c["masked"]).float()
        trk = int(torch.argmax(wm_track))
        self.trk_row[l, t % R] = trk
        ind = c["ind"]
        Ef = Am @ ind                                                 # (T, 16)
        js = [j for j in range(1, WINDOW + 1) if t - j >= 0]
        self.accE[l][t % R].zero_()
        self.accWM[l][t % R].zero_()
        self.accEcq[l][t % R].zero_()
        if js:
            ridx = torch.tensor([(t - j) % R for j in js], device=self.device)
            cols = torch.tensor([j - 1 for j in js], device=self.device)
            self.accE[l][ridx] += Ef[:, cols].T
            self.accWM[l][ridx] += wm[None, :]
        # ---- cached-query exposure
        if self.with_cq:
            for j in js:
                if not c["ind_has"][j - 1]:
                    continue
                qv = self.ringQ[l][(t - j) % R].float()
                Sv = torch.softmax(torch.matmul(qv, kf.transpose(-1, -2)) * scale, dim=-1).mean(0)
                self.accEcq[l][(t - j) % R] += Sv @ ind[:, j - 1]
        # ---- Elastic trigger cosines (rows = union of the tracked rows of all layers at the snapshot call)
        for si, s in enumerate(AGES):
            v0 = t - s
            if v0 < 0 or v0 not in self.union_tracks:
                continue
            rows = self.union_tracks[v0]
            qs_ = self.ringQ[l][v0 % R][:, rows].float()
            Kv = self.ringK[l][v0 % R].view(T, nh, hd).transpose(0, 1).float()
            past = torch.softmax(torch.matmul(qs_, Kv.transpose(-1, -2)) * scale, dim=-1)       # (nh, rows, T)
            cur = S[:, rows, :]
            c["cosh"][l, si] = torch.nn.functional.cosine_similarity(past, cur, dim=0).mean()
            c["cosk"][l, si] = torch.nn.functional.cosine_similarity(past, cur, dim=-1).mean()
        self.ringQ[l][t % R] = q[0]
        # ---- aggregates over ALL rows
        valid, onehot = c["valid"].float(), c["onehot"]                # (NA, T), (T, NCLS)
        for si in range(NA):
            vm = valid[si]
            d = D3[si, 0]
            flags = torch.stack([vm, vm * (d > THRESHOLDS[0]).float(), vm * (d > THRESHOLDS[1]).float(), vm * d], dim=1)   # (T, 4)
            c["agg"][l, si] = onehot.T @ flags
            logd = torch.log10(d.clamp_min(1e-7))
            b = torch.bucketize(logd, torch.tensor(HIST_BINS[1:-1], dtype=torch.float32, device=self.device))
            cl = c["cls"]
            sel = vm > 0
            flat = cl[sel] * NB + b[sel]
            self.hist[l, si].view(-1).index_add_(0, flat, torch.ones_like(flat))
        # ---- sampled rows
        samp = c["samp"]
        if samp.numel():
            sv = c["sv"][l]
            for si, s in enumerate(AGES):
                if t - s < 0:
                    continue
                sv[:, si, 0] = D3[si, 0][samp]
                sv[:, si, 1] = D3[si, 1][samp]
                sv[:, si, 2] = D3[si, 2][samp]
                sv[:, si, 3] = self.accE[l][(t - s) % R][samp]
                sv[:, si, 4] = self.accEcq[l][(t - s) % R][samp]
                sv[:, si, 5] = self.accWM[l][(t - s) % R][samp]
            c["wm_inst"][l] = wm[samp]

    # ------------------------------------------------------------------ outputs
    def pack(self) -> dict[str, np.ndarray]:
        """Concatenate the per-call records into dense arrays (one question)."""
        cs = self.calls
        out = dict(
            hist=self.hist.cpu().numpy(),
            call_t=np.array([r["t"] for r in cs]), n_blank=np.array([r["n_blank"] for r in cs]), n_new=np.array([r["n_new"] for r in cs]),
            agg=np.stack([r["agg"] for r in cs]), cosh=np.stack([r["cosh"] for r in cs]), cosk=np.stack([r["cosk"] for r in cs]),
            frac_valid=np.stack([r["frac_valid"] for r in cs]),
        )
        ns = [len(r["samp"]) for r in cs]
        out["samp_call"] = np.repeat(np.array([r["t"] for r in cs]), ns)
        for key in ("samp", "cls", "is_eos", "since_written"):
            out["s_" + key] = np.concatenate([r[key] for r in cs])
        out["s_valid"] = np.concatenate([r["valid"] for r in cs])            # (N, NA)
        out["s_dist"] = np.concatenate([r["dist"] for r in cs]).astype(np.float32)
        out["s_sv"] = np.concatenate([r["sv"] for r in cs], axis=1).astype(np.float32)        # (L, N, NA, 6)
        out["s_wm_inst"] = np.concatenate([r["wm_inst"] for r in cs], axis=1).astype(np.float32)   # (L, N)
        return out
