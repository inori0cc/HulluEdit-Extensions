"""Audio HulluEdit subspace editing and decoder hooks."""

from dataclasses import dataclass, replace
import torch


@dataclass(frozen=True)
class Config:
    name: str = 'post_full'
    placement: str = 'post_norm'
    anchor_layer: int = 26
    r: int = 8
    q: int = 5
    kappa: float = .60
    lambda0: float = .26
    lambda_max: float = 3.6
    eps: float = 1e-6
    gamma_v: float = .10
    gamma_p: float = .50
    text_window: int = 512
    blend: float = 1.
    norm_beta: float = 0.
    component: str = "full"


BASE = Config()


def basis(a, rank, eps):
    if not a.numel() or rank == 0:
        return a.new_zeros((a.shape[-1], 0))
    # For an [n,d] row-state matrix, the hidden-space basis is the RIGHT
    # singular vectors, dimension [d,k], with no covariance centering in Eq 2.
    _, s, vh = torch.linalg.svd(a, full_matrices=False)
    if not s.numel() or s[0] <= 0:
        return a.new_zeros((a.shape[-1], 0))
    keep = min(rank, int((s > s[0]*eps).sum().item()))
    return vh[:keep].T


def edit(h, visual, text, c, return_bases=False):
    original_dtype = h.dtype
    h, visual, text = h.float(), visual.float(), text.float()
    weights = ((visual@h)/(visual.norm(dim=-1)*h.norm()+c.eps)).softmax(0)
    u = basis(weights.sqrt()[:, None]*visual, c.r, c.eps)
    projected = text - (text@u)@u.T
    p = basis(projected, c.q, c.eps)
    if p.shape[1]:
        p = p-u@(u.T@p)
        p, _ = torch.linalg.qr(p, mode='reduced')
    hu, hp = u@(u.T@h), p@(p.T@h)
    hr = h-hu-hp
    energy = h.square().sum()+c.eps
    vcr, pcr = hu.square().sum()/energy, hp.square().sum()/energy
    ln = (c.kappa*(1-vcr)/(vcr+c.eps)).clamp(0, c.lambda_max)
    lp = (c.lambda0*pcr/(1-pcr+c.eps)).clamp(0, c.lambda_max)
    low_v, high_p = bool(vcr<c.gamma_v), bool(pcr>c.gamma_p)
    active = low_v or high_p
    scheduled_ln, scheduled_lp = ln, lp
    if c.component == 'prior_only':
        ln = ln*0
    elif c.component == 'non_evidence_only':
        lp = lp*0
    elif c.component != 'full':
        raise ValueError(c.component)
    # Return the exact original state for the identity control, avoiding
    # cancellation/rounding in hu+hp+hr when both strengths are zero.
    closed = h if c.lambda_max == 0 else hu+hp/(1+ln+lp)+hr/(1+ln)
    raw = closed if active else h
    transformed = raw
    if active and c.norm_beta:
        transformed = transformed*(h.norm()/(transformed.norm()+c.eps)).pow(c.norm_beta)
    final = h+c.blend*(transformed-h) if c.blend != 1. else transformed
    # Correct only when the original FP32 invariant would fail.
    projection_corrected = False
    if active and not c.norm_beta and u.shape[1]:
        error = (u.T @ (final-h)).norm() / (h.norm() + c.eps)
        if float(error) >= 1e-4:
            ud = u.double()
            delta = final.double() - h.double()
            coeff = torch.linalg.solve(ud.T @ ud, ud.T @ delta)
            final = (h.double() + delta - ud @ coeff).float()
            projection_corrected = True
            # Verify the corrected edit with the same high precision.
            corrected_error = (ud.T @ (final.double()-h.double())).norm()
    final_native = final.to(original_dtype)
    orth = float((u.T@p).abs().max()) if p.shape[1] and u.shape[1] else 0.
    evidence_delta = corrected_error if projection_corrected else (u.T@(final-h)).norm()
    stats = dict(projection_corrected=projection_corrected, component=c.component, scheduled_lambda_n=float(scheduled_ln), scheduled_lambda_p=float(scheduled_lp), vcr=float(vcr), pcr=float(pcr), active=active,
        low_vcr_gate=low_v, high_pcr_gate=high_p,
        lambda_n=float(ln), lambda_p=float(lp),
        lambda_n_at_cap=bool(ln >= c.lambda_max-1e-6) if c.lambda_max else False,
        norm_before=float(h.norm()), norm_closed=float(raw.norm()), norm_after=float(final.norm()),
        raw_norm_ratio=float(raw.norm()/(h.norm()+c.eps)),
        final_norm_ratio=float(final.norm()/(h.norm()+c.eps)),
        relative_change=float((final-h).norm()/(h.norm()+c.eps)),
        evidence_error_relative_h=float(evidence_delta/(h.norm()+c.eps)),
        evidence_error_relative_evidence=float(evidence_delta/(hu.norm()+c.eps)),
        orthogonality_error=orth, evidence_rank=u.shape[1], prior_rank=p.shape[1],
        text_cache_rows=text.shape[0], audio_rows=visual.shape[0])
    if not torch.isfinite(final_native).all():
        raise FloatingPointError('Nonfinite edited state')
    assert orth < 1e-4, stats
    # Global rescaling deliberately does not preserve the evidence component.
    if not c.norm_beta:
        assert stats['evidence_error_relative_h'] < 1e-4, stats
    if return_bases:
        return final_native, stats, (u, p)
    return final_native, stats


class Intervention:
    def __init__(self, decoder, audio_mask, cfg):
        self.cfg, self.audio_mask = cfg, audio_mask
        self.visual, self.text, self.trace, self.handles = None, None, [], []
        self.pending = None
        self.handles.append(decoder.layers[cfg.anchor_layer-1].register_forward_hook(self.capture))
        if cfg.placement == 'post_norm':
            self.handles.append(decoder.norm.register_forward_hook(self.after_norm))
        elif cfg.placement == 'pre_norm':
            self.handles.append(decoder.norm.register_forward_pre_hook(self.before_norm))
        else:
            raise ValueError(cfg.placement)

    def capture(self, module, args, output):
        z = output[0] if isinstance(output, tuple) else output
        z = z[0].detach().float()
        if self.visual is None:
            assert len(self.audio_mask) == z.shape[0], 'Audio mask length mismatch'
            mask = self.audio_mask.to(z.device)
            assert mask.any()
            self.visual = z[mask].clone()
            # The state being edited must not estimate its own prior subspace.
            self.text = z[:-1][~mask[:-1]][-self.cfg.text_window:].clone()
            self.pending = z[-1:].clone()
        else:
            assert z.shape[0] == 1, 'Prepared runner supports one sequence with KV cache'
            self.text = torch.cat([self.text, self.pending])[-self.cfg.text_window:]
            self.pending = z.clone()

    def transform(self, hidden):
        changed, stats = edit(hidden[0, -1].detach(), self.visual, self.text, self.cfg)
        stats['step'] = len(self.trace)
        self.trace.append(stats)
        result = hidden.clone()
        result[0, -1] = changed
        return result

    def after_norm(self, module, args, output):
        return self.transform(output)

    def before_norm(self, module, args):
        return (self.transform(args[0]),)+args[1:]

    def close(self):
        for handle in self.handles:
            handle.remove()


def numerical_checks(device='cpu'):
    torch.manual_seed(0)
    h = torch.randn(64, device=device)
    visual, text = torch.randn(24, 64, device=device), torch.randn(18, 64, device=device)
    out, _ = edit(h, visual, text, replace(BASE, lambda_max=0.))
    assert torch.equal(out, h), 'Zero-edit algebra is not identity'
    torch.manual_seed(0)
    full, s = edit(h, visual, text, BASE)
    assert s['evidence_error_relative_h'] < 1e-5
    torch.manual_seed(0)
    blended, sb = edit(h, visual, text, replace(BASE, blend=.1))
    assert torch.allclose(blended, h+.1*(full-h), atol=1e-6)
    assert sb['evidence_error_relative_h'] < 1e-5
    # Positive global logit scaling does not change exact greedy argmax.
    head = torch.randn(200, 64, device=device)
    assert (head@full).argmax() == (head@(3.*full)).argmax()
    # Compare the closed form with an independent dense solution of Eq 14.
    u = basis(visual, 8, BASE.eps)
    p = basis(text-(text@u)@u.T, 5, BASE.eps)
    p = p-u@(u.T@p)
    p, _ = torch.linalg.qr(p, mode='reduced')
    uu, pp = u@u.T, p@p.T
    eye = torch.eye(len(h),device=device)
    ln, lp = .7, .2
    dense = torch.linalg.solve(eye+ln*(eye-uu)+lp*pp,h)
    closed = uu@h+(pp@h)/(1+ln+lp)+((eye-uu-pp)@h)/(1+ln)
    assert torch.allclose(dense,closed,atol=2e-5,rtol=2e-5)
    # Verify cache timing without loading a model: current state is deferred.
    hook = Intervention.__new__(Intervention)
    hook.cfg = BASE
    hook.visual = hook.text = hook.pending = None
    hook.audio_mask = torch.tensor([False,True,True,False,False],device=device)
    z = torch.arange(5*8,device=device,dtype=torch.float32).reshape(1,5,8)
    hook.capture(None,None,z)
    assert torch.equal(hook.visual,z[0,1:3])
    assert torch.equal(hook.text,torch.stack([z[0,0],z[0,3]]))
    nxt = torch.ones(1,1,8,device=device)*100
    hook.capture(None,None,nxt)
    assert torch.equal(hook.text[-1],z[0,-1])
    assert torch.equal(hook.pending,nxt[0])
    return dict(zero_edit_identity=True, evidence_preservation=True,
                blend_identity=True, positive_scaling_argmax=True,
                dense_objective_solution=True,previous_only_cache=True)
