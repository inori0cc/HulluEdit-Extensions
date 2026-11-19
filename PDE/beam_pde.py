"""Apply PDE independently to each beam."""
import time
import numpy as np
import torch
from direction_solver import solve_direction, check_candidate, SolverFailure


ALPHA = 0.25
BETA = 0.5
RADIUS_FACTOR = 1.25


def numpy64(t):
    return t.detach().double().cpu().numpy()


@torch.no_grad()
def edit_beam(self, raw, hullu_logits, U, P, result):
    radius_factor = RADIUS_FACTOR
    preference_retention = BETA
    parameters = dict(radius_factor=radius_factor, preference_retention=preference_retention)
    assert raw.ndim == hullu_logits.ndim == 1 and raw.shape == hullu_logits.shape
    assert U.shape[1] == 8 and P.shape[1] == 5
    if not torch.isfinite(raw).all() or not torch.isfinite(hullu_logits).all():
        raise FloatingPointError("Nonfinite input logits")
    rec = dict(accepted=False, rank_p=P.shape[1])
    Un,Pn,hn=numpy64(U),numpy64(P),numpy64(result.h_edited)
    eu=float(np.linalg.norm(Un.T@Un-np.eye(U.shape[1]))/np.sqrt(U.shape[1]))
    ep=float(np.linalg.norm(Pn.T@Pn-np.eye(P.shape[1]))/np.sqrt(P.shape[1]))
    ec=float(np.linalg.norm(Un.T@Pn)/np.sqrt(P.shape[1]))
    geom_tol=max(1e-5,torch.finfo(U.dtype).eps,torch.finfo(P.dtype).eps)
    rec.update(u_orth_error=eu,p_orth_error=ep,up_cross_error=ec,geometry_tolerance=geom_tol)
    if max(eu,ep,ec)>geom_tol:
        rec['reason']='degenerate_geometry'; return hullu_logits, rec
    T=1.0  # Beam search uses log_softmax of the unscaled logits.
    # Preserve the original FP16 temperature operation. Certify the very
    # logits fed to softmax, including head-output/temperature rounding.
    z0=numpy64(raw/T); zh=numpy64(hullu_logits/T)
    xh=Pn.T@hn
    from scipy.special import softmax
    signal=float(np.abs(softmax(zh)-softmax(z0)).sum())
    Eh=float(np.ptp(zh-z0)/2)
    rec.update(signal_l1=signal,E_ref=Eh)
    if signal<=1e-12 or not np.linalg.norm(xh) or Eh==0:
        rec['reason']='degenerate_reference'; return hullu_logits, rec
    if not hasattr(self,'_direction_W'):
        self._direction_W=self.model.lm_head.weight.detach().float()
    G=self._direction_W@P.float()
    Gn=numpy64(G)/T
    started=time.monotonic()
    try:
        solution=solve_direction(Gn,z0,zh,xh,temperature=1.,**parameters)
    except SolverFailure as exc:
        rec.update(reason='solver_numeric',solver_error=str(exc),solve_seconds=time.monotonic()-started)
        return hullu_logits, rec
    rec.update({k:solution[k] for k in ('E_primary','solver_calls','secondary_used','solve_seconds')})
    rec.update({k:solution[k] for k in ('primary_ball_dual','primary_preference_dual',
               'preference_alignment_with_xh','continuous_preference_slack','continuous_ball_slack') if k in solution})
    delta=torch.as_tensor(solution['x']-xh,device=G.device,dtype=torch.float32)
    actual_x=xh+numpy64(delta)
    candidate=(hullu_logits.float()+G@delta).to(hullu_logits.dtype)
    check=check_candidate(Gn,z0,zh,xh,actual_x,numpy64(candidate/T),1.,**parameters)
    rec.update(check)
    change=Pn@numpy64(delta)
    denom=max(float(np.linalg.norm(change)),1e-30)
    leak=float(np.linalg.norm(Un.T@change)/denom)
    rec['relative_visual_leakage']=leak
    rec['p_change_norm']=float(np.linalg.norm(actual_x-xh))
    nx=float(np.linalg.norm(actual_x)); nh=float(np.linalg.norm(xh))
    rec['p_cosine']=float(actual_x@xh/(nx*nh)) if nx*nh else None
    projected_ratio=float(np.linalg.norm(Pn.T@(hn+change))/nh) if nh else 0.
    rec['actual_p_projection_ratio']=projected_ratio
    if leak>geom_tol*np.sqrt(P.shape[1])*(1+1e-6):
        rec.update(accepted=False,reason='hidden_numeric')
    if projected_ratio>radius_factor+geom_tol*np.sqrt(P.shape[1]):
        rec.update(accepted=False,reason='projection_numeric')
    if rec['accepted']:
        rec['final_top']=int(candidate.argmax())
        rec['logits_changed']=not torch.equal(candidate,hullu_logits)
        errors=numpy64(candidate/T)-z0
        idx=[int(np.argmax(errors)),int(np.argmin(errors))]
        p0=softmax(z0)
        rec['extreme_raw_probability']=[float(p0[j]) for j in idx]
        rec['extreme_token_ids']=idx
        if ALPHA != 1:
            candidate = torch.lerp(hullu_logits.float(), candidate.float(), ALPHA).to(hullu_logits.dtype)
        return candidate, rec
    return hullu_logits, rec
