"""Hullu-anchored P-ball minimax editing; no token labels or model calls."""
import time
import numpy as np
from scipy import sparse
from scipy.special import softmax
import clarabel

TOL=1e-6


class SolverFailure(RuntimeError):
    pass


def problem_data(G,z0,zh,xh,temperature):
    G,z0,zh,xh=[np.asarray(v,dtype=np.float64) for v in (G,z0,zh,xh)]
    if not all(np.isfinite(v).all() for v in (G,z0,zh,xh)):
        raise SolverFailure('nonfinite solver input')
    if temperature<=0 or G.shape!=(len(z0),len(xh)) or zh.shape!=z0.shape:
        raise ValueError('invalid dimensions or temperature')
    d=(zh-z0)/temperature; Eref=float(np.ptp(d)/2)
    g=softmax(zh/temperature)-softmax(z0/temperature)
    g-=g.sum()/max(1,len(g))
    signal=float(np.abs(g).sum()); r=float(np.linalg.norm(xh))
    gn=g/max(signal,1e-300)
    a=(G*r/temperature).T@gn
    ascale=float(np.linalg.norm(a))
    return G,z0,zh,xh,d,Eref,g,signal,r,gn,a,ascale


def check_candidate(G,z0,zh,xh,x,zp,temperature,radius_factor=1.,preference_retention=1.):
    if not (radius_factor>=1 and 0<=preference_retention<=1):
        raise ValueError('diagnostic parameters must retain Hullu feasibility')
    G,z0,zh,xh,d,Eh,g,signal,r,gn,a,ascale=problem_data(G,z0,zh,xh,temperature)
    x=np.asarray(x,dtype=float); zp=np.asarray(zp,dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(zp).all():
        return dict(accepted=False,reason='nonfinite_candidate')
    E=float(np.ptp((zp-z0)/temperature)/2)
    margin=float(gn@((zp-zh)/temperature))
    progress=float(gn@d)
    slack=margin+(1-preference_retention)*progress
    normalized=slack/(ascale if ascale>1e-12 else 1.)
    ratio=float(np.linalg.norm(x)/r) if r else float(np.linalg.norm(x))
    reason='accepted'
    if ratio>radius_factor*(1.+TOL): reason='ball_numeric'
    elif normalized < -TOL: reason='preference_numeric'
    elif E>=Eh-TOL*max(1.,Eh): reason='no_actual_improvement'
    return dict(accepted=reason=='accepted',reason=reason,E_actual=E,E_ref=Eh,
                E_ratio=E/Eh if Eh else 1.,preference_margin=margin,
                preference_normalized=normalized,p_norm_ratio=ratio,
                preference_slack=slack,preference_reference_progress=progress,
                preference_retention_actual=(1+margin/progress) if progress>1e-12 else None)


def _cone(D,f,yh,a,active,cap=None,radius_factor=1.,preference_rhs=None):
    k=D.shape[1]; secondary=cap is not None; n=k+1 if secondary else k+2
    Da=D[active]; fa=f[active]
    if secondary:
        linear=np.vstack([np.c_[Da,-np.ones(len(active))],np.c_[-Da,np.ones(len(active))]])
        rhs=np.r_[cap-fa,cap+fa]
        H=sparse.diags(np.r_[np.full(k,2.),0.],format='csc')
        cost=np.r_[-2*yh,0.]
    else:
        linear=np.vstack([np.c_[Da,-np.ones(len(active)),-np.ones(len(active))],
                          np.c_[-Da,np.ones(len(active)),-np.ones(len(active))],
                          np.r_[np.zeros(k+1),-1.][None,:]])
        rhs=np.r_[-fa,fa,0.]
        H=sparse.csc_matrix((n,n)); cost=np.r_[np.zeros(k+1),1.]
    if np.linalg.norm(a)>0:
        row=np.zeros(n); row[:k]=-a
        linear=np.vstack([linear,row]); rhs=np.r_[rhs,-a@yh if preference_rhs is None else -preference_rhs]
    soc=np.zeros((k+1,n)); soc[1:,:k]=-np.eye(k)
    A=sparse.vstack([sparse.csc_matrix(linear),sparse.csc_matrix(soc)],format='csc')
    b=np.r_[rhs,radius_factor,np.zeros(k)]
    settings=clarabel.DefaultSettings(); settings.verbose=False; settings.max_iter=200
    settings.tol_feas=1e-8; settings.tol_gap_abs=1e-8; settings.tol_gap_rel=1e-8
    if hasattr(settings,'max_threads'): settings.max_threads=1
    sol=clarabel.DefaultSolver(H,cost,A,b,
        [clarabel.NonnegativeConeT(len(rhs)),clarabel.SecondOrderConeT(k+1)],settings).solve()
    vals=np.asarray(sol.x,dtype=float)
    gap=abs(sol.obj_val-sol.obj_val_dual)/max(1.,abs(sol.obj_val),abs(sol.obj_val_dual))
    if (str(sol.status) not in ('Solved','AlmostSolved') or not np.isfinite(vals).all()
            or max(sol.r_prim,sol.r_dual,gap)>TOL):
        raise SolverFailure(f'{sol.status}: primal={sol.r_prim}, dual={sol.r_dual}, gap={gap}')
    return vals[:k],float(vals[k]),float(cap if secondary else vals[-1]),dict(
        status=str(sol.status),iterations=int(sol.iterations),active_rows=len(active),gap=float(gap),
        ball_dual=float(sol.z[len(rhs)]),
        preference_dual=float(sol.z[len(rhs)-1]) if np.linalg.norm(a)>0 else 0.)


def _all_rows(D,f,yh,a,initial,cap=None,radius_factor=1.,preference_rhs=None):
    active=np.array(sorted(set(initial)),dtype=int); records=[]
    for it in range(17):
        if it==16: active=np.arange(len(f))
        y,c,E,record=_cone(D,f,yh,a,active,cap,radius_factor,preference_rhs)
        records.append(record)
        violation=np.abs(D@y+f-c)-E
        if float(violation.max())<=TOL*max(1.,abs(E)):
            rhs=a@yh if preference_rhs is None else preference_rhs
            if np.linalg.norm(y)>radius_factor*(1+TOL) or a@y<rhs-TOL:
                raise SolverFailure('ball or preference feasibility')
            return y,records
        outside=np.setdiff1d(np.where(violation>TOL*max(1.,abs(E)))[0],active,assume_unique=True)
        if not len(outside): raise SolverFailure('active constraint violation')
        order=np.lexsort((outside,-violation[outside]))
        active=np.union1d(active,outside[order[:16]])
    raise SolverFailure('full vocabulary certification failed')


def solve_direction(G,z0,zh,xh,temperature=.05,radius_factor=1.,preference_retention=1.):
    if not (radius_factor>=1 and 0<=preference_retention<=1):
        raise ValueError('diagnostic parameters must retain Hullu feasibility')
    start=time.monotonic()
    G,z0,zh,xh,d,Eh,g,signal,r,gn,a,ascale=problem_data(G,z0,zh,xh,temperature)
    base=dict(x=xh.copy(),E=Eh,E_primary=Eh,E_ref=Eh,solver_calls=0,
              secondary_used=False,signal_l1=signal,solve_seconds=0.)
    if not len(xh) or r==0 or Eh==0 or signal<=1e-12:
        return dict(base,reason='degenerate_reference')
    yh=xh/r
    a=a/ascale if ascale>1e-12 else np.zeros_like(a)
    progress=float(gn@d)
    if progress < -TOL*max(1.,Eh):
        raise SolverFailure('negative reference preference progress')
    preference_rhs=float(a@yh-(1-preference_retention)*progress/ascale) if ascale>1e-12 else 0.
    if radius_factor==1 and preference_retention==1 and np.linalg.norm(a)>0 and np.linalg.norm(a-yh)<1e-12:
        return dict(base,reason='singleton')
    # Remove harmless common shifts for numerical conditioning only.
    D=G*r/temperature
    D-=D.mean(axis=0,keepdims=True)
    f=d-D@yh; f-=f.mean()
    scale=max(1.,Eh); Ds=D/scale; fs=f/scale
    initial=[int(np.argmax(d)),int(np.argmin(d)),int(np.argmax(np.linalg.norm(Ds,axis=1)))]
    y,records=_all_rows(Ds,fs,yh,a,initial,radius_factor=radius_factor,preference_rhs=preference_rhs)
    primary=float(np.ptp(D@y+f)/2)
    primary_record=records[-1].copy()
    if primary>Eh+TOL*scale: raise SolverFailure('primary worse than reference')
    cap=min(primary+TOL*scale,Eh)/scale
    secondary_used=False; secondary_error=None
    try:
        ys,recs=_all_rows(Ds,fs,yh,a,initial,cap=cap,radius_factor=radius_factor,preference_rhs=preference_rhs)
        if float(np.ptp(Ds@ys+fs)/2)<=cap+TOL:
            y=ys; records+=recs; secondary_used=True
    except SolverFailure as exc:
        secondary_error=str(exc)
    E=float(np.ptp(D@y+f)/2)
    return dict(x=r*y,E=E,E_primary=primary,E_ref=Eh,signal_l1=signal,
                solver_calls=len(records),secondary_used=secondary_used,
                secondary_error=secondary_error,reason='solved',
                primary_ball_dual=primary_record['ball_dual'],
                primary_preference_dual=primary_record['preference_dual'],
                preference_alignment_with_xh=float(a@yh),
                continuous_preference_slack=float(a@y-preference_rhs),
                continuous_ball_slack=float(radius_factor-np.linalg.norm(y)),
                solve_seconds=time.monotonic()-start)
