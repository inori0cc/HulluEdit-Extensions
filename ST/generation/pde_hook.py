"""Add PDE at the Omni output head after the existing pre-RMSNorm edit."""
import torch
from torch.nn import functional as F

from component_editor import Intervention, edit
from pde import solve_direction, validate_logits


class PDEIntervention(Intervention):
    def __init__(self, model, audio_mask, cfg, alpha):
        assert cfg.placement == 'pre_norm'
        super().__init__(model.model, audio_mask, cfg)
        self.norm = model.model.norm
        self.alpha = alpha
        self.reference = None
        self.handles.append(model.lm_head.register_forward_hook(self.edit_logits))

    def transform(self, hidden):
        state = hidden[0, -1].detach()
        changed, stats, bases = edit(state, self.visual, self.text, self.cfg, return_bases=True)
        stats.update(step=len(self.trace), pde='gate_closed', argmax_changed=False)
        self.trace.append(stats)
        self.reference = (state.clone(), bases)
        result = hidden.clone()
        result[0, -1] = changed
        return result

    def edit_logits(self, head, args, output):
        state, (evidence, prior) = self.reference
        self.reference = None
        stats = self.trace[-1]
        if not stats['active']:
            return output
        stats['pde'] = 'fallback'
        if not prior.shape[1] or not evidence.shape[1]:
            return output
        if (evidence.T @ prior).abs().max().item() > 1e-5:
            return output
        if (prior.T @ prior - torch.eye(prior.shape[1], device=prior.device)).abs().max().item() > 1e-5:
            return output
        # Invoke the norm implementation directly so the HulluEdit hook does
        # not fire for the counterfactual unedited state.
        z0 = F.linear(self.norm.forward(state[None]), head.weight, head.bias)[0]
        zh = output[0, -1]
        xh = prior.T @ args[0][0, -1].float()
        projected = torch.cat([rows.float() @ prior for rows in head.weight.split(4096)])
        def host(tensor):
            return tensor.float().cpu().numpy()

        base, reference = host(z0), host(zh)
        update = solve_direction(base, reference, host(projected), host(xh), beta=0.5)
        if update is None:
            return output
        candidate = (zh.float() + torch.as_tensor(self.alpha * update, dtype=torch.float32, device=zh.device)).to(zh.dtype)
        if not validate_logits(base, reference, host(candidate), beta=0.5):
            return output
        stats.update(pde='accepted', argmax_changed=bool(candidate.argmax() != zh.argmax()))
        result = output.clone()
        result[0, -1] = candidate
        return result
