"""Preference-constrained direction editing on an affine output head."""
import numpy as np
from scipy import sparse
from scipy.special import softmax


def half_range(values):
    return float(np.ptp(values) / 2)


def solve_direction(z0, zh, projected_head, prior_coordinates, rho=1.25, beta=0.5):
    """Return the logit update, or None when the reference should be retained."""
    import clarabel

    z0, zh, projected_head, prior_coordinates = [
        np.asarray(value, dtype=np.float64)
        for value in (z0, zh, projected_head, prior_coordinates)
    ]
    if not all(np.isfinite(v).all() for v in (z0, zh, projected_head, prior_coordinates)):
        return None
    radius = np.linalg.norm(prior_coordinates)
    difference = zh - z0
    reference_e = half_range(difference)
    g = softmax(zh) - softmax(z0)
    if radius == 0 or reference_e == 0 or np.abs(g).sum() <= 1e-12:
        return None
    reference_y = prior_coordinates / radius
    matrix = projected_head * radius
    offset = difference - matrix @ reference_y
    preference = g @ matrix
    threshold = preference @ reference_y - (1 - beta) * (g @ difference)
    preference_scale = max(np.linalg.norm(preference), abs(threshold), 1e-12)
    scale = max(1., reference_e)
    tolerance = 1e-6
    k = len(reference_y)

    def search(secondary_cap=None):
        order = np.argsort(difference)
        rows = np.union1d(order[:8], order[-8:])
        for attempt in range(17):
            if attempt == 16:
                rows = np.arange(len(difference))
            a = matrix[rows] / scale
            b = offset[rows] / scale
            ones = np.ones((len(rows), 1))
            inequalities = np.vstack([
                np.hstack([a, -ones, -ones]),
                np.hstack([-a, ones, -ones]),
                np.r_[-preference / preference_scale, 0, 0],
                np.r_[np.zeros(k + 1), -1],
            ])
            bounds = np.r_[-b, b, -threshold / preference_scale, 0.]
            if secondary_cap is not None:
                inequalities = np.vstack([inequalities, np.r_[np.zeros(k + 1), 1]])
                bounds = np.r_[bounds, secondary_cap / scale]
            cone = np.zeros((k + 1, k + 2))
            cone[1:, :k] = -np.eye(k)
            hessian = np.zeros(k + 2)
            linear = np.r_[np.zeros(k + 1), 1.]
            if secondary_cap is not None:
                hessian[:k] = 1
                linear = np.r_[-reference_y, 0., 0.]
            settings = clarabel.DefaultSettings()
            settings.verbose = False
            settings.max_iter = 200
            settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-8
            solver = clarabel.DefaultSolver(
                sparse.diags(hessian, format="csc"), linear,
                sparse.csc_matrix(np.vstack([inequalities, cone])),
                np.r_[bounds, rho, np.zeros(k)],
                [clarabel.NonnegativeConeT(len(bounds)), clarabel.SecondOrderConeT(k + 1)],
                settings,
            )
            solution = solver.solve()
            if str(solution.status) not in {"Solved", "AlmostSolved"}:
                return None
            y, center, objective = np.split(np.asarray(solution.x), [k, k + 1])
            if not np.isfinite(solution.x).all() or np.linalg.norm(y) > rho + tolerance:
                return None
            if preference @ y < threshold - tolerance * preference_scale:
                return None
            values = offset + matrix @ y
            violation = np.abs(values / scale - center[0]) - objective[0]
            if float(violation.max()) <= tolerance:
                actual_e = half_range(values)
                if secondary_cap is None or actual_e <= secondary_cap + tolerance * scale:
                    return y, actual_e
                return None
            worst = np.argsort(violation)[-16:]
            rows = np.union1d(rows, worst[violation[worst] > tolerance])
        return None

    primary = search()
    if primary is None or primary[1] >= reference_e - tolerance * scale:
        return None
    secondary = search(min(primary[1] + tolerance * scale, reference_e))
    y = (secondary if secondary is not None else primary)[0]
    return matrix @ (y - reference_y)


def validate_logits(z0, zh, candidate, beta=0.5):
    """Check the emitted, rounded logits against the same-prefix references."""
    z0, zh, candidate = [np.asarray(x, dtype=np.float64) for x in (z0, zh, candidate)]
    if not np.isfinite(candidate).all():
        return False
    g = softmax(zh) - softmax(z0)
    progress = g @ (zh - z0)
    reference_e = half_range(zh - z0)
    return bool(
        progress > 0
        and g @ (candidate - z0) >= beta * progress - 1e-8 * progress
        and half_range(candidate - z0) < reference_e - 1e-6 * max(1., reference_e)
    )
