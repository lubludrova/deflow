"""Shared return summaries and the MuJoCo sensitivity table."""

from return_metrics import ENVS, METRICS, MODELS, mean_ci, metric_summary

TABLE_MODELS = ["gauss", "flows", "flowe", "dime", "qsm"]
NAMES = {key: name for key, name, _ in MODELS}

def summary(runs):
    return {**{metric: metric_summary(runs, metric) for metric in METRICS},
            "at950k": mean_ci([r["return_950k"] for r in runs if r["return_950k"] is not None])}

def table_cell(cell, best=False):
    if cell["mean"] is None:
        return "--"
    val = f'{cell["mean"]:.0f}'
    if best:
        val = r"\mathbf{" + val + "}"
    if cell["ci95"] is not None:
        val += rf' \pm {(cell["ci95"][1]-cell["ci95"][0])/2:.0f}'
    return rf'${val}$ ({cell["n"]})'

def sensitivity_table(cells):
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Environment & Method & Last 100k mean & Return AUC & Return at 950k \\", r"\midrule"]
    for env, label in ENVS:
        if env not in cells:
            continue
        models = [m for m in TABLE_MODELS if m in cells[env]]
        for i, model in enumerate(models):
            row = [label if i == 0 else "", NAMES[model]]
            row += [table_cell(cells[env][model]["summary"][m]) for m in ("tail100k", "auc", "at950k")]
            lines.append(" & ".join(row) + r" \\")
        if env != ENVS[-1][0]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)
