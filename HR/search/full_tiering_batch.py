import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_asset_class(asset_notebook_path: Path):
    nb = json.loads(asset_notebook_path.read_text())
    source_lines = nb["cells"][0]["source"]
    cleaned = []
    for line in source_lines:
        if line.strip().startswith("%run"):
            continue
        cleaned.append(line)

    ns = {}
    exec("\n".join(cleaned), ns, ns)
    return ns["ASSET"]


def line_metrics(asset_cls, param, target, line_label):
    star_path = Path(param["dataset"]) / target
    search = asset_cls(parameters=param, line=line_label)
    spec_param = search.spec_analysis(str(star_path) + "/")
    if spec_param is None:
        return {"ok": False, "reason": "not_enough_spectra"}

    new_spectra, med, med_err = spec_param
    ref_spec = med[search.snr_idxrange]
    threshold = float(param["threshold"])
    peak_threshold = abs(threshold)
    width_filter = int(param["width_filt"])

    rows = []
    for i, spec in enumerate(new_spectra):
        snr = search.snr(spec, med, search.spectra_err[i], med_err)
        sd = np.std(snr)
        if (not np.isfinite(sd)) or sd == 0:
            continue

        sig = snr[search.snr_idxrange] / sd
        filtered_rv = search.radial_velocity[search.snr_idxrange]
        filtered_spec = spec[search.snr_idxrange]

        min_sigma = float(np.nanmin(sig))
        max_sigma = float(np.nanmax(sig))
        rv_at_min = float(filtered_rv[sig == min_sigma][0])

        width = 0
        if min_sigma < threshold:
            try:
                width = int(search.get_width(sig))
            except Exception:
                width = 0

        abs_depth = np.nan
        try:
            arr = (ref_spec[filtered_rv == rv_at_min] - filtered_spec[filtered_rv == rv_at_min]) / ref_spec[filtered_rv == rv_at_min]
            if len(arr) > 0:
                abs_depth = float(arr[0])
        except Exception:
            pass

        rows.append(
            {
                "epoch": i + 1,
                "min_sigma": min_sigma,
                "max_sigma": max_sigma,
                "rv_at_min": rv_at_min,
                "width": width,
                "abs_depth": abs_depth,
                "is_dip": bool((min_sigma < threshold) and (width >= width_filter)),
                "is_peak": bool(max_sigma >= peak_threshold),
            }
        )

    if len(rows) == 0:
        return {"ok": False, "reason": "invalid_snr"}

    df = pd.DataFrame(rows)
    dip_df = df[df["is_dip"]]
    strongest = df.loc[int(df["min_sigma"].idxmin())]

    return {
        "ok": True,
        "n_epochs": int(len(df)),
        "n_dip_epochs": int(dip_df.shape[0]),
        "n_peak_epochs": int(df["is_peak"].sum()),
        "strongest_min_sigma": float(strongest["min_sigma"]),
        "strongest_rv": float(strongest["rv_at_min"]),
        "strongest_width": int(strongest["width"]),
        "median_abs_depth_dips": float(np.nanmedian(dip_df["abs_depth"])) if dip_df.shape[0] > 0 else np.nan,
    }


def tier_and_reason(param, km, hm):
    threshold = float(param["threshold"])

    if (not km["ok"]) or (not hm["ok"]):
        return "Tier 3", "missing_or_low_quality_spectra"

    k_has = km["n_dip_epochs"] > 0
    h_has = hm["n_dip_epochs"] > 0
    dual_line = k_has and h_has

    rv_match = False
    if dual_line:
        rv_match = abs(km["strongest_rv"] - hm["strongest_rv"]) <= 25

    total_epochs = max(km["n_epochs"], hm["n_epochs"])
    total_dips = km["n_dip_epochs"] + hm["n_dip_epochs"]
    transient = total_dips <= max(2, int(np.ceil(0.35 * total_epochs)))

    narrow = (km["strongest_width"] <= 6) and (hm["strongest_width"] <= 6)
    strong_sig = (km["strongest_min_sigma"] <= threshold - 1.0) and (hm["strongest_min_sigma"] <= threshold - 0.7)
    peaky = (km["n_peak_epochs"] > 0) or (hm["n_peak_epochs"] > 0)
    too_many = total_dips >= max(4, int(np.ceil(0.60 * total_epochs)))

    score = 0
    if dual_line:
        score += 2
    if rv_match:
        score += 1
    if transient:
        score += 1
    if narrow:
        score += 1
    if strong_sig:
        score += 1
    if peaky:
        score -= 1
    if too_many:
        score -= 1

    reasons = [
        "dual_line" if dual_line else "single_line_or_weak_crossline",
        "rv_matched" if rv_match else "rv_not_matched",
        "transient" if transient else "repeated/persistent",
        "narrow" if narrow else "broad_or_mixed",
        "strong_sigma" if strong_sig else "marginal_sigma",
    ]
    if peaky:
        reasons.append("strong_positive_peaks_present")

    if score >= 5:
        tier = "Tier 1"
    elif score >= 3:
        tier = "Tier 2"
    else:
        tier = "Tier 3"

    return tier, "; ".join(reasons)


def main():
    root = Path(__file__).resolve().parents[2]
    search_dir = root / "HR" / "search"
    results_dir = root / "HR" / "results"

    param = json.loads((search_dir / "param.json").read_text())

    base_name = "candidates_{}sig_{}cut_{}width".format(param["threshold"], param["cutoff"], param["width_filt"])
    qsv2 = results_dir / "QuickSearch_V2"
    matches = sorted([p for p in qsv2.glob(base_name + "*.npy")], key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError("No QuickSearch_V2 candidate file matched {}".format(base_name))

    targets = [str(x) for x in np.load(matches[0], allow_pickle=True).tolist()]
    print("Using candidate file:", matches[0].name)
    print("Total targets:", len(targets))

    asset_cls = load_asset_class(search_dir / "asset.ipynb")

    out_rows = []
    for idx, target in enumerate(targets, start=1):
        try:
            km = line_metrics(asset_cls, param, target, "K")
            hm = line_metrics(asset_cls, param, target, "H")
            tier, reason = tier_and_reason(param, km, hm)
            out_rows.append(
                {
                    "target": target,
                    "tier": tier,
                    "reason": reason,
                    "K_ok": km.get("ok", False),
                    "H_ok": hm.get("ok", False),
                    "K_n_dips": km.get("n_dip_epochs", np.nan),
                    "H_n_dips": hm.get("n_dip_epochs", np.nan),
                    "K_strongest_min_sigma": km.get("strongest_min_sigma", np.nan),
                    "H_strongest_min_sigma": hm.get("strongest_min_sigma", np.nan),
                    "K_strongest_rv": km.get("strongest_rv", np.nan),
                    "H_strongest_rv": hm.get("strongest_rv", np.nan),
                    "K_width": km.get("strongest_width", np.nan),
                    "H_width": hm.get("strongest_width", np.nan),
                    "K_peaks": km.get("n_peak_epochs", np.nan),
                    "H_peaks": hm.get("n_peak_epochs", np.nan),
                }
            )
        except Exception as exc:
            out_rows.append(
                {
                    "target": target,
                    "tier": "Tier 3",
                    "reason": "analysis_error: {}".format(exc),
                    "K_ok": False,
                    "H_ok": False,
                }
            )

        if idx % 50 == 0:
            print("Processed {}/{}".format(idx, len(targets)))

    df = pd.DataFrame(out_rows).sort_values(["tier", "target"]).reset_index(drop=True)

    assessed = results_dir / "Assessed"
    assessed.mkdir(parents=True, exist_ok=True)

    csv_path = assessed / "all_spectra_tiers_full.csv"
    md_path = assessed / "all_spectra_tiers_full.md"

    df.to_csv(csv_path, index=False)

    with md_path.open("w") as f:
        f.write("# Full Spectra Tiering (K + H)\n\n")
        f.write("Total targets analyzed: {}\n\n".format(len(df)))
        for tier in ["Tier 1", "Tier 2", "Tier 3"]:
            sub = df[df["tier"] == tier]
            f.write("## {} ({})\n\n".format(tier, len(sub)))
            for _, row in sub.iterrows():
                f.write("- {}: {}\n".format(row["target"], row["reason"]))
            f.write("\n")

    print("Done")
    print("Saved:", csv_path)
    print("Saved:", md_path)
    print(df["tier"].value_counts().to_string())


if __name__ == "__main__":
    main()
