#!/usr/bin/env python3
import csv
import math
import os
from collections import defaultdict

DATA_DIR = "datos"
OUTPUT_DIR = "outputs"


def to_int(x):
    return int(x)


def parse_seed(seed_text):
    digits = "".join(ch for ch in seed_text if ch.isdigit())
    return int(digits) if digits else 20


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_team_features(prefix):
    detailed_path = os.path.join(DATA_DIR, f"{prefix}RegularSeasonDetailedResults.csv")
    seeds_path = os.path.join(DATA_DIR, f"{prefix}NCAATourneySeeds.csv")

    rows = read_rows(detailed_path)
    rows.sort(key=lambda r: (to_int(r["Season"]), to_int(r["DayNum"])))

    agg = defaultdict(lambda: {
        "games": 0,
        "wins": 0,
        "pts_for": 0.0,
        "pts_against": 0.0,
        "margin": 0.0,
        "poss_for": 0.0,
        "poss_against": 0.0,
        "opp": [],
    })

    elo = defaultdict(lambda: 1500.0)
    last_season = None
    K = 20.0
    home_adv = 80.0

    for r in rows:
        season = to_int(r["Season"])
        if last_season is None or season != last_season:
            elo = defaultdict(lambda: 1500.0)
            last_season = season

        w = to_int(r["WTeamID"])
        l = to_int(r["LTeamID"])
        ws = to_int(r["WScore"])
        ls = to_int(r["LScore"])
        loc = r["WLoc"]

        w_poss = to_int(r["WFGA"]) - to_int(r["WOR"]) + to_int(r["WTO"]) + 0.475 * to_int(r["WFTA"])
        l_poss = to_int(r["LFGA"]) - to_int(r["LOR"]) + to_int(r["LTO"]) + 0.475 * to_int(r["LFTA"])

        w_key = (season, w)
        l_key = (season, l)

        agg[w_key]["games"] += 1
        agg[w_key]["wins"] += 1
        agg[w_key]["pts_for"] += ws
        agg[w_key]["pts_against"] += ls
        agg[w_key]["margin"] += (ws - ls)
        agg[w_key]["poss_for"] += max(w_poss, 1.0)
        agg[w_key]["poss_against"] += max(l_poss, 1.0)
        agg[w_key]["opp"].append(l)

        agg[l_key]["games"] += 1
        agg[l_key]["pts_for"] += ls
        agg[l_key]["pts_against"] += ws
        agg[l_key]["margin"] += (ls - ws)
        agg[l_key]["poss_for"] += max(l_poss, 1.0)
        agg[l_key]["poss_against"] += max(w_poss, 1.0)
        agg[l_key]["opp"].append(w)

        ra = elo[w]
        rb = elo[l]
        adj = 0.0
        if loc == "H":
            adj = home_adv
        elif loc == "A":
            adj = -home_adv
        exp_w = 1.0 / (1.0 + 10 ** (((rb - (ra + adj)) / 400.0)))
        elo[w] = ra + K * (1.0 - exp_w)
        elo[l] = rb + K * (0.0 - (1.0 - exp_w))

    seed_lookup = {}
    for r in read_rows(seeds_path):
        seed_lookup[(to_int(r["Season"]), to_int(r["TeamID"]))] = parse_seed(r["Seed"])

    feat = {}
    for key, a in agg.items():
        season, team = key
        g = max(a["games"], 1)
        win_pct = a["wins"] / g
        avg_margin = a["margin"] / g
        off_eff = 100.0 * a["pts_for"] / max(a["poss_for"], 1.0)
        def_eff = 100.0 * a["pts_against"] / max(a["poss_against"], 1.0)
        net_eff = off_eff - def_eff

        opp_win_rates = []
        for opp in a["opp"]:
            ok = (season, opp)
            if ok in agg:
                og = max(agg[ok]["games"], 1)
                opp_win_rates.append(agg[ok]["wins"] / og)
        sos = sum(opp_win_rates) / max(len(opp_win_rates), 1)

        seed = seed_lookup.get((season, team), 20)

        feat[key] = {
            "win_pct": win_pct,
            "avg_margin": avg_margin,
            "off_eff": off_eff,
            "def_eff": def_eff,
            "net_eff": net_eff,
            "sos": sos,
            "seed": seed,
            "elo": elo[team],
        }

    return feat


def build_examples(prefix, feat):
    tourney_path = os.path.join(DATA_DIR, f"{prefix}NCAATourneyCompactResults.csv")
    rows = read_rows(tourney_path)
    examples = []

    for r in rows:
        season = to_int(r["Season"])
        w = to_int(r["WTeamID"])
        l = to_int(r["LTeamID"])

        t1, t2 = (w, l) if w < l else (l, w)
        label = 1.0 if w == t1 else 0.0

        f1 = feat.get((season, t1))
        f2 = feat.get((season, t2))
        if f1 is None or f2 is None:
            continue

        x = [
            f1["win_pct"] - f2["win_pct"],
            f1["avg_margin"] - f2["avg_margin"],
            f1["net_eff"] - f2["net_eff"],
            f1["sos"] - f2["sos"],
            (f2["seed"] - f1["seed"]) / 16.0,
            (f1["elo"] - f2["elo"]) / 400.0,
        ]
        examples.append((season, x, label))

    return examples


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logistic(train_xy, lr=0.05, epochs=900, l2=1e-3):
    n_features = len(train_xy[0][0])
    w = [0.0] * n_features
    b = 0.0
    n = len(train_xy)

    for _ in range(epochs):
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for x, y in train_xy:
            z = b + sum(wi * xi for wi, xi in zip(w, x))
            p = sigmoid(z)
            err = p - y
            for i in range(n_features):
                grad_w[i] += err * x[i]
            grad_b += err
        for i in range(n_features):
            grad_w[i] = grad_w[i] / n + l2 * w[i]
            w[i] -= lr * grad_w[i]
        b -= lr * (grad_b / n)
    return w, b


def predict_prob(w, b, x):
    return sigmoid(b + sum(wi * xi for wi, xi in zip(w, x)))


def log_loss(y_true, y_prob):
    eps = 1e-15
    total = 0.0
    for y, p in zip(y_true, y_prob):
        p = min(max(p, eps), 1 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / max(len(y_true), 1)


def seasonal_validation(examples, first_val_season=2010):
    seasons = sorted({s for s, _, _ in examples if s >= first_val_season})
    results = []
    for val_season in seasons:
        train = [(x, y) for s, x, y in examples if s < val_season]
        val = [(x, y) for s, x, y in examples if s == val_season]
        if not train or not val:
            continue
        w, b = fit_logistic(train)
        probs = [predict_prob(w, b, x) for x, _ in val]
        ys = [y for _, y in val]
        results.append((val_season, log_loss(ys, probs), len(val)))
    return results


def train_and_predict(prefix, sample_ids):
    feat = build_team_features(prefix)
    examples = build_examples(prefix, feat)

    val_results = seasonal_validation(examples)

    train_all = [(x, y) for _, x, y in examples if _ < 2026]
    w, b = fit_logistic(train_all)

    preds = {}
    for season, t1, t2 in sample_ids:
        if season != 2026:
            continue
        f1 = feat.get((season, t1))
        f2 = feat.get((season, t2))
        if f1 is None or f2 is None:
            preds[(season, t1, t2)] = 0.5
            continue
        x = [
            f1["win_pct"] - f2["win_pct"],
            f1["avg_margin"] - f2["avg_margin"],
            f1["net_eff"] - f2["net_eff"],
            f1["sos"] - f2["sos"],
            (f2["seed"] - f1["seed"]) / 16.0,
            (f1["elo"] - f2["elo"]) / 400.0,
        ]
        preds[(season, t1, t2)] = predict_prob(w, b, x)

    return val_results, preds


def read_sample_ids():
    rows = read_rows(os.path.join(DATA_DIR, "SampleSubmissionStage2.csv"))
    out = []
    for r in rows:
        season, t1, t2 = [int(p) for p in r["ID"].split("_")]
        out.append((season, t1, t2))
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_ids = read_sample_ids()
    men_ids = [x for x in sample_ids if x[1] < 3000 and x[2] < 3000]
    women_ids = [x for x in sample_ids if x[1] >= 3000 and x[2] >= 3000]

    m_val, m_preds = train_and_predict("M", men_ids)
    w_val, w_preds = train_and_predict("W", women_ids)

    pred_lookup = {}
    pred_lookup.update(m_preds)
    pred_lookup.update(w_preds)

    sub_path = os.path.join(OUTPUT_DIR, "submission_stage2.csv")
    with open(os.path.join(DATA_DIR, "SampleSubmissionStage2.csv"), newline="", encoding="utf-8") as f_in, \
         open(sub_path, "w", newline="", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=["ID", "Pred"])
        writer.writeheader()
        for r in reader:
            season, t1, t2 = [int(p) for p in r["ID"].split("_")]
            p = pred_lookup.get((season, t1, t2), 0.5)
            writer.writerow({"ID": r["ID"], "Pred": f"{p:.6f}"})

    report_path = os.path.join(OUTPUT_DIR, "validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Validation log loss by season\n")
        f.write("Men:\n")
        for season, ll, n in m_val:
            f.write(f"  {season}: logloss={ll:.5f}, games={n}\n")
        f.write("Women:\n")
        for season, ll, n in w_val:
            f.write(f"  {season}: logloss={ll:.5f}, games={n}\n")
        if m_val:
            f.write(f"Men average logloss: {sum(x[1] for x in m_val)/len(m_val):.5f}\n")
        if w_val:
            f.write(f"Women average logloss: {sum(x[1] for x in w_val)/len(w_val):.5f}\n")

    print(f"Saved: {sub_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
