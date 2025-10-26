import argparse, os, numpy as np
import matplotlib.pyplot as plt
import json, datetime
from functools import partial
from config import SimConfig
from simulator import Simulator
from gc_algos import get_gc_policy, cat_policy
from workload import make_phased_workload

# ---------- helpers ----------
def summarize_gc_events(ev):
    if not ev: return {"zgc_ratio": 0.0, "mv_p50": 0, "mv_p95": 0, "mv_p99": 0}
    mv = sorted(e.get("moved_valid", 0) for e in ev)
    zgc = sum(1 for x in mv if x == 0)
    return {
        "zgc_ratio": zgc / len(ev),
        "mv_p50": float(np.percentile(mv, 50)),
        "mv_p95": float(np.percentile(mv, 95)),
        "mv_p99": float(np.percentile(mv, 99)),
    }

def make_b_workload(cfg, base_seed: int = 42):
    total = cfg.user_total_pages
    phases = [
        {"n_ops": int(total*1.2), "update_ratio": 0.2, "hot_ratio": 0.2, "hot_weight": 0.85, "enable_trim": False},
        {"n_ops": int(total*1.5), "update_ratio": 0.9, "hot_ratio": 0.2, "hot_weight": 0.9,  "enable_trim": False},
    ]
    return make_phased_workload(phases, cfg.user_total_pages, base_seed=base_seed)

def make_rocksdb_workload(cfg,
                          cycles: int = 3,
                          ingest_ops_factor: float = 1.0,
                          update_ops_factor: float = 1.2,
                          update_ratio: float = 0.85,
                          trim_ratio: float = 0.10,
                          hot_ratio: float = 0.25,
                          hot_weight: float = 0.90,
                          base_seed: int = 42):
    """
    LSM/compaction 느낌:
      [Ingest(write-heavy)] → [Update/Invalidate-heavy(+TRIM)] 를 cycles 번 반복.
    - ingest_ops_factor: cfg.user_total_pages 대비 ingest ops 배수
    - update_ops_factor: cfg.user_total_pages 대비 update ops 배수
    """
    total = cfg.user_total_pages
    phases = []
    seed = base_seed
    for _ in range(cycles):
        # 1) SST ingest (신규 write 위주)
        phases.append({
            "n_ops": int(total * ingest_ops_factor),
            "update_ratio": 0.2,          # 신규 많은 구간
            "hot_ratio": hot_ratio,
            "hot_weight": hot_weight,
            "enable_trim": False,
            "seed": seed
        })
        seed += 1
        # 2) compaction/overwrite (갱신·무효화·TRIM 혼합)
        phases.append({
            "n_ops": int(total * update_ops_factor),
            "update_ratio": update_ratio,  # overwrite 비중 ↑
            "hot_ratio": hot_ratio,
            "hot_weight": hot_weight,
            "enable_trim": True,
            "trim_ratio": trim_ratio,
            "seed": seed
        })
        seed += 1
    return make_phased_workload(phases, total, base_seed=base_seed)

def run_once(policy_name, three_stream=False, bg_gc_every=0, gc_thresh=0.5, recency_tau=200,
             cat_alpha=0.55, cat_beta=0.25, cat_gamma=0.15, cat_delta=0.05, seed=42):
    cfg = SimConfig()
    cfg.gc_free_block_threshold = gc_thresh
    cfg.rng_seed = seed

    if policy_name == "cat":
        from gc_algos import make_cat_probe
        sim.ssd.score_probe = make_cat_probe(cat_alpha, cat_beta, cat_gamma, cat_delta)

    if policy_name == "cat":
        policy_fn = partial(cat_policy, alpha=cat_alpha, beta=cat_beta, gamma=cat_gamma, delta=cat_delta)
    else:
        policy_fn = get_gc_policy(policy_name)

    sim = Simulator(cfg, policy_fn, enable_trace=True, bg_gc_every=bg_gc_every)

    if three_stream:
        sim.ssd.three_stream = True
        sim.ssd.hotness_mode = "recency"
        sim.ssd.recency_tau  = recency_tau

    wl = make_b_workload(cfg, base_seed=seed)
    sim.run(wl)
    waf = sim.ssd.device_write_pages / max(1, sim.ssd.host_write_pages)
    s = summarize_gc_events(sim.ssd.gc_event_log)
    return waf, sim.ssd.gc_count, s, sim

def run_once_rocks(policy_name, three_stream=True, bg_gc_every=0, gc_thresh=0.5, recency_tau=200,
                   cat_alpha=0.55, cat_beta=0.25, cat_gamma=0.15, cat_delta=0.05,
                   cycles=3, ingest_ops_factor=1.0, update_ops_factor=1.2,
                   update_ratio=0.85, trim_ratio=0.10, hot_ratio=0.25, hot_weight=0.90, seed=42):
    cfg = SimConfig()
    cfg.gc_free_block_threshold = gc_thresh
    cfg.rng_seed = seed

    if policy_name == "cat":
        policy_fn = partial(cat_policy, alpha=cat_alpha, beta=cat_beta, gamma=cat_gamma, delta=cat_delta)
    else:
        policy_fn = get_gc_policy(policy_name)

    sim = Simulator(cfg, policy_fn, enable_trace=True, bg_gc_every=bg_gc_every)

    if three_stream:
        sim.ssd.three_stream = True
        sim.ssd.hotness_mode = "recency"
        sim.ssd.recency_tau  = recency_tau

    wl = make_rocksdb_workload(cfg,
                               cycles=cycles,
                               ingest_ops_factor=ingest_ops_factor,
                               update_ops_factor=update_ops_factor,
                               update_ratio=update_ratio,
                               trim_ratio=trim_ratio,
                               hot_ratio=hot_ratio,
                               hot_weight=hot_weight,
                               base_seed=seed)
    sim.run(wl)
    waf = sim.ssd.device_write_pages / max(1, sim.ssd.host_write_pages)
    s = summarize_gc_events(sim.ssd.gc_event_log)
    return waf, sim.ssd.gc_count, s, sim

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def parse_policy_list(s: str):
    items = [p.strip() for p in s.split(",") if p.strip()]
    if len(items) == 1 and items[0].lower() == "all":
        return ["greedy", "cb", "bsgc", "cat"]
    return items

    seed_list = [int(x) for x in getattr(args, "seeds", "42").split(",") if x.strip()]


# ---------- plotting ----------
def plot_three_stream(res, out_dir):
    # res: [(flag(False/True), waf, gcs, sdict)]
    ensure_dir(out_dir)
    flags = [("OFF" if not f else "ON") for f,_,_,_ in res]
    wafs  = [w for _,w,_,_ in res]
    zgc   = [sd["zgc_ratio"] for *_, sd in res]

    # WAF bar
    plt.figure()
    plt.title("Three-Stream ON/OFF – WAF")
    plt.bar(flags, wafs)
    plt.ylabel("WAF")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "three_stream_waf.png"), dpi=150); plt.close()

    # ZGC bar
    plt.figure()
    plt.title("Three-Stream ON/OFF – Zero-copy GC Ratio")
    plt.bar(flags, zgc)
    plt.ylabel("zero-copy ratio")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "three_stream_zgc.png"), dpi=150); plt.close()

def plot_policies(res, out_dir, gc_thresh, recency_tau, seed):
    ensure_dir(out_dir)
    names = [n for n,_,_,_ in res]
    wafs  = [w for _,w,_,_ in res]
    mv95  = [sd["mv_p95"] for *_, sd in res]
    zgc   = [sd["zgc_ratio"] for *_, sd in res]
    subtitle = f"(thresh={gc_thresh}, tau={recency_tau}, seed={seed})"

    plt.figure()
    plt.title("Policies – WAF\n" + subtitle)
    plt.bar(names, wafs)
    plt.ylabel("WAF")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "policies_waf.png"), dpi=150); plt.close()

    plt.figure()
    plt.title("Policies – moved_valid p95\n" + subtitle)
    plt.bar(names, mv95)
    plt.ylabel("moved_valid (p95)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "policies_mv_p95.png"), dpi=150); plt.close()

    plt.figure()
    plt.title("Policies – Zero-copy GC Ratio\n" + subtitle)
    plt.bar(names, zgc)
    plt.ylabel("zero-copy ratio")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "policies_zgc.png"), dpi=150); plt.close()

def save_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def save_result_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def plot_bg_pacing(res, out_dir):
    # res: [(k, waf, p99_ms, gcs, zgc)]
    ensure_dir(out_dir)
    ks   = [k for k,_,_,_,_ in res]
    wafs = [w for _,w,_,_,_ in res]
    p99  = [p for *_, p,_,_ in res]
    zgc  = [z for *_, _,_, z in res]

    # WAF vs k
    plt.figure()
    plt.title("BG GC pacing – WAF vs k")
    plt.plot(ks, wafs, marker="o")
    plt.xlabel("bg_gc_every (ops)")
    plt.ylabel("WAF")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bg_waf.png"), dpi=150); plt.close()

    # p99 vs k
    plt.figure()
    plt.title("BG GC pacing – GC p99(ms) vs k")
    plt.plot(ks, p99, marker="o")
    plt.xlabel("bg_gc_every (ops)")
    plt.ylabel("gc p99 (ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bg_p99.png"), dpi=150); plt.close()

    # zgc vs k
    plt.figure()
    plt.title("BG GC pacing – Zero-copy ratio vs k")
    plt.plot(ks, zgc, marker="o")
    plt.xlabel("bg_gc_every (ops)")
    plt.ylabel("zero-copy ratio")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bg_zgc.png"), dpi=150); plt.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc_thresh", type=float, default=0.5)
    ap.add_argument("--policies", default="all")
    ap.add_argument("--bg_list", default="0,200,500,1000")
    ap.add_argument("--recency_tau", type=int, default=200)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--cat_alpha", type=float, default=0.55)
    ap.add_argument("--cat_beta", type=float, default=0.25)
    ap.add_argument("--cat_gamma", type=float, default=0.15)
    ap.add_argument("--cat_delta", type=float, default=0.05)
    ap.add_argument("--mode", required=True, choices=["three_stream", "policies", "bg_pacing", "cat_ablation", "tau_sweep", "rocksdb"])
    ap.add_argument("--tau_list", default="50,100,200,300,400")
    ap.add_argument("--rocks_policies", default="all")
    ap.add_argument("--rocks_cycles", type=int, default=3)
    ap.add_argument("--rocks_ingest_ops_factor", type=float, default=1.0)
    ap.add_argument("--rocks_update_ops_factor", type=float, default=1.2)
    ap.add_argument("--rocks_update_ratio", type=float, default=0.85)
    ap.add_argument("--rocks_trim_ratio", type=float, default=0.10)
    ap.add_argument("--rocks_hot_ratio", type=float, default=0.25)
    ap.add_argument("--rocks_hot_weight", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    if args.mode == "three_stream":
        out = []
        for flag in [False, True]:
            waf, gcs, s, _ = run_once(
                "cb", three_stream=flag,
                gc_thresh=args.gc_thresh, recency_tau=args.recency_tau,
                seed=args.seed
            )
            print(f"three_stream={flag} | WAF={waf:.3f} | GCs={gcs:4d} | "
                  f"zgc={s['zgc_ratio']:.2%} | mv_p50/p95/p99={s['mv_p50']}/{s['mv_p95']}/{s['mv_p99']}")
            out.append((flag, waf, gcs, s))
        if args.plot:
            plot_three_stream(out, os.path.join(args.out_dir, "three_stream"))

    elif args.mode == "policies":
        res = []
        for name in parse_policy_list(args.policies):
            waf, gcs, s, _ = run_once(
                name, three_stream=True,
                gc_thresh=args.gc_thresh, recency_tau=args.recency_tau,
                cat_alpha=args.cat_alpha, cat_beta=args.cat_beta,
                cat_gamma=args.cat_gamma, cat_delta=args.cat_delta,
                seed=args.seed
            )
            print(f"{name:>7} | WAF={waf:.3f} | GCs={gcs:4d} | "
                  f"zgc={s['zgc_ratio']:.2%} | mv_p50/p95/p99={s['mv_p50']}/{s['mv_p95']}/{s['mv_p99']}")
            res.append((name, waf, gcs, s))
        if args.plot:
            plot_policies(res, os.path.join(args.out_dir, "policies"),
                          gc_thresh=args.gc_thresh, recency_tau=args.recency_tau, seed=args.seed)


    elif args.mode == "bg_pacing":
        res = []
        for k in [int(x) for x in args.bg_list.split(",") if x.strip()]:
            waf, gcs, s, sim = run_once(
                "cb", three_stream=True, bg_gc_every=k,
                gc_thresh=args.gc_thresh, recency_tau=args.recency_tau,
                seed=args.seed
            )
            p99 = np.percentile(sim.ssd.gc_durations, 99) * 1000 if sim.ssd.gc_durations else 0.0
            print(f"bg_gc_every={k:4d} | WAF={waf:.3f} | gc_p99_ms={p99:.3f} | GCs={gcs:4d} | zgc={s['zgc_ratio']:.2%}")
            res.append((k, waf, p99, gcs, s["zgc_ratio"]))
        if args.plot:
            plot_bg_pacing(res, os.path.join(args.out_dir, "bg_pacing"))

    elif args.mode == "cat_ablation":
        # 기준값
        base = dict(alpha=args.cat_alpha, beta=args.cat_beta, gamma=args.cat_gamma, delta=args.cat_delta)

        cases = [
            ("cat_base", base["alpha"], base["beta"], base["gamma"], base["delta"]),
            ("cat_no_alpha", 0.0, base["beta"], base["gamma"], base["delta"]),
            ("cat_no_beta",  base["alpha"], 0.0, base["gamma"], base["delta"]),
            ("cat_no_gamma", base["alpha"], base["beta"], 0.0, base["delta"]),
            ("cat_no_delta", base["alpha"], base["beta"], base["gamma"], 0.0),
        ]

        res = []
        for name, a, b, g, d in cases:
            waf, gcs, s, _ = run_once("cat", three_stream=True, gc_thresh=args.gc_thresh,
                                      recency_tau=args.recency_tau,
                                      cat_alpha=a, cat_beta=b, cat_gamma=g, cat_delta=d)
            print(f"{name:>12} | WAF={waf:.3f} | GCs={gcs:4d} | zgc={s['zgc_ratio']:.2%} "
                  f"| mv_p50/p95/p99={s['mv_p50']}/{s['mv_p95']}/{s['mv_p99']}")
            res.append((name, waf, s))

        if args.plot:
            out_dir = os.path.join(args.out_dir, "cat_ablation")
            ensure_dir(out_dir)

            # WAF 막대
            labels = [n for n,_,_ in res]
            wafs   = [w for _,w,_ in res]
            plt.figure(); plt.title("CAT Ablation – WAF"); plt.bar(labels, wafs); plt.ylabel("WAF"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ablation_waf.png"), dpi=150); plt.close()

            # moved_valid p95 막대
            mv95 = [sd["mv_p95"] for *_, sd in res]
            plt.figure(); plt.title("CAT Ablation – moved_valid p95"); plt.bar(labels, mv95); plt.ylabel("mv_p95"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ablation_mv_p95.png"), dpi=150); plt.close()

            # zero-copy 비율 막대
            zgc = [sd["zgc_ratio"] for *_, sd in res]
            plt.figure(); plt.title("CAT Ablation – Zero-copy GC Ratio"); plt.bar(labels, zgc); plt.ylabel("zero-copy"); plt.ylim(0,1); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "ablation_zgc.png"), dpi=150); plt.close()

    elif args.mode == "tau_sweep":
        taus = [int(x) for x in args.tau_list.split(",") if x.strip()]
        res = []
        for tau in taus:
            waf, gcs, s, _ = run_once("cat", three_stream=True, gc_thresh=args.gc_thresh,
                                      recency_tau=tau,
                                      cat_alpha=args.cat_alpha, cat_beta=args.cat_beta,
                                      cat_gamma=args.cat_gamma, cat_delta=args.cat_delta,
                                      seed=args.seed)
            print(f"tau={tau:4d} | WAF={waf:.3f} | GCs={gcs:4d} | zgc={s['zgc_ratio']:.2%} "
                  f"| mv_p50/p95/p99={s['mv_p50']}/{s['mv_p95']}/{s['mv_p99']}")
            res.append((tau, waf, s))

        if args.plot:
            out_dir = os.path.join(args.out_dir, "tau_sweep")
            ensure_dir(out_dir)

            taus = [t for t,_,_ in res]
            wafs = [w for _,w,_ in res]
            mv95 = [sd["mv_p95"] for *_, sd in res]
            zgc  = [sd["zgc_ratio"] for *_, sd in res]

            plt.figure(); plt.title("tau sweep – WAF"); plt.plot(taus, wafs, marker="o"); plt.xlabel("recency_tau"); plt.ylabel("WAF"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "tau_waf.png"), dpi=150); plt.close()

            plt.figure(); plt.title("tau sweep – moved_valid p95"); plt.plot(taus, mv95, marker="o"); plt.xlabel("recency_tau"); plt.ylabel("mv_p95"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "tau_mv_p95.png"), dpi=150); plt.close()

            plt.figure(); plt.title("tau sweep – Zero-copy ratio"); plt.plot(taus, zgc, marker="o"); plt.xlabel("recency_tau"); plt.ylabel("zero-copy"); plt.ylim(0,1); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "tau_zgc.png"), dpi=150); plt.close()

    elif args.mode == "rocksdb":
        res = []
        for name in parse_policy_list(args.rocks_policies):
            waf, gcs, s, _ = run_once_rocks(
                name, three_stream=True,
                gc_thresh=args.gc_thresh, recency_tau=args.recency_tau,
                cat_alpha=args.cat_alpha, cat_beta=args.cat_beta,
                cat_gamma=args.cat_gamma, cat_delta=args.cat_delta,
                cycles=args.rocks_cycles,
                ingest_ops_factor=args.rocks_ingest_ops_factor,
                update_ops_factor=args.rocks_update_ops_factor,
                update_ratio=args.rocks_update_ratio,
                trim_ratio=args.rocks_trim_ratio,
                hot_ratio=args.rocks_hot_ratio,
                hot_weight=args.rocks_hot_weight,
                seed=args.seed
            )
            print(f"[rocks] {name:>7} | WAF={waf:.3f} | GCs={gcs:4d} | "
                  f"zgc={s['zgc_ratio']:.2%} | mv_p50/p95/p99={s['mv_p50']}/{s['mv_p95']}/{s['mv_p99']}")
            res.append((name, waf, gcs, s))

        if args.plot:
            out_dir = os.path.join(args.out_dir, "rocksdb")
            ensure_dir(out_dir)
            # WAF
            names = [n for n,_,_,_ in res]
            wafs  = [w for _,w,_,_ in res]
            plt.figure(); plt.title("RocksDB-like – WAF"); plt.bar(names, wafs); plt.ylabel("WAF"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rocks_waf.png"), dpi=150); plt.close()
            # moved_valid p95
            mv95 = [sd["mv_p95"] for *_, sd in res]
            plt.figure(); plt.title("RocksDB-like – moved_valid p95"); plt.bar(names, mv95); plt.ylabel("mv_p95"); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rocks_mv_p95.png"), dpi=150); plt.close()
            # zero-copy
            zgc = [sd["zgc_ratio"] for *_, sd in res]
            plt.figure(); plt.title("RocksDB-like – Zero-copy GC Ratio"); plt.bar(names, zgc); plt.ylabel("zero-copy"); plt.ylim(0,1); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "rocks_zgc.png"), dpi=150); plt.close()

    elif args.mode == "policies":
        res = []
        for name in parse_policy_list(args.policies):
            wafs, gcss, zgcs, mv95s = [], [], [], []
            for sd in seed_list:
                waf, gcs, s, sim = run_once(
                    name, three_stream=True,
                    gc_thresh=args.gc_thresh, recency_tau=args.recency_tau,
                    cat_alpha=args.cat_alpha, cat_beta=args.cat_beta,
                    cat_gamma=args.cat_gamma, cat_delta=args.cat_delta,
                    seed=sd
                )
                wafs.append(waf);
                gcss.append(gcs);
                zgcs.append(s["zgc_ratio"]);
                mv95s.append(s["mv_p95"])

            # 통계
            def mean(x):
                return float(np.mean(x))

            def std(x):
                return float(np.std(x))

            aggregate = {
                "policy": name, "seeds": seed_list,
                "waf_mean": mean(wafs), "waf_std": std(wafs),
                "gcs_mean": mean(gcss), "gcs_std": std(gcss),
                "zgc_mean": mean(zgcs), "zgc_std": std(zgcs),
                "mv95_mean": mean(mv95s), "mv95_std": std(mv95s),
                "cfg": {
                    "gc_thresh": args.gc_thresh, "recency_tau": args.recency_tau,
                    "cat": dict(alpha=args.cat_alpha, beta=args.cat_beta, gamma=args.cat_gamma, delta=args.cat_delta)
                },
                "timestamp": datetime.datetime.now().isoformat()
            }
            # 저장
            out_dir = os.path.join(args.out_dir, "policies_json")
            save_result_json(os.path.join(out_dir, f"{name}.json"), aggregate)
            print(f"{name:>7} | WAF={aggregate['waf_mean']:.3f}±{aggregate['waf_std']:.3f} "
                  f"| GCs={aggregate['gcs_mean']:.1f} "
                  f"| ZGC={aggregate['zgc_mean']:.2%}")
            res.append((name, aggregate["waf_mean"], aggregate["gcs_mean"],
                        {"mv_p95": aggregate["mv95_mean"], "zgc_ratio": aggregate["zgc_mean"]}))
        if args.plot:
            plot_policies(res, os.path.join(args.out_dir, "policies"),
                          gc_thresh=args.gc_thresh, recency_tau=args.recency_tau, seed=seed_list[0])

if __name__ == "__main__":
    main()