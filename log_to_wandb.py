#!/usr/bin/env python
"""Follow a train_unified.py log and mirror it to wandb, without touching the job.

The trainer has no wandb hook, and restarting a running 48 h job to add one
would throw away its progress. This tails logs/uni-<job>.out from the start,
logs every training row and every eval line under the trainer's own step, and
keeps following until the job leaves the queue. Re-running it with the same
--run_id resumes the same wandb run, so the dashboard has no duplicates.

  python log_to_wandb.py --job 16535 --name dual_lj_ls960
"""
import argparse
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = re.compile(r"^\s+(\d+)\s+([-\d.naif]+)\s+([-\d.naif]+)\s+([-\d.naif]+)\s+([-\d.naif]+)"
                 r"\s+([-\d.naif]+)\s+([-\d.naif]+)\s+([-\d.naif]+)%\s+([-\d.naif]+)\s+([-\d.naif]+)")
EVAL = re.compile(r"\[eval (\d+) (raw|ema)\] TTS mel err/trivial ([-\d.]+).*?"
                  r"S2T text err/trivial ([-\d.]+) WER ([-\d.]+)% tok_acc ([-\d.]+)%")
TUSE = re.compile(r"TTS text use .*?: (.*)$")
COLS = ["loss", "l2", "l2_over_trivial", "tts_l2_over_trivial", "s2t_l2_over_trivial",
        "ce", "ce_acc", "grad_norm", "sec_per_it"]


def in_queue(job):
    r = subprocess.run(["squeue", "-j", str(job), "-h", "-o", "%i"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="voiceflow-unified")
    ap.add_argument("--run_id", default="")
    a = ap.parse_args()
    import wandb
    log_path = os.path.join(HERE, "logs", f"uni-{a.job}.out")
    run = wandb.init(project=a.project, name=a.name, id=a.run_id or f"{a.name}-{a.job}",
                     resume="allow", config={"slurm_job": a.job, "log": log_path})
    for k in ["train/*", "eval_ema/*", "eval_raw/*"]:
        wandb.define_metric(k, step_metric="step")
    wandb.define_metric("step")
    last = run.summary.get("step", 0) or 0          # resume: skip what is logged
    pos, buf, idle = 0, "", 0
    last_eval = None
    while True:
        with open(log_path) as f:
            f.seek(pos)
            chunk = f.read()
            pos = f.tell()
        buf += chunk
        lines = buf.split("\n")
        buf = lines.pop()
        for line in lines:
            m = ROW.match(line)
            if m:
                st = int(m.group(1))
                if st <= last:
                    continue
                vals = {}
                for c, v in zip(COLS, m.groups()[1:]):
                    try:
                        vals[f"train/{c}"] = float(v)
                    except ValueError:
                        pass
                wandb.log({"step": st, **vals})
                last = st
                continue
            m = TUSE.search(line)
            if m and last_eval is not None:
                vals = {f"eval_ema/tts_text_use_{k}": float(v) for k, v in
                        re.findall(r"t=([\d.]+) ([\d.]+)", m.group(1))}
                wandb.log({"step": last_eval, **vals})
                continue
            m = EVAL.search(line)
            if m:
                st, tag = int(m.group(1)), m.group(2)
                last_eval = st
                if st < last - 1 and run.resumed:
                    continue
                wandb.log({"step": st,
                           f"eval_{tag}/tts_mel_err_over_trivial": float(m.group(3)),
                           f"eval_{tag}/s2t_err_over_trivial": float(m.group(4)),
                           f"eval_{tag}/s2t_wer": float(m.group(5)),
                           f"eval_{tag}/s2t_tok_acc": float(m.group(6))})
        if not chunk:
            idle += 1
            if idle % 10 == 0 and not in_queue(a.job):
                break
            time.sleep(30)
        else:
            idle = 0
    wandb.finish()


if __name__ == "__main__":
    main()
