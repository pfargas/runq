# runq

Resumable parameter sweeps on a SQLite queue. You write **one plain function with keyword
defaults** that returns a dict of scalars; runq owns the grid, the queue, the workers, the
resumability, and the results table.

```python
# cs_point.py — the entire project-side contract
def run_point(L=0.8, N=5, lr=3e-3, n_epochs=2000, seed=0, run_dir=None):
    ...  # your physics
    return {"e_per_n": e, "err_per_n": err}
```

```bash
runq run cs_point.py --axis L=0.5,0.8,1.2 --axis N=2,5,10 --seeds 0 1 2 --gpus 0,1
runq status
runq failed
runq table --group L,N --sort e_per_n
```

There is **no physics/hyperparameter split at the interface** — one flat `--axis`
vocabulary. Defaults in the signature give the types used to coerce CLI values. `run_dir`,
if present in the signature, receives a per-run artifact directory. Raise `runq.Skip("why")`
inside the function to record a point as skipped rather than failed.

Design (decided 2026-07-02, extracted from qvarnet's `soft_sphere_gas` + `cs_sweep`):

- **Key** = canonical JSON of the fully resolved parameter dict; enqueue is idempotent
  (`INSERT OR IGNORE`), so re-running a sweep extends it and skips `done` points.
- **Claim** = `BEGIN IMMEDIATE` on a WAL SQLite DB: multiple workers on one node never run
  the same point. `requeue` resets rows left `running` by a crash.
- **Local backend**: one worker subprocess per GPU (`CUDA_VISIBLE_DEVICES` pinned,
  CPU threads capped so workers don't thrash each other).
- **Results**: a single `result_json` column; `runq.load_table(conn)` expands params +
  results into a pandas DataFrame (the only place pandas is needed). `runq table` is that
  same DataFrame from the shell, for when you don't want to open a notebook:

  ```bash
  runq table                                    # done rows: axes + results
  runq table --group L,N --sort e_per_n         # seeds averaged away: mean + _sem + n
  runq table --where L=0.5 --where N>=5         # =, !=, <, <=, >, >= ; repeatable (ANDed)
  runq table --status failed                    # or --status all
  runq table --cols L,N,e_per_n --csv out.csv   # full table to disk, not the terminal
  ```

  `--group` aggregates **results only** — parameter axes are dropped, never averaged, and
  `n` reports how many runs went into each mean. If `n` exceeds your seed count you pooled
  over an axis you forgot to name, which is the number to check before trusting the error
  bars. Needs the extra: `pip install 'runq[table]'`.
- **Merge**: rows are keyed, so multi-machine sweeps merge with done-precedence
  (`runq merge merged.db pc1.db pc2.db`).
- **SLURM**: no backend needed — a job is just an sbatch script wrapping `runq run`:

  ```bash
  #!/bin/bash
  #SBATCH --gres=gpu:2 --cpus-per-task=16 --time=24:00:00
  runq run point.py --axis L=0.5,0.8 --seeds 0 1 2 --db "$SLURM_TMPDIR/outputs/cs.db"
  rsync -a "$SLURM_TMPDIR/outputs/" ~/sweep/outputs/
  ```

  Put the DB on node-local scratch (`$SLURM_TMPDIR`) and rsync back — SQLite locking is
  unreliable on NFS/Lustre. Time-limit kills are fine: the next submission requeues
  interrupted points. To spread one sweep over several nodes, partition by an axis
  (e.g. one job per seed, each with its own `--db`) and `runq merge` afterwards.

- **Email notification**: `runq run ... --notify` emails you when the drain finishes
  (fully drained / N failed / stopped with unfinished); `runq notify --db X` sends the
  current status on demand (drop it at the end of an sbatch script); `runq notify --test`
  verifies the setup. Stdlib SMTP; configure once in `~/.config/runq/notify.toml`:

  ```toml
  [email]
  to = "you@example.com"
  smtp_host = "smtp.gmail.com"
  smtp_port = 587                # 587 = STARTTLS, 465 = SSL
  user = "you@gmail.com"
  password = "app-password"      # Gmail: App Password (needs 2FA), not your real password
  ```

  Env vars override the file on clusters: `RUNQ_EMAIL_TO`, `RUNQ_SMTP_HOST`,
  `RUNQ_SMTP_PORT`, `RUNQ_SMTP_USER`, `RUNQ_SMTP_PASSWORD`, `RUNQ_EMAIL_FROM`,
  `RUNQ_NOTIFY_CONFIG`. A failing mail setup never breaks a run (warning only).

Naming convention: hyperparameters are spelled `HyperParams` in full, never abbreviated.
