# CHARMPheno — working notes for Claude

## Cluster commands must be self-contained (runnable on a fresh cluster)

Every command handed to the user to run **on the Dataproc cluster** (any `make
-C analysis/cloud …`, `spark-submit`, or other cluster-side command) MUST be
prefixed with a preamble that changes into the repo and puts the checkout on
the current development branch at the latest commit. A fresh cluster clones the
repo on `main`, so a bare `make …` runs stale code or fails with "No rule to
make target"; and a plain `git pull origin <branch>` from `main` fails on
divergent branches. The preamble below handles both the fresh-clone case (the
branch does not exist locally — `git checkout` creates a tracking branch) and
the already-checked-out case (fast-forward to origin):

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
```

Then the actual command on the next line(s). So a cluster command is always
delivered as, e.g.:

```bash
cd ~/repos/CHARMPheno && git fetch origin claude/gated-conditional-voi && git checkout claude/gated-conditional-voi && git pull --ff-only
make -C analysis/cloud diag-episode-probe ID=110 GPR_ARGS="--gap-days 90"
```

- The development branch is currently **`claude/gated-conditional-voi`**. When
  the active branch changes, update the name in the preamble above and in this
  note.
- `git pull --ff-only` is deliberate: it fast-forwards or refuses, never
  merges or destroys local state. The cluster is a pure runner with no local
  commits, so a refusal means something unexpected — surface it, do not paper
  over it with `reset --hard`.
- This applies ONLY to commands the user runs on the cluster. Commands I run
  here in the session's own working copy do not need it.
