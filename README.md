# Lightweight model version control

A minimal, GitHub-based registry for tracking how model checkpoints descend from one another
when several researchers work on the same training pipeline. This repository is a demo: every
experiment, owner, organisation and revision in it is fictional, so the links do not resolve.

**Author:** Antonio Lobo-Santos

## Idea

Researchers train, evaluate and upload models exactly as they do today. The only extra step is
one small YAML file per experiment. Everything else is derived.

| Concern | Lives in |
| --- | --- |
| Experiment metadata, ownership, decisions | This repository (`experiments/*.yaml`) |
| Checkpoints | Hugging Face, pinned to an immutable revision |
| Training code | Its own repository, pinned to a commit |
| Evaluation results | The common evaluation repository; records here only point to a run |
| Lineage graph and tables | [`LINEAGE.md`](LINEAGE.md), generated automatically |

Nothing is recorded twice: scores are not copied here, and the graph is never drawn by hand.

## Result

See [`LINEAGE.md`](LINEAGE.md) for the compiled graph, the experiment and checkpoint tables, and
the discarded-work log. The demo covers a seven-phase pipeline with parallel experiments from the
same parent, an experiment producing two checkpoints, a weighted merge, a discarded branch and a
run still in progress.

## Workflow

1. **Start.** Copy [`experiments/_template.yaml`](experiments/_template.yaml), set `status: running`
   and open a pull request. The experiment appears in the graph before any checkpoint exists.
2. **Register a checkpoint.** Add an entry under `outputs` with the Hugging Face revision and the
   evaluation run. Set `status` to `accepted`, `discarded` (with a `reason`) or `inconclusive`.
3. **Promote.** Change `main:` in [`pipeline.yaml`](pipeline.yaml) in a reviewed pull request.

A merge is an ordinary experiment with several `inputs`. Discarded experiments are kept, so the
team can see what has already been tried.

## Experiment record

```yaml
id: exp-006
title: Weighted soup of two RLVR checkpoints
phase: souping                      # a phase id from pipeline.yaml
owner: researcher-a                 # GitHub handle
status: accepted
method: weighted_average
code: {repo: example-org/merging, commit: 8f3a21c}
inputs:
  - {model: rlvr-a-v2, weight: 0.6}
  - {model: rlvr-b-v1, weight: 0.4}
outputs:
  - id: soup-v1
    hf: {repo: example-org/soup-v1, revision: 4c1d9e2}
    eval: runs/soup-v1              # path in the evaluation repository
notes: Better aggregate results than either input.
```

`pipeline.yaml` holds the phase list, the external base models and the `main` pointer. Phases
only group and order the display; ancestry comes solely from `inputs` and `outputs`.

## Validation

`scripts/registry.py` rejects unknown phases or statuses, duplicate identifiers, inputs that do
not resolve, outputs without a pinned revision, discarded experiments without a reason, a `main`
pointer to an unregistered model, and cycles.

```bash
pip install -r requirements.txt
python scripts/registry.py          # validate and write LINEAGE.md
python scripts/registry.py --check  # validate and print, without writing
pytest
```

## Automation

A single workflow, [`.github/workflows/registry.yml`](.github/workflows/registry.yml):

- **Pull request:** validates the registry and shows the resulting graph in the job summary.
- **Push to `main`:** regenerates and commits `LINEAGE.md`.

It never downloads weights, runs benchmarks or calls external services. If `main` is a protected
branch, allow the workflow to push to it or run the script locally before merging.

## Layout

```
pipeline.yaml          phases, base models, main pointer
experiments/           one YAML file per experiment
scripts/registry.py    validator and graph compiler
tests/                 validator tests
LINEAGE.md             generated output
```
