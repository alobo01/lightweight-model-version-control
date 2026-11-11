# Lightweight model version control

A minimal, GitHub-based registry for tracking how model checkpoints descend from one another
when several researchers work on the same training pipeline. This repository is a demo: every
experiment, person, email address, organisation and revision in it is fictional, so the links
do not resolve.

**Author:** Antonio Lobo-Santos

## Idea

Researchers train, evaluate and upload models exactly as they do today. The only extra step is
one small YAML file per experiment. Everything else is derived.

| Concern | Lives in |
| --- | --- |
| Experiment metadata, owner (name, email, GitHub handle), status | This repository (`experiments/*.yaml`) |
| Checkpoints, parents and children | Hugging Face, pinned to an immutable revision |
| Training code | Its own repository, pinned to a commit |
| Scores | Copied into each checkpoint record, with the commit of the common evaluation code that produced them |
| Rejected experiments | Closed GitHub issues labelled `rejected` |
| Lineage graph and tables | [`LINEAGE.md`](LINEAGE.md), generated automatically |

## Result

See [`LINEAGE.md`](LINEAGE.md) for the compiled graph and the experiment and checkpoint tables
(with scores). The demo covers a seven-phase pipeline with parallel experiments from the same
parent, an experiment producing two checkpoints, a soup merging three checkpoints into one, and a
run still in progress. Two rejected experiments are kept as
[issues](../../issues?q=is%3Aissue+label%3Arejected).

## Workflow

1. **Start.** Copy [`experiments/_template.yaml`](experiments/_template.yaml), fill in the owner
   and the pinned parents, set `status: running` and open a pull request.
2. **Register a checkpoint.** Add an entry under `outputs` with its Hugging Face revision, the
   scores and the evaluation-code commit. Set `status` to `accepted` or `inconclusive`.
3. **Reject.** File the *Rejected experiment* issue form and close it as not planned. Nothing is
   added to `experiments/`.
4. **Promote.** Change `main:` in [`pipeline.yaml`](pipeline.yaml) in a reviewed pull request.

## Experiment record

```yaml
id: exp-006
title: Weighted soup of three RLVR checkpoints
phase: souping                      # a phase id from pipeline.yaml
owner: {name: Researcher A, email: researcher-a@example.org, github: researcher-a}
status: accepted
method: weighted_average
code: {repo: example-org/merging, commit: 8f3a21c}
inputs:                             # parents, pinned exactly as registered
  - {model: rlvr-a-v1, hf: {repo: example-org/rlvr-a, revision: 6a1c3e5}, weight: 0.2}
  - {model: rlvr-a-v2, hf: {repo: example-org/rlvr-a, revision: 8c2e4f6}, weight: 0.4}
  - {model: rlvr-b-v1, hf: {repo: example-org/rlvr-b, revision: 2f5b8d1}, weight: 0.4}
outputs:
  - id: soup-v1
    hf: {repo: example-org/soup-v1, revision: 4c1d9e2}
    eval: {commit: d41e7aa}         # evaluation-code commit that produced the scores
    scores: {math: 0.612, code: 0.548, chat: 7.6, safety: 0.946}
```

A soup is an ordinary experiment with several weighted `inputs`. `pipeline.yaml` holds the phase
list, the benchmark names, the external base models and the `main` pointer. Phases only group and
order the display; ancestry comes solely from `inputs` and `outputs`.

## Validation

`scripts/registry.py` rejects:

- unknown phases or statuses, and duplicate identifiers;
- owners without a name, email or GitHub handle;
- checkpoints without a pinned revision, evaluation-code commit or a score for every benchmark;
- parents that are not registered, or whose repository or revision differs from the registered
  checkpoint;
- soups with fewer than two inputs or weights that do not sum to 1;
- a `main` pointer to an unregistered model, and cycles.

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
pipeline.yaml          phases, benchmarks, base models, main pointer
experiments/           one YAML file per experiment
scripts/registry.py    validator and graph compiler
tests/                 validator tests
.github/               workflow, pull request template, rejected-experiment issue form
LINEAGE.md             generated output
```
