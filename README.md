# Lightweight model version control

Modern AI model development usually involves several teams, each in charge of a different part
of the work: supervised fine-tuning, preference optimisation, reinforcement learning, checkpoint
merging and evaluation. Keeping the metadata of such a project up to date is hard when it is
scattered across notebooks, chats and spreadsheets. This repository is shared by all of those
teams to organise and keep track of the model's training, while adding as little documentation
burden as possible to the development itself.

This is a demo: every experiment, person, email address, organisation and revision in it is
fictional, so the links do not resolve.

## How it works

Researchers keep training, evaluating and uploading models exactly as they do today. The only
extra step is a small YAML file per experiment, added through a pull request.

When a researcher starts an experiment, they copy
[`experiments/_template.yaml`](experiments/_template.yaml) and fill in who they are (name, email
and GitHub handle), the GitHub repository and commit of the code they are running, and the parent
checkpoints they start from, each pinned to its Hugging Face repository and revision. Opening the
pull request is enough for the experiment to appear in the lineage as running.

Once the run produces checkpoints, the same file gains an `outputs` entry for each of them: the
Hugging Face revision, the scores copied from the shared evaluation code, and the commit of that
evaluation code, so that results remain reproducible and comparable. Merging checkpoints needs
nothing special: a soup is simply an experiment with several weighted parents.

Not every idea works. A rejected experiment is not added to the registry; instead, its owner
files the *Rejected experiment* issue form, recording the parents, checkpoints, scores and the
reason, and closes it as not planned. The team keeps a searchable record of what has already been
tried without cluttering the lineage. Two examples are kept as
[issues](../../issues?q=is%3Aissue+label%3Arejected).

When a checkpoint becomes the team's reference model, someone changes the `main:` line in
[`pipeline.yaml`](pipeline.yaml) in a reviewed pull request. That file also lists the pipeline
phases, the benchmarks and the external base models.

Nobody draws the graph by hand. On every merge, a GitHub Actions workflow compiles the experiment
files into [`LINEAGE.md`](LINEAGE.md): a graph of how every checkpoint descends from its parents,
with each experiment's code repository and commit, followed by tables of experiments and of
checkpoints with their scores. Ancestry comes only from the declared parents and outputs, never
from the order of the phases.

## An experiment record

The soup below merges three checkpoints from two parallel experiments into one:

```yaml
id: exp-006
title: Weighted soup of three RLVR checkpoints
phase: souping
owner: {name: Researcher A, email: researcher-a@example.org, github: researcher-a}
status: accepted
method: weighted_average
code: {repo: example-org/merging, commit: 8f3a21c}
inputs:
  - {model: rlvr-a-v1, hf: {repo: example-org/rlvr-a, revision: 6a1c3e5}, weight: 0.2}
  - {model: rlvr-a-v2, hf: {repo: example-org/rlvr-a, revision: 8c2e4f6}, weight: 0.4}
  - {model: rlvr-b-v1, hf: {repo: example-org/rlvr-b, revision: 2f5b8d1}, weight: 0.4}
outputs:
  - id: soup-v1
    hf: {repo: example-org/soup-v1, revision: 4c1d9e2}
    eval: {commit: d41e7aa}
    scores: {math: 0.612, code: 0.548, chat: 7.6, safety: 0.946}
```

## What is checked

Before anything is merged, `scripts/registry.py` checks that each owner is complete, that the
code points to a GitHub repository (`owner/name`) and a commit, and that every checkpoint has a
pinned revision, an evaluation commit and a score for each benchmark. It also checks that every
parent exists and matches the registered revision, that soup weights sum to one, that the `main`
pointer is valid and that the lineage contains no cycles. A pull request that breaks any of these
fails, with a message naming the experiment at fault.

```bash
pip install -r requirements.txt
python scripts/registry.py          # validate and write LINEAGE.md
python scripts/registry.py --check  # validate and print, without writing
pytest
```

The workflow never downloads weights, runs benchmarks or calls external services. If `main` is a
protected branch, allow it to push the regenerated `LINEAGE.md`, or run the script before merging.

## Repository layout

```
pipeline.yaml          phases, benchmarks, base models, main pointer
experiments/           one YAML file per experiment
scripts/registry.py    validator and graph compiler
tests/                 validator tests
.github/               workflow, pull request template, rejected-experiment issue form
LINEAGE.md             generated output
```

## Citation

If you use or adapt this approach, please cite:

> Antonio Lobo-Santos. *Lightweight model version control*. GitHub repository, 2026.
> https://github.com/alobo01/lightweight-model-version-control

```bibtex
@misc{lobosantos2026lightweight,
  author       = {Lobo-Santos, Antonio},
  title        = {Lightweight model version control},
  year         = {2026},
  howpublished = {\url{https://github.com/alobo01/lightweight-model-version-control}}
}
```
