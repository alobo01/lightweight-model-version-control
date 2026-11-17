# GitHub CI/CD

The [registry workflow](../.github/workflows/registry.yml) validates experiment metadata and publishes
[`LINEAGE.md`](../LINEAGE.md). Its project settings come from [`config.yaml`](../config.yaml):
`project_name` titles the document and `main_branch` selects the Git branch used for publication.

1. **Pull requests:** Install the Python dependencies, run the tests and validate the active and
   rejected YAML records. Show the generated lineage in the workflow's GitHub Actions summary.
   Invalid metadata or failing tests fail the check.
2. **Pushes:** Run the same checks on every branch. On `main_branch`, regenerate `LINEAGE.md`
   and commit and push it if its contents changed. Other branches run checks without publishing.
3. **Rejected experiments:** Validate records in `rejected/` with `status: rejected` and link
   them in a separate section. Show their branches with red arrows in the overview and exclude
   their checkpoints from the active tables.

## What validation checks

The [registry validator](../scripts/registry.py) checks:

- **Project configuration:** Exactly `project_name` and `main_branch`, both nonempty strings;
  `main_branch` must be a valid Git branch name.
- **Experiment metadata:** Required `id`, `title`, `owner`, `status`, `code` and nonempty `inputs`;
  unique experiment and output checkpoint IDs; a nonempty string `method`. Owners need a name,
  email containing `@` and GitHub handle.
- **Statuses and tags:** Status is `in_progress`, `accepted` or `rejected`. Optional tags are lists of nonempty strings and require no registration.
- **Code and checkpoint references:** Code needs a repository in `owner/name` format and a commit.
  Inputs and outputs need Hugging Face repositories and revisions. Repeated references to a model
  must match its recorded repository and revision. Inputs with no producing experiment are external checkpoints.
- **Evaluation and scores:** Output checkpoints need an evaluation repository in `owner/name`
  format, an evaluation commit and a nonempty score dictionary. Benchmark names are nonempty
  strings and values are numeric, excluding booleans. External inputs can omit evaluation data;
  when supplied, it follows the same checks. Benchmark columns come from recorded scores, and
  checkpoints can have different benchmark coverage.
- **Parameters and run configuration:** Supplied input `parameters` must be dictionaries.
  An experiment's optional `config` contains `file`, `cli` or both as nonempty strings.
- **Rejected records:** Files in `rejected/` have `status: rejected`; rejected experiments belong
  in that folder. Active experiments cannot consume checkpoints produced by rejected experiments.
- **Lineage:** Experiment dependencies must contain no cycles.

`description` and `reject_reason` are optional metadata. Validation checks reference presence
and consistency without fetching remote repositories, checking hash existence or reproducing
scores. Method-specific rules, such as aggregation weights summing to one, belong to the
referenced experiment code.

## Publication permissions

The workflow requests `contents: write` to publish the document. Repository rules must allow the
Actions bot to push to `main_branch`. See [GitHub's token permissions documentation](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token).
Its commits use `GITHUB_TOKEN`, whose pushes do not trigger another workflow run, preventing a
publication loop. See [GitHub's workflow triggering rules](https://docs.github.com/en/actions/concepts/security/github_token).

Training, weight aggregation and evaluation run outside this workflow; it does not download models.

## Optional local checks

Run these commands from the repository root:

```bash
pip install -r requirements.txt
pytest -q
python scripts/registry.py --check  # validate and print without writing
python scripts/registry.py          # validate and regenerate LINEAGE.md
```
