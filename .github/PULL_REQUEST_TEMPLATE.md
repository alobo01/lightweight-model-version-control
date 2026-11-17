**Experiment:** exp-NNN
**Change:** new experiment / checkpoint added / status change

- [ ] `owner` has name, email and GitHub handle
- [ ] Every parent in `inputs` has its Hugging Face `repo` and `revision` (and a `parameters` dictionary when needed for aggregation)
- [ ] Hugging Face `revision` values are commit hashes, not branch names
- [ ] `code` names the GitHub repository (`owner/name`) and the commit that produced the checkpoint
- [ ] Configuration needed to reproduce the run is recorded in `config.file`, `config.cli`, or both
- [ ] Scores copied from the evaluation output, with the evaluation-code `eval.repo` and `eval.commit`
- [ ] Rejected experiments are in `rejected/` with `status: rejected` and a `reject_reason`
- [ ] `LINEAGE.md` not edited by hand
