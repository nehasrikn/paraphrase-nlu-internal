`bash post_to_turk.sh`

```
python -m mturk.defeasible.defeasible_hit_creator
python -m mturk.defeasible.post_process
python -m annotated_data.paraphrase_validation.validate_examples.py
```