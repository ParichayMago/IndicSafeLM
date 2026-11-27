# Sanitized test fixtures

Only small, sanitized fixtures used by unit and integration tests belong here. Do not commit raw offensive examples, personal data, or copyrighted corpus extracts.

`phase2_input.jsonl` is synthetic text created solely to exercise normalization,
script checks, PII rejection, exact/near-duplicate removal, split freezing, and
balanced-view generation. The email address and phone-like value are reserved
test placeholders, not personal data. It is not a corpus sample.
