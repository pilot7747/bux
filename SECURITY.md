# Security policy

## Reporting a vulnerability

Report a vulnerability privately, through GitHub's
[private vulnerability reporting](https://github.com/pilot7747/bux/security/advisories/new),
not as a public issue. Include what an attacker can do, the steps or files that
reproduce it, and the commit you found it on.

bux has no releases yet: fixes land on `main`.

## What to trust

bux trains models from files you give it, so treat these as code from whoever wrote
them:

- **Exported models.** `bux export` writes the model's Python next to its weights, and
  transformers runs it when it loads the directory with `trust_remote_code=True`.
  Load exports only from people you trust.
- **`.bux` files, datasets and checkpoints.** They decide what runs, what it reads and
  where it writes.

`bux studio` edits files on disk. It listens on 127.0.0.1 by default, refuses a
request whose host is not the machine's, and asks every call to its API for the token
in the URL it prints. Reach it from another machine through an SSH tunnel rather than
`--host 0.0.0.0`.
