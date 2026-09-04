# Security policy

## Supported version

Security fixes are applied to the current `0.1.x` development line.

## Reporting a vulnerability

Use GitHub private vulnerability reporting when it is enabled for the repository.
If that feature is unavailable, contact the maintainers through a private
institutional channel. Do not open a public issue containing an exploit, credential,
private dataset location, unpublished result, or personal information.

Include the affected version, relevant file or component, reproduction conditions,
expected impact, and any proposed mitigation. The maintainers will confirm receipt
before public disclosure is discussed.

## Credential handling

The software reads provider credentials from process environment variables. Secrets
must not be stored in source files, notebooks, configuration files, command history,
logs, checkpoints, or release archives. Revoke any credential immediately if it is
committed or otherwise disclosed.
