# Publishing SemCred-FL on GitHub and Zenodo

This procedure publishes the source repository and archives a tagged software
release in Zenodo. It deliberately excludes datasets, prepared artifacts,
checkpoints, experiment outputs, baseline PDFs, and manuscript sources.

The GitHub–Zenodo procedure below was checked against the official GitHub and Zenodo
documentation on 2026-09-04.

## 1. Review the release identity

The initial software release is `0.1.0`. Its creators are recorded in this order:

1. Nahideh Derakhshanfard
2. Ata Mohammadi

Both creators use the affiliation:

> Department of Computer Engineering, Ta.C., Islamic Azad University, Tabriz, Iran

Confirm the spelling, order, affiliation, and version before publishing. Add ORCID
identifiers to `CITATION.cff` and `.zenodo.json` only when the identifiers have been
verified by their owners.

The repository currently uses an all-rights-reserved source-available notice. This
does not grant permission to reuse or modify the code. If the copyright holders want
an open-source release, they must jointly select and approve a software license
before the first public tag. Do not describe the project as MIT, Apache, GPL, or
another open-source license unless the corresponding license text replaces the
current `LICENSE` and `pyproject.toml` is updated consistently.

## 2. Run the local release gates

Open PowerShell and run:

```powershell
Set-Location -LiteralPath "F:\University\کارشناسی ارشد\Article\Paper 2\SemCred-FL\Code"
conda run --no-capture-output -n semcred-fl ruff check src scripts tests
conda run --no-capture-output -n semcred-fl ruff format --check src scripts tests
conda run --no-capture-output -n semcred-fl mypy --strict src
conda run --no-capture-output -n semcred-fl python -m pytest -q
conda run --no-capture-output -n semcred-fl python scripts\audit_public_release.py
```

Do not continue unless every command exits successfully and the final command prints:

```text
status=ready
datasets_included=false
generated_artifacts_included=false
manuscript_included=false
high_confidence_secrets_detected=false
```

The audit uses Git's own candidate list. It therefore checks the files that would be
added, not an independently maintained copy list.

## 3. Inspect the Git candidate set

Initialize the local repository if it has not already been initialized:

```powershell
git init -b main
git config user.name "Ata Mohammadi"
$gitEmail = Read-Host "Enter the email address to associate with Git commits"
git config user.email $gitEmail
```

Inspect ignored files before staging:

```powershell
git status --short --ignored
git check-ignore -v "Datasets" "artifacts" "Baselines PDF" "paper" "tmp"
```

Each of those five paths must be reported as ignored. Stage the release candidate and
inspect it again:

```powershell
git add .
git status --short
git diff --cached --stat
git diff --cached -- .gitignore CITATION.cff .zenodo.json pyproject.toml README.md
```

Before committing, verify that the staged list contains source, scripts, tests,
configs, notebooks, and documentation, but no provider data, generated artifacts,
manuscript files, credentials, PDFs, archives, databases, Parquet files, or model
checkpoints.

Create the first local commit:

```powershell
git commit -m "Prepare SemCred-FL research software v0.1.0"
```

## 4. Create the GitHub repository

### GitHub website method

1. Sign in to GitHub and open <https://github.com/new>.
2. Set the repository name to `SemCred-FL`.
3. Add the description: `Research implementation of LLM-guided dual-view credibility aggregation for federated intrusion detection.`
4. Select **Public** when the source is ready for public access.
5. Do not initialize the remote repository with a README, `.gitignore`, or license;
   those files already exist locally.
6. Create the repository and copy its HTTPS URL.
7. Return to PowerShell and run:

```powershell
$remoteUrl = Read-Host "Paste the empty GitHub repository HTTPS URL"
git remote add origin $remoteUrl
git remote -v
git push -u origin main
```

GitHub documents this empty-remote procedure in
[Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

### GitHub CLI method

If GitHub CLI is installed, the equivalent authenticated workflow is:

```powershell
gh auth login
gh repo create SemCred-FL --public --source . --remote origin --push
```

Use only one creation method. Do not run `git remote add origin` when the GitHub CLI
has already created that remote.

## 5. Check the GitHub landing page

Verify the following before creating a release:

- The README renders correctly.
- The **Cite this repository** control is visible. GitHub creates it from the root
  `CITATION.cff` file.
- `Datasets/`, `artifacts/`, `Baselines PDF/`, `paper/`, and `tmp/` are absent.
- The Actions tab contains a successful CI run.
- The repository contains `.zenodo.json`, `CITATION.cff`, `LICENSE`, and
  `CHANGELOG.md`.

In **Settings → Security**, enable Dependabot alerts, secret scanning, push
protection, and private vulnerability reporting when those controls are available.
GitHub recommends these controls for public repositories. The repository already
contains a security policy and monthly Dependabot configuration.

GitHub warns above 50 MiB and blocks individual repository files above 100 MiB. The
local release audit uses 50 MiB as the fail-closed threshold. See
[About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

## 6. Connect GitHub to Zenodo

1. Sign in to <https://zenodo.org/>.
2. Open the profile menu and select **Linked accounts**.
3. Connect the GitHub account if it is not connected already.
4. Open the Zenodo profile menu and select **GitHub**.
5. Select **Sync now**.
6. Find `SemCred-FL` and enable its repository toggle.
7. Refresh the page and confirm that the repository remains enabled.

Zenodo states that enabled repositories automatically ingest new GitHub releases.
See [Enable a repository](https://help.zenodo.org/docs/github/enable-repository/).

The repository intentionally contains both metadata formats. GitHub reads
`CITATION.cff`; Zenodo reads `.zenodo.json` when both files are present. Zenodo's
[software metadata documentation](https://help.zenodo.org/docs/github/describe-software/)
describes this precedence rule.

## 7. Create the DOI-bearing release

Run the release audit once more on the exact commit:

```powershell
git status --short
conda run --no-capture-output -n semcred-fl python scripts\audit_public_release.py
git tag -a v0.1.0 -m "SemCred-FL research software v0.1.0"
git push origin v0.1.0
```

`git status --short` must print nothing before tagging.

On GitHub:

1. Open **Releases** and select **Draft a new release**.
2. Choose the existing `v0.1.0` tag.
3. Set the title to `SemCred-FL v0.1.0`.
4. Use the `0.1.0` section of `CHANGELOG.md` as the release notes.
5. State that datasets, generated artifacts, and manuscript sources are not included.
6. Publish the release.

GitHub's current release procedure is documented in
[Managing releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).

Do not create a second manual Zenodo upload for the same software version. Wait for
the enabled GitHub integration to process the release. In Zenodo's GitHub page,
select the repository and inspect the release status. When processing completes,
open the record DOI and verify the title, version, creators, affiliation, resource
type, files, and access setting. Zenodo documents this process in
[Archive a release from GitHub](https://help.zenodo.org/docs/github/archive-software/github-upload/).

## 8. Record the assigned DOI

After Zenodo assigns the software DOI:

1. Copy the DOI directly from the Zenodo record.
2. Add the DOI to the root `CITATION.cff` as the top-level `doi` field.
3. Add the GitHub repository URL as the `repository-code` field in `CITATION.cff`.
4. Add a DOI badge and DOI-linked citation to the README.
5. Commit and push those metadata-only changes to `main`.
6. Do not reuse the journal article DOI as the software DOI. They identify different
   research objects and should be linked as related outputs in Zenodo when the
   article DOI exists.

Zenodo automatically registers a DOI when a record is published. A DOI may also be
reserved in a manual draft, but that is a different deposit route and should not be
combined with an automatic GitHub deposit for the same release. See Zenodo's
[DOI guidance](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/).

## 9. Publish later versions

For each software update:

1. Update the version in `pyproject.toml`, `src/semcred_fl/__init__.py`,
   `CITATION.cff`, and `.zenodo.json`.
2. Add a dated section to `CHANGELOG.md`.
3. Run every local release gate.
4. Commit and push the change.
5. Create and push a matching annotated tag.
6. Publish a GitHub release from that tag.
7. Verify the corresponding Zenodo version record.

Zenodo treats a file-changing version as a new immutable record linked to the other
versions. Its [version-management guide](https://help.zenodo.org/docs/deposit/manage-versions/)
explains that model.
