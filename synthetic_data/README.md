# Synthetic claims documents

The documents LakeRCM processes in a demo come from here. The notebook
[`generate_documents.py`](generate_documents.py) writes realistic, entirely
fictional PDFs into the pipeline's input volume:

| Document | The classifier label it should get |
|---|---|
| Specialist referral | `referral_workqueue` |
| Prior authorization request | `prior_authorization` |
| Notice of claim denial | `denial_management` |
| Explanation of benefits | `explanation_of_benefits` |
| Laboratory report | `lab_results` |
| Clinical progress note | `clinical_notes` |

## How a document is made

1. **The facts come from code** ([`docgen.py`](docgen.py)).
   - People, clinics and addresses come from [Faker](https://faker.readthedocs.io/).
   - Provider NPIs deliberately fail the NPI check digit, so none can belong
     to a real provider.
   - Payers are the fictional ones in `scripts/payer_policy_content.py`:
     Veridane, Solvara, Kestrel, Oakhollow and Silver Harbor.
   - Diagnosis and procedure codes come from the reference sets the pipeline
     validates against.
   - Prior authorizations and denials are built around one of those payers'
     policies, and a denial gives a CARC reason that fits the policy and that
     the pipeline's denial coder maps back to its code.
2. **Some documents are hard on purpose.** About 30% carry one deliberate
   problem, so they land in the review queue instead of being auto-verified:
   - a non-billable diagnosis code;
   - an OCR-style malformed code;
   - a missing member ID.
3. **Claude Opus 5.5 writes the narrative.** `databricks-claude-opus-5-5` on the
   Foundation Model APIs writes the reason for referral, the clinical
   justification, the denial rationale and the rest, around the facts it is
   given. A reply is rejected and retried if it isn't well-formed JSON or if it
   names a real insurer.
4. **Ground truth is kept.** Each document gets a row in
   `<catalog>.<schema>.synthetic_document_manifest`: its expected label, payer,
   codes, identifiers and the problem it was given. You can score extraction
   and classification against it.

Every page carries a footer saying it is synthetic and that every name and
identifier is fictional.

## Run it

**As a job.** The jobs bundle deploys `lakercm-generate-synthetic-documents`
(add `_dev` on the dev target). Run it from the Jobs page, or:

```bash
databricks bundle run lakercm_generate_synthetic_documents -t dev -C bundles/jobs
```

It writes 20 documents, records them in the manifest, and starts a documents
pipeline update. Change `count` or `review_share` in the run's parameters.

**As a notebook.** Open `synthetic_data/generate_documents.py` in the workspace,
set the `catalog` and `schema` widgets, and run it.

The reviewer app lists a document once it has been through the pipeline, as
long as the Lakebase synced table `gold_extraction_labels_sync` exists
([docs/dev-data-plane.md](../docs/dev-data-plane.md)). Otherwise, download a
few PDFs from the volume and upload them on the **Reviewer** page.

## Dependencies

[`requirements.txt`](requirements.txt) pins every package the notebook
installs: databricks-sdk (Apache-2.0), Faker (MIT) and reportlab (BSD). The
model is set by the `synthetic_data_model` bundle variable (default
`databricks-claude-opus-5-5`). Any Foundation Model chat endpoint works; the
public-release policy lists Llama 4 (`databricks-llama-4-maverick`) among the
approved generators, so a release built on another model says so in its
acknowledgements.
