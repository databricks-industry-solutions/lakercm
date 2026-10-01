"""LakeRCM knowledge-graph routes for the reviewer app.

Renders the one-hop neighbourhood of the open document: the codes it bills, the
policy that governs them, its payer, any denial reason -- and its patient, which
is the spoke that leads to the rest of that patient's documents.

Every spoke also carries the OTHER documents that share it (`linked`): the
patient's other documents, the other claims with this diagnosis, the other
denials for this reason. That is the question a node exists to answer, and
answering it per node is what makes a click on the graph go somewhere. Each
linked document comes back with its reviewer id, resolved here, so opening one
does not depend on it happening to sit in whatever page of the document list
the browser has cached -- the earlier name-based lookup missed every
auto-verified document, because the default list excludes them.

WHY THIS READS THE TRIPLESTORE DIRECTLY
    The same path the agent's traverse_claims_graph tool takes: a SELECT against
    <catalog>.<kg_schema>.triplestore_lakercm_v1 on the SQL warehouse. Not the
    OntoBricks app's HTTP API -- one code path for both consumers, and the
    reviewer SP already holds USE_SCHEMA + SELECT on the registry schema
    (bundles/ontobricks/resources/registry.yml).

TWO THINGS THAT LOOK RIGHT AND ARE NOT
    1. The join key is NOT document_name. medical_documents.document_name is the
       ORIGINAL upload filename ("invoice.pdf"); gold derives its document_name
       from the volume path, which the upload prefixed
       ("{email}_{ts}_invoice.pdf"). Joining those returns zero rows, silently.
       The reliable bridge is medical_documents.file_path, which equals
       gold_extraction_labels.document_path exactly (see routes/documents.py),
       and the graph's Document id is that path's BASENAME.
    2. Subjects and objects are FULL URIs
       (`https://lakercm.example/ontology/Document/<document_name>`), so
       `subject = '<document_name>'` matches nothing -- silently, returning an
       empty neighbourhood indistinguishable from a document the graph has not
       ingested. Anchor on the `/Document/<name>` suffix instead, and strip
       neighbours to their last URI segment for display.
    3. Predicate URIs must be split on BOTH separators. This store's grammar is
       inconsistent: predicates are `.../ontology/hasDiagnosis` (slash) while
       rdf:type objects are `...#Document` (hash). Splitting on '/' alone matched
       nothing and every edge came back empty -- the bug #88 fixed in the agent
       tool. SPLIT(..., '[/#]') is correct for either form.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from config import settings
from dependencies import get_lakercm_db, get_workspace_client
from services.warehouse import warehouse_rows

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kg")

# Edges whose far end is worth drawing as a spoke, and how to label them. An
# allowlist rather than "whatever the store holds", so a new internal predicate
# cannot silently appear in the UI.
EDGE_LABELS = {
    "documentsPatient": "patient",
    "billedTo": "payer",
    "hasDiagnosis": "diagnosis",
    "hasProcedure": "procedure",
    "deniedFor": "denial reason",
}

# A one-hop neighbourhood of a claim document is small (single-figure to low
# tens). The cap is a guard against a pathological node, not a paging boundary.
MAX_EDGES = 200

# Properties across a whole neighbourhood: a handful of entities with ~5 literals
# each. Generous, so a rich policy is never silently truncated mid-entity.
MAX_DETAIL_ROWS = 400

# Documents returned per spoke. A patient has a handful; a payer can have
# thousands, so this caps what is RETURNED and `total` still says how many exist.
MAX_LINKED_PER_NODE = 25


def _basename(path: str) -> str:
    return (path or "").rstrip("/").rsplit("/", 1)[-1]


def _edge_list_sql() -> str:
    # The allowlist as a SQL IN-list. Built from the module constant, never from
    # a request, so interpolating it is not an injection surface.
    return ", ".join(f"'{edge}'" for edge in EDGE_LABELS)


def _as_int(value) -> int:
    # The Statement Execution API returns every column as a string.
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _neighbourhood_sql(kg: str) -> str:
    """One-hop edges in both directions, anchored on the document's URI.

    SUBJECTS AND OBJECTS ARE FULL URIs, not bare ids:
    `https://lakercm.example/ontology/Document/<document_name>`. Comparing
    `subject = '<document_name>'` therefore matches NOTHING -- and it fails
    silently, returning an empty neighbourhood that looks exactly like a
    document the graph has not ingested yet. That is how the first cut of this
    route shipped: the endpoint answered 200 with `edges: []` for a document
    that has 15 triples.

    Matching on the `/Document/<name>` suffix rather than building the full URI
    keeps this independent of the base URI, which differs between the instance
    namespace (`.../ontology/`, slash) and BASE_URI in
    scripts/lakercm_domain_content.py (`...#`, hash).

    The neighbour value is stripped to its last URI segment for the same reason
    the UI needs it: `DiagnosisCode/M54.50` reads as `M54.50`.
    """
    return f"""
        SELECT ELEMENT_AT(SPLIT(predicate, '[/#]'), -1) AS edge,
               ELEMENT_AT(SPLIT(object, '/'), -1)       AS neighbour,
               'out'                                    AS direction
        FROM {kg}
        WHERE subject LIKE CONCAT('%/Document/', :doc)
        UNION ALL
        SELECT ELEMENT_AT(SPLIT(predicate, '[/#]'), -1) AS edge,
               ELEMENT_AT(SPLIT(subject, '/'), -1)      AS neighbour,
               'in'                                     AS direction
        FROM {kg}
        WHERE object LIKE CONCAT('%/Document/', :doc)
        LIMIT {MAX_EDGES}
    """


def _linked_sql(kg: str) -> str:
    """Every OTHER document that shares each spoke, by the same edge.

    Out from the document to each neighbour, then back in to the other
    documents that point at that neighbour. For the patient this is the
    question the graph exists to answer and no single table could: the rest of
    this patient's documents. The same two hops give "the other claims with
    this diagnosis" and "the other denials for this reason".

    THE PREDICATE IS PART OF THE JOIN, not just the object. A document's
    outgoing triples include its rdf:type, whose object (`...#Document`) every
    document shares -- joining on the object alone would link each document to
    the entire store. Matching the predicate too, restricted to the EDGE_LABELS
    allowlist, means "documents with hasDiagnosis M54.50", never "documents that
    are documents".

    Capped per spoke with a window rather than one LIMIT, so a payer with
    thousands of claims cannot crowd out the patient's handful; `total` is
    counted before the cap so the panel can say how many it did not list.
    """
    return f"""
        WITH nb AS (
            SELECT DISTINCT object AS uri,
                   ELEMENT_AT(SPLIT(predicate, '[/#]'), -1) AS edge
            FROM {kg}
            WHERE subject LIKE CONCAT('%/Document/', :doc)
              AND ELEMENT_AT(SPLIT(predicate, '[/#]'), -1) IN ({_edge_list_sql()})
        ),
        pairs AS (
            SELECT DISTINCT nb.edge, t.object AS uri, t.subject AS doc_uri
            FROM {kg} t
            JOIN nb
              ON t.object = nb.uri
             AND ELEMENT_AT(SPLIT(t.predicate, '[/#]'), -1) = nb.edge
            WHERE t.subject LIKE '%/Document/%'
              AND t.subject NOT LIKE CONCAT('%/Document/', :doc)
        ),
        ranked AS (
            SELECT edge,
                   ELEMENT_AT(SPLIT(uri, '/'), -1)     AS neighbour,
                   ELEMENT_AT(SPLIT(doc_uri, '/'), -1) AS document_name,
                   ROW_NUMBER() OVER (PARTITION BY edge, uri ORDER BY doc_uri) AS rn,
                   COUNT(*) OVER (PARTITION BY edge, uri)                      AS total
            FROM pairs
        )
        SELECT edge, neighbour, document_name, total
        FROM ranked
        WHERE rn <= {MAX_LINKED_PER_NODE}
    """


def _resolve_documents(db, names: list) -> dict:
    """Reviewer ids (and status/type) for documents the graph names by basename.

    Advisory, like the panel: a Lakebase hiccup must not cost the reviewer the
    graph, so a failure here leaves the documents listed but unopenable rather
    than failing the route.
    """
    if not names:
        return {}
    try:
        return db.get_documents_by_names(names) or {}
    except Exception as e:  # noqa: BLE001 - degrade, never break the panel
        logger.warning("KG linked-document lookup failed: %s", e)
        return {}


def _details_sql(kg: str) -> str:
    """Class, label and data properties for every neighbour, in one query.

    Fetched UP FRONT rather than per click. A one-hop neighbourhood is single-figure
    to low tens of entities, so this is one extra statement instead of a round trip
    (and a spinner) every time a reviewer opens a node.

    The `/` vs `#` split in this store's URI grammar does the filtering for us:
    instance links are `.../ontology/DiagnosisCode/M54.50` (slash) while rdf:type
    objects are `...#DiagnosisCode` (hash). So excluding `/ontology/` drops the
    object-property edges -- which are already returned as edges -- and keeps the
    literals plus the type.
    """
    return f"""
        WITH nb AS (
            SELECT object AS uri FROM {kg}
            WHERE subject LIKE CONCAT('%/Document/', :doc)
            UNION
            SELECT subject AS uri FROM {kg}
            WHERE object LIKE CONCAT('%/Document/', :doc)
        )
        SELECT ELEMENT_AT(SPLIT(t.subject, '/'), -1)        AS entity,
               ELEMENT_AT(SPLIT(t.predicate, '[/#]'), -1)   AS prop,
               t.object                                     AS value
        FROM {kg} t
        JOIN nb ON t.subject = nb.uri
        WHERE t.object NOT LIKE 'https://lakercm.example/ontology/%'
        LIMIT {MAX_DETAIL_ROWS}
    """


def _kg_table() -> Optional[str]:
    """Fully-qualified triplestore view, or None when the feature is off.

    Both the flag AND the schema are required: an enabled panel pointed at an
    empty schema would issue a query against a half-known name. #88's lesson is
    that this state has to be explicit rather than inferred from an empty result.
    """
    if not settings.kg_enabled:
        return None
    schema = (settings.kg_schema or "").strip()
    if not schema:
        logger.warning("kg_enabled is true but LAKERCM_KG_SCHEMA is empty")
        return None
    return f"`{settings.catalog}`.`{schema}`.triplestore_lakercm_v1"


@router.get("/status", response_model=dict)
async def kg_status():
    """Whether the panel can work at all, and why not when it cannot.

    The reviewer frontend asks this before rendering, so "no graph" is a stated
    condition instead of an empty panel the user has to interpret.
    """
    schema = (settings.kg_schema or "").strip()
    return {
        "enabled": bool(settings.kg_enabled),
        "schema": schema,
        "available": _kg_table() is not None,
        "reason": (
            "ok"
            if _kg_table()
            else (
                "LAKERCM_KG_ENABLED is false"
                if not settings.kg_enabled
                else "LAKERCM_KG_SCHEMA is empty"
            )
        ),
    }


@router.get("/document/{document_id}/neighbourhood", response_model=dict)
async def document_neighbourhood(
    request: Request,
    document_id: str,
    db=Depends(get_lakercm_db),
    workspace_client=Depends(get_workspace_client),
):
    """One-hop graph neighbourhood of a document, and who else shares each spoke."""
    kg = _kg_table()
    if kg is None:
        return {
            "available": False,
            "node": None,
            "node_type": None,
            "edges": [],
            "siblings": [],
            "details": {},
            "linked": {},
        }

    document = db.get_document_by_id(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # basename(file_path), NOT document_name -- see the module docstring.
    node = _basename(document.get("file_path") or "")
    # What the hub IS ("denial_management"), so the graph can name its centre
    # instead of drawing an anonymous dot. Absent until gold has the document.
    node_type = document.get("extracted_label")
    if not node:
        return {
            "available": True,
            "node": None,
            "node_type": node_type,
            "edges": [],
            "siblings": [],
            "details": {},
            "linked": {},
        }

    try:
        warehouse_id = settings.get_warehouse_id()
        rows = warehouse_rows(
            workspace_client, warehouse_id, _neighbourhood_sql(kg), {"doc": node}
        )
        edges = [
            {
                "edge": r["edge"],
                "label": EDGE_LABELS[r["edge"]],
                "neighbour": r["neighbour"],
                "direction": r["direction"],
            }
            for r in rows
            if r.get("edge") in EDGE_LABELS and r.get("neighbour")
        ]

        # Keyed `edge:neighbour`, the same key the panel gives a spoke, so two
        # entities that share a local name across classes cannot collide.
        linked: dict = {}
        for r in warehouse_rows(
            workspace_client, warehouse_id, _linked_sql(kg), {"doc": node}
        ):
            edge, neighbour, name = (
                r.get("edge"),
                r.get("neighbour"),
                r.get("document_name"),
            )
            if edge not in EDGE_LABELS or not neighbour or not name:
                continue
            entry = linked.setdefault(
                f"{edge}:{neighbour}", {"total": 0, "documents": []}
            )
            entry["documents"].append({"name": name})
            entry["total"] = max(
                entry["total"], _as_int(r.get("total")), len(entry["documents"])
            )

        resolved = _resolve_documents(
            db,
            sorted({d["name"] for e in linked.values() for d in e["documents"]}),
        )
        for entry in linked.values():
            entry["documents"].sort(key=lambda d: d["name"])
            for d in entry["documents"]:
                hit = resolved.get(d["name"]) or {}
                # id None = the graph knows the document but the review queue
                # does not yet (a streamed file before its first backfill). Still
                # listed -- it exists -- just not openable.
                d["id"] = hit.get("id")
                d["status"] = hit.get("status")
                d["label"] = hit.get("label")

        # Kept for callers of the earlier shape: the patient's documents.
        siblings = next(
            (
                [d["name"] for d in entry["documents"]]
                for key, entry in linked.items()
                if key.startswith("documentsPatient:")
            ),
            [],
        )

        # What each neighbour IS, so the panel can explain a node instead of just
        # naming it. Without this a spoke reads "diagnosis M54.50" and the reviewer
        # still has to know that M54.50 is low back pain, unspecified.
        details: dict = {}
        for r in warehouse_rows(
            workspace_client, warehouse_id, _details_sql(kg), {"doc": node}
        ):
            entity, prop, value = r.get("entity"), r.get("prop"), r.get("value")
            if not entity or not prop:
                continue
            d = details.setdefault(entity, {"cls": None, "label": None, "props": {}})
            if prop == "type":
                # `...#DiagnosisCode` -> `DiagnosisCode`
                d["cls"] = str(value).rsplit("#", 1)[-1]
            elif prop == "label":
                d["label"] = value
            else:
                d["props"][prop] = value

        return {
            "available": True,
            "node": node,
            "node_type": node_type,
            "edges": edges,
            "siblings": siblings,
            "details": details,
            "linked": linked,
        }
    except Exception as e:
        # Advisory panel: a graph that is mid-rebuild, or a warehouse that is
        # cold, must never break the review page.
        logger.warning("KG neighbourhood failed for %s: %s", document_id, e)
        return {
            "available": True,
            "node": node,
            "node_type": node_type,
            "edges": [],
            "siblings": [],
            "details": {},
            "linked": {},
            "error": str(e),
        }
