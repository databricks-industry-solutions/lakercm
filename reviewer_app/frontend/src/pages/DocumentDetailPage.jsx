import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router';
import {
  ExtractedDataPanel,
  ReviewPanel,
  buildAttribution,
  parseIdentifiers,
} from '../components/ExtractionComparison';
import BoundingBoxLayer from '../components/BoundingBoxLayer';
import ConnectionLine from '../components/ConnectionLine';
import ReviewerAssistantPane from '../components/ReviewerAssistant/ReviewerAssistantPane';
import AgentVerification from '../components/AgentVerification';
import ReviewSideTabs from '../components/ReviewSideTabs';
import ReviewNotes from '../components/ReviewNotes';
import KnowledgeGraphPanel from '../components/KnowledgeGraphPanel';
import { findFieldForNode } from '../components/kgFieldMatch';
import {
  getDocument,
  getExtractionComparisons,
  syncDocumentStatus,
  listDocuments,
} from '../api/lakeRcmApi';
import { getStatusLabel } from '../components/DocumentCard';
import './DocumentDetailPage.css';

// Module-level cache of recent listDocuments calls so the detail page can
// derive prev/next neighbors without re-fetching on every navigation. The
// cache survives across detail-page mounts within a session — entries are
// keyed by the params signature and evicted only when DocumentsPage refetches.
const _NEIGHBOR_CACHE = new Map();
const _NEIGHBOR_CACHE_TTL_MS = 60 * 1000;

async function fetchNeighborList(params) {
  const key = JSON.stringify(params);
  const cached = _NEIGHBOR_CACHE.get(key);
  if (cached && Date.now() - cached.t < _NEIGHBOR_CACHE_TTL_MS) {
    return cached.docs;
  }
  const data = await listDocuments(params);
  const docs = data?.documents || [];
  _NEIGHBOR_CACHE.set(key, { t: Date.now(), docs });
  return docs;
}

const LS_DATA_WIDTH = 'detail.dataPanelWidthPct';
const LS_REVIEW_HEIGHT = 'detail.reviewPanelHeight';
const LS_DATA_PANEL_COLLAPSED = 'detail.dataPanelCollapsed';
const LS_ASSISTANT_WIDTH = 'detail.assistantWidth';
const DEFAULT_DATA_WIDTH_PCT = 42;
const DEFAULT_REVIEW_HEIGHT = 260;
const DEFAULT_ASSISTANT_WIDTH = 420;
// 360px is the floor the composer, the mic button and the tier selector stay
// usable at. The ceiling is 720 because that is the viewport width at which the
// page stops reserving a gutter and the drawer starts overlaying — wider than
// that and the drawer would exceed the layout it is supposed to sit beside. 92vw
// preserves the cap the original `min(420px, 92vw)` had.
const ASSISTANT_MIN_WIDTH = 360;
const ASSISTANT_MAX_WIDTH = 720;
const ASSISTANT_OVERLAY_BREAKPOINT = 720;

export const clampAssistantWidth = (px, viewportWidth) => {
  const wanted = Number.isFinite(px) ? px : DEFAULT_ASSISTANT_WIDTH;
  // An unknown viewport means the 92vw cap cannot be computed, so only the
  // absolute ceiling applies. Defaulting the viewport to the ceiling instead
  // would apply 92% of it and quietly cap at 662px.
  const ceiling = Number.isFinite(viewportWidth)
    ? Math.min(ASSISTANT_MAX_WIDTH, Math.round(viewportWidth * 0.92))
    : ASSISTANT_MAX_WIDTH;
  // The viewport cap can fall BELOW the minimum on a narrow window, so the floor
  // is applied last and wins. Under 640px the drawer is 100vw from CSS anyway, so
  // an over-wide value there is inert rather than wrong.
  return Math.max(ASSISTANT_MIN_WIDTH, Math.min(ceiling, wanted));
};

const readStoredNumber = (key, fallback) => {
  if (typeof window === 'undefined') return fallback;
  const raw = window.localStorage.getItem(key);
  const n = raw == null ? NaN : Number(raw);
  return Number.isFinite(n) ? n : fallback;
};

// Close-edge hover debounce. Wraps a state setter so the "clear" call (next ===
// null) is deferred by `closeDelayMs`; any non-null call before the timer fires
// cancels the pending clear. Currently unused — kept available for any future
// hover-driven popover that needs flicker protection.
// eslint-disable-next-line no-unused-vars -- deliberately kept; see above.
function useDebouncedClose(setter, closeDelayMs = 180) {
  const timerRef = useRef(null);
  useEffect(
    () => () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    },
    []
  );
  return useCallback(
    (next) => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      if (next == null) {
        timerRef.current = setTimeout(() => {
          setter(null);
          timerRef.current = null;
        }, closeDelayMs);
      } else {
        setter(next);
      }
    },
    [setter, closeDelayMs]
  );
}

const DocumentDetailPage = ({ currentUser }) => {
  const { documentId } = useParams();
  const navigate = useNavigate();

  // In-document assistant drawer + the verdict it has proposed (approved cards
  // fill the review form via this; the reviewer still submits).
  const [showAssistant, setShowAssistant] = useState(false);
  const [proposedReview, setProposedReview] = useState(null);
  // An assistant-proposed note, relayed to ReviewNotes — the single owner of that
  // record. The nonce makes it idempotent across re-renders.
  const [agentNote, setAgentNote] = useState(null);
  // Whether the assistant is mid-turn on THIS document, and which tool is
  // running. Drives the "Agent verification" strip beside the review form.
  const [agentActivity, setAgentActivity] = useState({
    busy: false,
    tool: null,
    tier: null,
    tierSource: null,
  });

  // MERGES rather than replaces. The pane reports {busy, tool} on every tool
  // event and {tier, tierSource} only once, on the routing event -- so a plain
  // setState would wipe the tier the moment the next tool started, and the
  // reasoning strength would flash and vanish. `busy: false` still clears the
  // tier, because a tier from a finished turn says nothing about the next one.
  const handleAgentActivity = useCallback((next) => {
    setAgentActivity((prev) =>
      next?.busy === false
        ? { busy: false, tool: null, tier: null, tierSource: null }
        : { ...prev, ...next }
    );
  }, []);

  const [document, setDocument] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [comparisonData, setComparisonData] = useState(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonError, setComparisonError] = useState(null);

  const [zoom, setZoom] = useState(1);
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);
  const [showPanel, setShowPanel] = useState(() => {
    if (typeof window === 'undefined') return true;
    return window.localStorage.getItem(LS_DATA_PANEL_COLLAPSED) !== '1';
  });
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [imageError, setImageError] = useState(false);

  const [hoveredField, setHoveredField] = useState(null);
  const [hoveredBboxIdx, setHoveredBboxIdx] = useState(null);

  // Sticky lock state. When set, persists across hover noise: the locked row
  // expands inline with detail content, its bbox keeps its halo, and ↑/↓
  // navigates the panel. Cleared by clicking the same target again, by Esc,
  // or by clicking through to a different identifier.
  const [selectedField, setSelectedField] = useState(null);
  const [selectedBboxIdx, setSelectedBboxIdx] = useState(null);

  // The field a knowledge-graph node points at, while one is engaged. Separate
  // from selectedField on purpose: a graph node lights up its source, it does
  // not take over the reviewer's own pinned field or its connection line.
  const [kgField, setKgField] = useState(null);

  // Map<fieldKey, HTMLElement> populated by FieldRow refs. Used to scroll a
  // newly-selected row into view in the right-hand data panel without
  // blowing past it. The Map identity stays stable across renders.
  const fieldRefs = useRef(new Map());

  // Field-level corrections the reviewer typed inline. Keyed by `id:<idx>`
  // (stable across renders via parseIdentifiers). Sent alongside the review
  // verdict so curation downstream can train on the corrected values.
  const [corrections, setCorrections] = useState({});

  const handleFieldEdit = useCallback((fieldKey, newValue) => {
    setCorrections((prev) => {
      const next = { ...prev };
      if (newValue == null || newValue === '') {
        delete next[fieldKey];
      } else {
        next[fieldKey] = newValue;
      }
      return next;
    });
  }, []);

  const handleCorrectionsLoaded = useCallback((loaded) => {
    if (!loaded || typeof loaded !== 'object') return;
    // `prev` deliberately wins over `loaded`: the load is async, so a reviewer
    // who starts editing before it lands must not have that keystroke
    // overwritten. Safe only because the effect below empties `corrections`
    // on every document change -- without that reset this merge is what
    // carried one document's edits onto the next.
    setCorrections((prev) => ({ ...loaded, ...prev }));
  }, []);

  // Corrections are per-document, and this component is NOT remounted between
  // documents: the route is `/review/documents/:documentId` with no `key`
  // (App.jsx), `documentId` comes from useParams(), and j/k navigation calls
  // navigate() to the same route -- so React keeps the instance and only the
  // param changes. Without this reset, `corrections` survived the transition
  // and handleCorrectionsLoaded's `prev`-wins merge then let the PREVIOUS
  // document's field edits both persist and override the new document's real
  // saved corrections -- which handleSubmitReview would then write against the
  // wrong document. Runs synchronously on the documentId change, before the
  // async review load resolves.
  useEffect(() => {
    setCorrections({});
    setKgField(null);
    // A turn that was streaming about the PREVIOUS document must not leave this
    // strip spinning over the next one: the pane aborts on unmount, but j/k
    // changes only the route param, so nothing else clears it.
    setAgentActivity({ busy: false, tool: null, tier: null, tierSource: null });
  }, [documentId]);

  // Queue navigation: prev/next IDs derived from a cached listDocuments call.
  // We pull a wide page (limit=200) so most realistic queues are covered in a
  // single fetch. If the user's current doc isn't in the page, neighbors are
  // null and j/k become no-ops (graceful degradation).
  const [neighborList, setNeighborList] = useState([]);
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchNeighborList({ limit: 200, offset: 0 })
      .then((docs) => {
        if (!cancelled) setNeighborList(docs);
      })
      .catch(() => {
        // Non-fatal: the queue strip will simply be hidden.
        if (!cancelled) setNeighborList([]);
      });
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const queuePosition = useMemo(() => {
    const idx = neighborList.findIndex((d) => d.id === documentId);
    if (idx === -1) return null;
    return {
      idx,
      total: neighborList.length,
      prev: idx > 0 ? neighborList[idx - 1].id : null,
      next: idx < neighborList.length - 1 ? neighborList[idx + 1].id : null,
    };
  }, [neighborList, documentId]);

  // The graph hands back a document NAME (the basename of its volume path); the
  // route needs an id. `documents` is already loaded for the j/k queue, so this
  // resolves locally rather than adding a lookup endpoint.
  // Opens a document the knowledge graph links to. The graph route resolves each
  // linked document's reviewer id server-side, so this navigates by id. The
  // name-only lookup below is a fallback for an id-less caller, and it is NOT
  // enough on its own: neighborList is the cached j/k page, which excludes
  // auto-verified documents -- that is how six of seven patient links on a
  // family document silently did nothing.
  const handleOpenSiblingDocument = useCallback(
    (doc) => {
      const id = typeof doc === 'object' ? doc?.id : null;
      if (id) {
        navigate(`/review/documents/${id}`);
        return;
      }
      const documentName = typeof doc === 'object' ? doc?.name : doc;
      if (!documentName) return;
      const match = (neighborList || []).find(
        (d) => (d.file_path || '').endsWith(`/${documentName}`)
      );
      if (match?.id) navigate(`/review/documents/${match.id}`);
    },
    [navigate, neighborList]
  );

  const goToNeighbor = useCallback(
    (id) => {
      if (!id) return;
      navigate(`/review/documents/${id}`);
    },
    [navigate]
  );

  const [dataPanelWidthPct, setDataPanelWidthPct] = useState(() =>
    readStoredNumber(LS_DATA_WIDTH, DEFAULT_DATA_WIDTH_PCT)
  );
  const [reviewPanelHeight, setReviewPanelHeight] = useState(() =>
    readStoredNumber(LS_REVIEW_HEIGHT, DEFAULT_REVIEW_HEIGHT)
  );
  // Clamped on READ too: a width stored on a wide monitor must not survive
  // verbatim onto a laptop.
  const [assistantWidth, setAssistantWidth] = useState(() =>
    clampAssistantWidth(
      readStoredNumber(LS_ASSISTANT_WIDTH, DEFAULT_ASSISTANT_WIDTH),
      typeof window === 'undefined' ? undefined : window.innerWidth
    )
  );
  const [isResizingAssistant, setIsResizingAssistant] = useState(false);

  const imageRef = useRef(null);
  const contentRef = useRef(null);
  const topRowRef = useRef(null);
  // Scrollable container around the document image — drives "scroll to source"
  // when a field is locked or the user clicks a bbox below the fold.
  const imageAreaRef = useRef(null);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LS_DATA_WIDTH, String(dataPanelWidthPct));
    }
  }, [dataPanelWidthPct]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LS_ASSISTANT_WIDTH, String(assistantWidth));
    }
  }, [assistantWidth]);

  // Re-clamp when the window changes: 700px is legal on a desktop and wider than
  // 92vw on a tablet, and a drawer wider than its viewport has no handle to grab.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const onResize = () =>
      setAssistantWidth((w) => clampAssistantWidth(w, window.innerWidth));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LS_REVIEW_HEIGHT, String(reviewPanelHeight));
    }
  }, [reviewPanelHeight]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.localStorage.setItem(LS_DATA_PANEL_COLLAPSED, showPanel ? '0' : '1');
    }
  }, [showPanel]);

  useEffect(() => {
    if (!documentId) return;
    let cancelled = false;

    const loadOnce = async () => {
      try {
        const data = await getDocument(documentId);
        return data?.document || data || null;
      } catch (err) {
        console.error('Failed to load document:', err);
        throw err;
      }
    };

    const run = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await loadOnce();
        if (cancelled) return;
        setDocument(data);
        // If this doc is still 'pending', trigger a sync and refresh once —
        // covers the case where the user lands here directly after upload
        // without ever loading the list page.
        if (data?.processing_status === 'pending') {
          try {
            await syncDocumentStatus();
            const refreshed = await loadOnce();
            if (!cancelled) setDocument(refreshed);
          } catch (err) {
            console.warn('post-load sync failed (non-fatal):', err);
          }
        }
      } catch (err) {
        if (!cancelled) setError(err.message || 'Failed to load document');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [documentId]);

  const loadComparisons = async () => {
    try {
      setComparisonLoading(true);
      setComparisonError(null);
      const result = await getExtractionComparisons(documentId);
      setComparisonData(result);
    } catch (err) {
      setComparisonError(err.message || 'Failed to load comparison data');
    } finally {
      setComparisonLoading(false);
    }
  };

  useEffect(() => {
    if (!documentId) return;
    loadComparisons();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documentId]);

  const record = comparisonData?.items?.[0] || null;
  const allElements = useMemo(() => {
    const raw = record?.elements;
    if (!Array.isArray(raw)) return [];
    return raw;
  }, [record]);

  // Only render bboxes for page 1 until we wire multi-page navigation.
  const page1Elements = useMemo(
    () =>
      allElements.filter(
        (el) => el?.page_number == null || el.page_number === 1
      ),
    [allElements]
  );

  const attribution = useMemo(
    () => buildAttribution(record?.identifiers || [], allElements),
    [record, allElements]
  );

  const boxes = useMemo(
    () =>
      page1Elements
        .map((el, i) => {
          const bb = el?.bounding_box;
          if (!bb) return null;
          return {
            x: bb.x,
            y: bb.y,
            width: bb.width,
            height: bb.height,
            type: el.element_type,
            text: el.text_content,
            confidence: el.confidence_score,
            originalIdx: allElements.indexOf(el),
            _i: i,
          };
        })
        .filter(Boolean),
    [page1Elements, allElements]
  );

  // Pointer intent first, then the graph (the more recent intent when a node is
  // engaged), then the reviewer's pinned field.
  const highlightIndices = useMemo(() => {
    if (hoveredField != null) {
      return attribution.forward.get(hoveredField) ?? new Set();
    }
    if (kgField != null) {
      return attribution.forward.get(kgField) ?? new Set();
    }
    if (selectedField != null) {
      return attribution.forward.get(selectedField) ?? new Set();
    }
    return null;
  }, [hoveredField, kgField, selectedField, attribution]);

  // ── Parsed identifier indexes for selection / inspector / arrow-key nav ──

  const parsedFields = useMemo(
    () => parseIdentifiers(record?.identifiers || []),
    [record]
  );

  const parsedFieldsByKey = useMemo(() => {
    const map = new Map();
    parsedFields.topLevel.forEach((f) => map.set(f.key, f));
    parsedFields.sectionHeaders.forEach((f) => map.set(f.key, f));
    parsedFields.tableHeaders.forEach((f) => map.set(f.key, f));
    parsedFields.rows.forEach((r) => r.fields.forEach((f) => map.set(f.key, f)));
    return map;
  }, [parsedFields]);

  // Bridge for the in-document assistant's approved proposals. Edits reuse the
  // same corrections path as inline editing; the verdict fills the review form
  // (never auto-submits). Both ensure the data/review panel is visible.
  const fieldLabelForKey = useCallback(
    (key) => parsedFieldsByKey.get(key)?.name || key,
    [parsedFieldsByKey]
  );

  const handleAssistantApplyCorrection = useCallback(
    (correctionKey, value) => {
      handleFieldEdit(correctionKey, value);
      setShowPanel(true);
    },
    [handleFieldEdit]
  );

  const handleAssistantApplyVerdict = useCallback(({ verdict, reasoning }) => {
    setProposedReview({ verdict, reasoning, ts: Date.now() });
    setShowPanel(true);
  }, []);

  // The assistant proposed a note. ReviewNotes owns that record, so the text is
  // handed to it rather than written here — and the panel is forced open for the
  // same reason the two handlers above do it: ReviewNotes is unmounted while the
  // panel is collapsed, so a note handed to a hidden component would be dropped.
  const handleAgentNote = useCallback((text) => {
    if (!text) return;
    setAgentNote({ id: Date.now(), text });
    setShowPanel(true);
  }, []);

  // Visual order for ↑/↓ navigation: top-level → section headers → table
  // headers → rows in numeric order. Matches the rendering order of
  // ExtractedDataPanel so arrow keys advance the way the eye moves.
  const orderedFieldKeys = useMemo(() => {
    const keys = [];
    parsedFields.topLevel.forEach((f) => keys.push(f.key));
    parsedFields.sectionHeaders.forEach((f) => keys.push(f.key));
    parsedFields.tableHeaders.forEach((f) => keys.push(f.key));
    parsedFields.rows.forEach((r) => r.fields.forEach((f) => keys.push(f.key)));
    return keys;
  }, [parsedFields]);

  // originalIdx → box lookup so the inline detail's "Scroll to source" button
  // can resolve the selected element idx without re-walking `boxes`.
  const boxByOriginalIdx = useMemo(() => {
    const map = new Map();
    boxes.forEach((b) => map.set(b.originalIdx, b));
    return map;
  }, [boxes]);

  // First citation bbox for the locked field — drives the inline detail's
  // "Scroll to source" action and the panel-side click-to-lock scroll-sync.
  // Returns null when the selected field has no rendered citation on this page.
  const selectedFieldFirstBox = useMemo(() => {
    if (selectedField == null) return null;
    const eis = attribution.forward.get(selectedField);
    if (!eis || eis.size === 0) return null;
    for (const ei of eis) {
      const b = boxByOriginalIdx.get(ei);
      if (b) return b;
    }
    return null;
  }, [selectedField, attribution, boxByOriginalIdx]);

  // ── Click-to-lock handlers ───────────────────────────────────────────────

  // Lock a field. Toggles off if already locked. Mirrors the selection to
  // the bbox side via attribution so the document and the panel stay in sync.
  const handleFieldClick = useCallback(
    (key) => {
      if (key == null) return;
      setSelectedField((prev) => {
        if (prev === key) {
          setSelectedBboxIdx(null);
          return null;
        }
        const eis = attribution.forward.get(key);
        if (eis && eis.size > 0) {
          // Take the first attributed element as the visual anchor.
          for (const ei of eis) {
            setSelectedBboxIdx(ei);
            break;
          }
        } else {
          setSelectedBboxIdx(null);
        }
        return key;
      });
    },
    [attribution]
  );

  // Lock a bbox. `localIdx` is the index into the local `boxes` array (what
  // BoundingBoxLayer hands back); we translate to the originalIdx the
  // attribution map is keyed by.
  const handleBboxClick = useCallback(
    (localIdx) => {
      if (localIdx == null) {
        setSelectedBboxIdx(null);
        setSelectedField(null);
        return;
      }
      const box = boxes[localIdx];
      if (!box) return;
      const originalIdx = box.originalIdx;
      // VS Code-style "reopen on intent": clicking a bbox while the right
      // panel is collapsed restores the panel so the reviewer can see the
      // newly-locked row's inline detail. The scroll-into-view in the
      // selected-field useEffect handles centering once the panel is back.
      setShowPanel((prev) => (prev ? prev : true));
      setSelectedBboxIdx((prev) => {
        if (prev === originalIdx) {
          setSelectedField(null);
          return null;
        }
        const fks = attribution.inverse.get(originalIdx);
        if (fks && fks.size > 0) {
          for (const fk of fks) {
            setSelectedField(fk);
            break;
          }
        } else {
          setSelectedField(null);
        }
        return originalIdx;
      });
    },
    [boxes, attribution]
  );

  const clearSelection = useCallback(() => {
    setSelectedField(null);
    setSelectedBboxIdx(null);
  }, []);

  // Smooth-scroll a bbox into the center of the document viewport when the
  // user locks it via the panel side. Skipped when the box would already be
  // visible to avoid jumpy corrections on small docs.
  const scrollBboxIntoView = useCallback((box) => {
    const area = imageAreaRef.current;
    const img = imageRef.current;
    if (!area || !img || !box) return;
    const imgRect = img.getBoundingClientRect();
    const areaRect = area.getBoundingClientRect();
    if (imgRect.width === 0 || imgRect.height === 0) return;
    // Image natural pixels → displayed pixels.
    const sy = imgRect.height / Math.max(1, imageDimensions.height || 1);
    // Bbox top in image-natural pixels → top relative to area-scroll-content.
    const bboxTopInArea =
      imgRect.top - areaRect.top + box.y * sy + area.scrollTop;
    const targetScrollTop =
      bboxTopInArea - areaRect.height / 2 + (box.height * sy) / 2;
    const currentVisibleTop = bboxTopInArea - area.scrollTop;
    const margin = areaRect.height * 0.15;
    if (
      currentVisibleTop > margin &&
      currentVisibleTop < areaRect.height - margin
    ) {
      // Already comfortably in view — no-op.
      return;
    }
    area.scrollTo({
      top: Math.max(0, targetScrollTop),
      behavior: 'smooth',
    });
  }, [imageDimensions]);

  // The knowledge graph's half of the field<->bbox highlight: an engaged graph
  // node lights up the field its value was extracted from, exactly as hovering
  // that field does. Pinning a node (`scroll`) also brings the box into view;
  // hover alone never scrolls, or the document would jump as the pointer
  // crossed the ring. Returns the field's name only when a box for it is
  // actually drawn on this page, so the panel's "Highlighted on the page" line
  // states a fact rather than a match that lit nothing.
  const handleKgHighlight = useCallback(
    (node, { scroll = false } = {}) => {
      const key = node ? findFieldForNode(parsedFields, node) : null;
      setKgField(key);
      if (!key) return null;
      let firstBox = null;
      for (const ei of attribution.forward.get(key) || []) {
        firstBox = boxByOriginalIdx.get(ei) || null;
        if (firstBox) break;
      }
      if (!firstBox) return null;
      if (scroll) scrollBboxIntoView(firstBox);
      return fieldLabelForKey(key);
    },
    [parsedFields, attribution, boxByOriginalIdx, scrollBboxIntoView, fieldLabelForKey]
  );

  // Sync scroll on the panel side whenever the locked field changes.
  useEffect(() => {
    if (selectedField == null) return;
    const el = fieldRefs.current.get(selectedField);
    if (el && typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
    if (selectedFieldFirstBox) {
      scrollBboxIntoView(selectedFieldFirstBox);
    }
  }, [selectedField, selectedFieldFirstBox, scrollBboxIntoView]);

  // Sync scroll on the bbox side whenever the locked bbox changes from the
  // bbox-click direction (i.e. selectedField hasn't taken responsibility).
  useEffect(() => {
    if (selectedBboxIdx == null) return;
    const box = boxByOriginalIdx.get(selectedBboxIdx);
    if (box) scrollBboxIntoView(box);
    // When locked via bbox click, also pull the corresponding field row
    // (set by handleBboxClick) into view if attribution mapped one.
    if (selectedField != null) {
      const el = fieldRefs.current.get(selectedField);
      if (el && typeof el.scrollIntoView === 'function') {
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }
  }, [selectedBboxIdx, selectedField, boxByOriginalIdx, scrollBboxIntoView]);

  // ── Keyboard model ───────────────────────────────────────────────────────

  useEffect(() => {
    const handleKeyPress = (e) => {
      // Skip when the user is typing into an editable element (textarea,
      // input, or contentEditable field-value) so hotkeys don't clobber edits.
      const target = e.target;
      const tag = target?.tagName;
      const isTyping =
        tag === 'TEXTAREA' ||
        tag === 'INPUT' ||
        tag === 'SELECT' ||
        target?.isContentEditable;

      if (e.key === '?' && !isTyping) {
        e.preventDefault();
        setShowHelp((s) => !s);
        return;
      }
      if (e.key === 'Escape') {
        if (showHelp) {
          setShowHelp(false);
          return;
        }
        // Modal-style Esc: clear an active lock first, only fall back to
        // back-navigation on a second press.
        if (selectedField != null || selectedBboxIdx != null) {
          clearSelection();
          return;
        }
        navigate('/review');
        return;
      }

      // Cmd/Ctrl+Enter submits the review even from inside the reasoning
      // textarea — power-user pattern. Plain Enter only submits when focus
      // is outside an editable element (handled below).
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        const btn = window.document.querySelector('.review-submit-btn');
        if (btn && !btn.disabled) {
          e.preventDefault();
          btn.click();
        }
        return;
      }

      if (isTyping) return;

      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        if (orderedFieldKeys.length === 0) return;
        e.preventDefault();
        const dir = e.key === 'ArrowDown' ? 1 : -1;
        const currentIdx =
          selectedField != null
            ? orderedFieldKeys.indexOf(selectedField)
            : -1;
        let nextIdx;
        if (currentIdx === -1) {
          nextIdx = dir > 0 ? 0 : orderedFieldKeys.length - 1;
        } else {
          nextIdx = Math.min(
            orderedFieldKeys.length - 1,
            Math.max(0, currentIdx + dir)
          );
        }
        const nextKey = orderedFieldKeys[nextIdx];
        if (!nextKey || nextKey === selectedField) return;
        setSelectedField(nextKey);
        const eis = attribution.forward.get(nextKey);
        if (eis && eis.size > 0) {
          for (const ei of eis) {
            setSelectedBboxIdx(ei);
            break;
          }
        } else {
          setSelectedBboxIdx(null);
        }
        return;
      }
      if (e.key === '+' || e.key === '=') {
        setZoom((prev) => Math.min(prev + 0.25, 3));
      } else if (e.key === '-' || e.key === '_') {
        setZoom((prev) => Math.max(prev - 0.25, 0.5));
      } else if (e.key === '0') {
        setZoom(1);
      } else if (e.key === '1' || e.key === '2' || e.key === '3') {
        const value =
          e.key === '1'
            ? 'correct'
            : e.key === '2'
              ? 'partially_correct'
              : 'incorrect';
        // The styled <label> is what we'd ideally click, but the radio inside
        // is `display: none` (decorative), so label-click delegation skips
        // the input's change handler. Click the input directly so React's
        // onChange fires.
        // NOTE: `document` is a useState variable in this component
        // (`const [document, setDocument] = useState(null)`), so it shadows
        // the global. Use `window.document` to reach the DOM API.
        window.document
          .querySelector(`input[type="radio"][name="verdict"][value="${value}"]`)
          ?.click();
      } else if (e.key === 'Enter') {
        const btn = window.document.querySelector('.review-submit-btn');
        if (btn && !btn.disabled) {
          e.preventDefault();
          btn.click();
        }
      } else if (e.key === 'j' || e.key === ']') {
        if (queuePosition?.next) goToNeighbor(queuePosition.next);
      } else if (e.key === 'k' || e.key === '[') {
        if (queuePosition?.prev) goToNeighbor(queuePosition.prev);
      } else if (e.key === '\\') {
        // Toggle the right Extracted Data panel — sibling to the j/k queue
        // nav. Matches Figma/VS Code's Cmd+\ sidebar convention without the
        // modifier, since other shortcuts in this view are unmodified.
        e.preventDefault();
        setShowPanel((s) => !s);
      } else if (e.key.toLowerCase() === 'e') {
        window.document.querySelector('.review-reasoning')?.focus();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [
    navigate,
    queuePosition,
    goToNeighbor,
    showHelp,
    selectedField,
    selectedBboxIdx,
    clearSelection,
    orderedFieldKeys,
    attribution,
  ]);

  // Same shape as handleVerticalDragStart below: window listeners, body cursor +
  // userSelect, clamp on every move. The drawer grows leftward, so the delta sign
  // matches the vertical handle's.
  const handleAssistantResizeStart = (e) => {
    e.preventDefault();
    if (typeof window === 'undefined') return;
    // Below the breakpoint the drawer overlays instead of reflowing the page, so
    // there is no gutter to trade against and resizing means nothing. The CSS
    // hides the handle too; this covers a window shrunk mid-session.
    if (window.innerWidth <= ASSISTANT_OVERLAY_BREAKPOINT) return;
    const startX = e.clientX;
    const startWidth = assistantWidth;
    setIsResizingAssistant(true);

    const onMouseMove = (ev) => {
      setAssistantWidth(
        clampAssistantWidth(startWidth + (startX - ev.clientX), window.innerWidth)
      );
    };
    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.document.body.style.cursor = '';
      window.document.body.style.userSelect = '';
      setIsResizingAssistant(false);
    };

    window.document.body.style.cursor = 'col-resize';
    window.document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  // The existing resize handles are mouse-only. This one is keyboard-operable
  // because a pane you cannot size without a pointer is a pane some reviewers
  // cannot size at all.
  const handleAssistantResizeKey = (e) => {
    const vw = typeof window === 'undefined' ? undefined : window.innerWidth;
    const step = e.shiftKey ? 96 : 24;
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setAssistantWidth((w) => clampAssistantWidth(w + step, vw));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setAssistantWidth((w) => clampAssistantWidth(w - step, vw));
    } else if (e.key === 'Home') {
      e.preventDefault();
      setAssistantWidth(clampAssistantWidth(ASSISTANT_MAX_WIDTH, vw));
    } else if (e.key === 'End') {
      e.preventDefault();
      setAssistantWidth(clampAssistantWidth(ASSISTANT_MIN_WIDTH, vw));
    } else if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setAssistantWidth(clampAssistantWidth(DEFAULT_ASSISTANT_WIDTH, vw));
    }
  };

  const handleVerticalDragStart = (e) => {
    e.preventDefault();
    const row = topRowRef.current;
    if (!row) return;
    const rect = row.getBoundingClientRect();
    const startX = e.clientX;
    const startPct = dataPanelWidthPct;

    const onMouseMove = (ev) => {
      const deltaPx = startX - ev.clientX;
      const deltaPct = (deltaPx / rect.width) * 100;
      const next = Math.min(70, Math.max(20, startPct + deltaPct));
      setDataPanelWidthPct(next);
    };
    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.document.body.style.cursor = '';
      window.document.body.style.userSelect = '';
    };

    window.document.body.style.cursor = 'col-resize';
    window.document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  const handleHorizontalDragStart = (e) => {
    e.preventDefault();
    const contentEl = contentRef.current;
    if (!contentEl) return;
    const contentHeight = contentEl.getBoundingClientRect().height;
    const startY = e.clientY;
    const startHeight = reviewPanelHeight;
    const minH = 140;
    const maxH = Math.max(minH, contentHeight - 200);

    const onMouseMove = (ev) => {
      const delta = startY - ev.clientY;
      const next = Math.min(maxH, Math.max(minH, startHeight + delta));
      setReviewPanelHeight(next);
    };
    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.document.body.style.cursor = '';
      window.document.body.style.userSelect = '';
    };

    window.document.body.style.cursor = 'row-resize';
    window.document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  if (loading) {
    return (
      <div className="detail-page">
        <div className="detail-loading">
          <div className="detail-loading-spinner" />
          <p>Loading document...</p>
        </div>
      </div>
    );
  }

  if (error || !document) {
    return (
      <div className="detail-page">
        <div className="detail-error">
          <div className="detail-error-icon">!</div>
          <h2>Unable to load document</h2>
          <p>{error || 'Document not found'}</p>
          <button className="btn-back-error" onClick={() => navigate('/review')}>
            ← Back to Documents
          </button>
        </div>
      </div>
    );
  }


  const imageAreaFlexBasis = showPanel
    ? `${100 - dataPanelWidthPct}%`
    : '100%';
  const dataPanelFlexBasis = `${dataPanelWidthPct}%`;

  return (
    <div
      className={`detail-page${
        showAssistant ? ' detail-page--assistant-open' : ''
      }${isResizingAssistant ? ' detail-page--resizing' : ''}`}
      /* One number moves both the drawer and the page's right gutter: the CSS
         rule for --assistant-width is the DEFAULT, and both consumers already
         read the property. --resizing suppresses the 180ms padding transition,
         which would otherwise make the gutter lag the cursor on every move. */
      style={{ '--assistant-width': `${assistantWidth}px` }}
    >
      {/* Header */}
      <div className="detail-header">
        <div className="detail-header-left">
          <button
            className="btn-back"
            onClick={() => navigate('/review')}
            title="Back to documents (Esc)"
          >
            ← Back
          </button>
          <h1 className="detail-title">{document.document_name}</h1>
          {document.document_type && (
            <span className="detail-type-badge">{document.document_type}</span>
          )}
        </div>

        <div className="detail-header-right">
          {queuePosition && (
            <div className="detail-queue-strip" aria-label="Queue navigation">
              <button
                className="btn-queue-nav"
                onClick={() => goToNeighbor(queuePosition.prev)}
                disabled={!queuePosition.prev}
                title="Previous document (k)"
                aria-label="Previous document"
              >
                ←
              </button>
              <span className="detail-queue-pos">
                <strong>{queuePosition.idx + 1}</strong>
                <span className="detail-queue-sep">/</span>
                {queuePosition.total}
              </span>
              <button
                className="btn-queue-nav"
                onClick={() => goToNeighbor(queuePosition.next)}
                disabled={!queuePosition.next}
                title="Next document (j)"
                aria-label="Next document"
              >
                →
              </button>
            </div>
          )}

          <button
            className="btn-header-action"
            onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
            title={showBoundingBoxes ? 'Hide annotations' : 'Show annotations'}
          >
            Annotations
          </button>

          <button
            className="btn-header-action"
            onClick={() => setShowPanel(!showPanel)}
            title={showPanel ? 'Hide extraction panel' : 'Show extraction panel'}
          >
            Extractions
          </button>

          <button
            className={`btn-header-action ${showAssistant ? 'is-active' : ''}`}
            onClick={() => setShowAssistant((s) => !s)}
            title={showAssistant ? 'Hide assistant' : 'Open in-document assistant'}
            aria-pressed={showAssistant}
          >
            ✦ Assistant
          </button>

          <button
            className="btn-header-action"
            onClick={() => setShowHelp((s) => !s)}
            title="Keyboard shortcuts (?)"
            aria-label="Keyboard shortcuts"
          >
            ?
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div
        ref={contentRef}
        className={`detail-content ${showPanel ? 'with-panel' : 'full-width'}`}
      >
        {/* Top row: image | extracted data */}
        <div ref={topRowRef} className="detail-top-row">
          <div
            ref={imageAreaRef}
            className="detail-image-area"
            style={{ flexBasis: imageAreaFlexBasis }}
          >
            <div
              className="detail-image-wrapper"
              style={{
                transform: `scale(${zoom})`,
                transformOrigin: 'center top',
              }}
            >
              <div className="detail-image-container">
                {imageError ? (
                  <div className="detail-image-error">
                    <div className="detail-image-error-icon">—</div>
                    <p className="detail-image-error-title">
                      Unable to load document image
                    </p>
                    <p className="detail-image-error-subtitle">
                      The file may have been removed from the UC Volume or is
                      not accessible.
                    </p>
                    <button
                      className="btn-retry-image"
                      onClick={() => setImageError(false)}
                    >
                      Retry
                    </button>
                  </div>
                ) : (
                  <>
                    <img
                      ref={imageRef}
                      src={`/api/documents/${documentId}/image`}
                      alt={document.document_name}
                      className="detail-image"
                      onLoad={(e) => {
                        setImageDimensions({
                          width: e.target.naturalWidth,
                          height: e.target.naturalHeight,
                        });
                      }}
                      onError={() => {
                        console.error(
                          '[DocumentDetailPage] Failed to load image for document:',
                          documentId
                        );
                        setImageError(true);
                      }}
                    />
                    {showBoundingBoxes && boxes.length > 0 && (
                      <BoundingBoxLayer
                        boxes={boxes}
                        imageWidth={imageDimensions.width}
                        imageHeight={imageDimensions.height}
                        highlightIndices={highlightIndices}
                        hoveredBboxIdx={hoveredBboxIdx}
                        selectedBboxIdx={selectedBboxIdx}
                        onBboxHover={(idx) => {
                          if (idx == null) {
                            setHoveredBboxIdx(null);
                          } else {
                            const box = boxes[idx];
                            setHoveredBboxIdx(box ? box.originalIdx : null);
                          }
                        }}
                        onBboxClick={handleBboxClick}
                      />
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {!showPanel && (
            <button
              type="button"
              className="detail-panel-restore"
              onClick={() => setShowPanel(true)}
              title="Show extracted data panel (\\)"
              aria-label="Show extracted data panel"
            >
              ‹
            </button>
          )}

          {showPanel && (
            <>
              <div
                className="detail-vertical-resizer"
                onMouseDown={handleVerticalDragStart}
                title="Drag to resize"
              >
                <div className="detail-vertical-resizer-grip" />
                <button
                  type="button"
                  className="detail-vertical-resizer-toggle"
                  onMouseDown={(e) => e.stopPropagation()}
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowPanel(false);
                  }}
                  title="Hide extracted data panel (\\)"
                  aria-label="Hide extracted data panel"
                >
                  ›
                </button>
              </div>

              <div
                className="detail-data-panel"
                style={{ flexBasis: dataPanelFlexBasis }}
              >
                <ExtractedDataPanel
                  data={comparisonData}
                  loading={comparisonLoading}
                  error={comparisonError}
                  onRetry={loadComparisons}
                  attribution={attribution}
                  hoveredField={hoveredField ?? kgField}
                  hoveredBboxIdx={hoveredBboxIdx}
                  selectedField={selectedField}
                  selectedBboxIdx={selectedBboxIdx}
                  onFieldHover={setHoveredField}
                  onFieldClick={handleFieldClick}
                  onUnlock={clearSelection}
                  onScrollToSource={
                    selectedFieldFirstBox
                      ? () => scrollBboxIntoView(selectedFieldFirstBox)
                      : undefined
                  }
                  corrections={corrections}
                  onFieldEdit={handleFieldEdit}
                  fieldRefs={fieldRefs}
                />
              </div>
            </>
          )}
        </div>

        {/* Horizontal resizer between top row and review panel */}
        {showPanel && (
          <div
            className="detail-resize-handle"
            onMouseDown={handleHorizontalDragStart}
          >
            <div className="detail-resize-grip" />
          </div>
        )}

        {/* Review panel — pinned to bottom, split in half: the verdict on the
            left, the reviewer's notes on the right. They are written together —
            the note is the justification for the verdict — so putting the notes
            beside it rather than below (or inside the assistant drawer, where
            they used to live) means neither has to be scrolled away to reach the
            other. */}
        {showPanel && (
          <div
            className="detail-review-panel"
            style={{ height: reviewPanelHeight }}
          >
            <div className="detail-review-split">
              <div className="detail-review-left">
                <AgentVerification
                  busy={agentActivity.busy}
                  tool={agentActivity.tool}
                  tier={agentActivity.tier}
                  tierSource={agentActivity.tierSource}
                />
                <ReviewPanel
                  documentId={documentId}
                  hasRecord={Boolean(record)}
                  corrections={corrections}
                  onCorrectionsLoaded={handleCorrectionsLoaded}
                  proposedReview={proposedReview}
                />
              </div>
              <div className="detail-review-right">
                {/* Tabs, not a stack. Both were stacked here below the review
                    form, which put the graph permanently below the fold on a
                    laptop. They are alternatives, not a sequence: a reviewer is
                    either writing up what they checked or looking at what the
                    document connects to. The section is resizable because a
                    radial diagram and a markdown notepad want very different
                    amounts of room. */}
                <ReviewSideTabs
                  tabs={[
                    {
                      id: 'notes',
                      label: 'Notes',
                      node: (
                        <ReviewNotes
                          documentId={documentId}
                          appendRequest={agentNote}
                        />
                      ),
                    },
                    {
                      id: 'graph',
                      label: 'Knowledge graph',
                      node: (
                        <KnowledgeGraphPanel
                          documentId={documentId}
                          onOpenDocument={handleOpenSiblingDocument}
                          onHighlightNode={handleKgHighlight}
                        />
                      ),
                    },
                  ]}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* In-document assistant drawer. Same LakeRCM assistant, scoped to
          this document: proposes edits/verdict as approve-first cards and keeps
          a private notepad. */}
      {showAssistant && (
        <ReviewerAssistantPane
          documentId={documentId}
          currentUser={currentUser}
          onClose={() => setShowAssistant(false)}
          onApplyCorrection={handleAssistantApplyCorrection}
          onApplyVerdict={handleAssistantApplyVerdict}
          fieldLabelForKey={fieldLabelForKey}
          onNoteAppended={handleAgentNote}
          assistantWidth={assistantWidth}
          minWidth={ASSISTANT_MIN_WIDTH}
          maxWidth={ASSISTANT_MAX_WIDTH}
          onResizeStart={handleAssistantResizeStart}
          onResizeKey={handleAssistantResizeKey}
          onActivityChange={handleAgentActivity}
        />
      )}

      {/* Connection line — soft Bézier from the locked field row to its
          first citation bbox. Page-level so it spans both panes' coordinate
          spaces. The component itself is responsible for hiding gracefully
          when either endpoint is offscreen, zero-size, or unresolvable. */}
      <ConnectionLine
        visible={
          showPanel &&
          selectedField != null &&
          selectedFieldFirstBox != null
        }
        fieldRefs={fieldRefs}
        fromKey={selectedField}
        imageEl={imageRef.current}
        imageWidth={imageDimensions.width}
        imageHeight={imageDimensions.height}
        box={selectedFieldFirstBox}
      />

      {/* Hotkey help overlay */}
      {showHelp && (
        <div className="detail-help-overlay" onClick={() => setShowHelp(false)}>
          <div
            className="detail-help-card"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-label="Keyboard shortcuts"
          >
            <div className="detail-help-header">
              <h3 className="detail-help-title">Keyboard shortcuts</h3>
              <button
                className="detail-help-close"
                onClick={() => setShowHelp(false)}
                aria-label="Close shortcuts"
              >
                ×
              </button>
            </div>
            <ul className="detail-help-list">
              <li><kbd>1</kbd> Mark Correct</li>
              <li><kbd>2</kbd> Mark Partially Correct</li>
              <li><kbd>3</kbd> Mark Incorrect</li>
              <li><kbd>Enter</kbd> / <kbd>⌘↵</kbd> Submit review</li>
              <li><kbd>↑</kbd> / <kbd>↓</kbd> Navigate identifiers</li>
              <li><kbd>\</kbd> Toggle extracted data panel</li>
              <li><kbd>Esc</kbd> Clear selection / back to list</li>
              <li><kbd>j</kbd> / <kbd>]</kbd> Next document</li>
              <li><kbd>k</kbd> / <kbd>[</kbd> Previous document</li>
              <li><kbd>e</kbd> Focus reasoning</li>
              <li><kbd>+</kbd> / <kbd>−</kbd> Zoom</li>
              <li><kbd>0</kbd> Reset zoom</li>
              <li><kbd>?</kbd> Toggle this help</li>
            </ul>
            <p className="detail-help-foot">
              Click an identifier or a bounding box to lock the selection — the
              two stay in sync. Click an extracted value to correct it.
            </p>
          </div>
        </div>
      )}

      {/* Footer Controls */}
      <div className="detail-footer">
        <div className="detail-footer-section">
          {(() => {
            const isReviewed = Boolean(document.review_verdict);
            const displayStatus = isReviewed
              ? 'reviewed'
              : (document.processing_status || 'processing');
            return (
              <span
                className="detail-status-badge"
                data-status={displayStatus}
              >
                {getStatusLabel(displayStatus)}
              </span>
            );
          })()}
        </div>

        <div className="detail-footer-section">
          <button
            className="btn-footer-control"
            onClick={() => setZoom((prev) => Math.max(prev - 0.25, 0.5))}
            disabled={zoom <= 0.5}
            title="Zoom out (-)"
          >
            −
          </button>
          <span
            className="detail-zoom-indicator"
            onClick={() => setZoom(1)}
            title="Reset zoom (0)"
          >
            {Math.round(zoom * 100)}%
          </span>
          <button
            className="btn-footer-control"
            onClick={() => setZoom((prev) => Math.min(prev + 0.25, 3))}
            disabled={zoom >= 3}
            title="Zoom in (+)"
          >
            +
          </button>
        </div>

        <div className="detail-footer-section detail-keyboard-hints">
          <span className="detail-hint">1 2 3 Verdict</span>
          <span className="detail-hint">↑↓ Navigate</span>
          <span className="detail-hint">↵ Submit</span>
          <span className="detail-hint">j k Next/Prev</span>
          <span className="detail-hint">? Help</span>
        </div>
      </div>
    </div>
  );
};

export default DocumentDetailPage;
