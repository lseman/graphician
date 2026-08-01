// ═══════════════════════════════════════════════════════════════
// ARIADNE GRAPH EXPLORER — sigma.js / WebGL renderer
// ═══════════════════════════════════════════════════════════════

(function () {
  'use strict';

  // ── State ────────────────────────────────────────────────────
  let graphData = null;
  let graph = null; // graphology.Graph instance
  let sigma = null;
  let layoutFrame = null;
  let selectedNodeId = null;
  let hoveredNodeId = null;
  let searchResults = [];
  let selectedIndex = -1;
  let currentDepthFilter = 0; // 0 = all
  let reachableSet = null; // Set<string> when depth filter active, else null
  let followLayoutCamera = true;

  // ── Filter state ─────────────────────────────────────────────
  let visibleNodeTypes = new Set();
  let visibleEdgeTypes = new Set(['Calls', 'Contains', 'Defines', 'Imports', 'Extends', 'Implements', 'MemberOf', '']);
  let activeCommunities = new Set();
  let communityColorFilterOn = false;

  // ── Node type config ─────────────────────────────────────────
  const NODE_TYPE_COLORS = {
    File: '#3b82f6', Folder: '#6366f1', Function: '#059669', Method: '#0d9488',
    Class: '#d97706', Interface: '#db2777', Variable: '#64748b', Import: '#475569',
    Type: '#a78bfa', Module: '#0891b2', Community: '#818cf8',
  };
  const NODE_TYPE_SIZES = {
    Folder: 9, File: 5, Class: 7, Interface: 6, Type: 4,
    Function: 3, Method: 2, Variable: 2, Import: 1.5, Module: 8, Community: 0,
  };
  const EDGE_TYPE_COLORS = {
    Calls: '#7c3aed', Contains: '#059669', Defines: '#0891b2',
    Imports: '#2563eb', Extends: '#c2410c', Implements: '#be185d',
    MemberOf: '#64748b', Reference: '#475569',
  };
  const COMMUNITY_PALETTE = [
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
    '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f',
  ];
  visibleNodeTypes = new Set(Object.keys(NODE_TYPE_SIZES));

  // ── DOM ──────────────────────────────────────────────────────
  const searchInput = document.getElementById('searchInput');
  const searchResultsEl = document.getElementById('searchResults');
  const graphCanvas = document.getElementById('graphCanvas');
  const hoverTooltip = document.getElementById('hoverTooltip');
  const selectionBar = document.getElementById('selectionBar');
  const selectionName = document.getElementById('selectionName');
  const selectionType = document.getElementById('selectionType');
  const legend = document.getElementById('legend');
  const rightPanel = document.getElementById('rightPanel');
  const nodeDetails = document.getElementById('nodeDetails');
  const fileTree = document.getElementById('fileTree');
  const nodeCountEl = document.getElementById('nodeCount');
  const edgeCountEl = document.getElementById('edgeCount');
  const fileSearch = document.getElementById('fileSearch');

  // ── Init ─────────────────────────────────────────────────────
  async function init() {
    try {
      await loadGraph();
      buildGraphology();
      setupEventListeners();
      setupKeyboardShortcuts();
      renderSigma();
      zoomToFit(false);
      runLayout(true);
      renderLegend();
      renderFileTypeFilters();
      renderEdgeTypeFilters();
      renderFileTree();
      updateStats();
    } catch (error) {
      console.error('Failed to initialize:', error);
      graphCanvas.innerHTML = `<div class="loading-state">Failed to load graph.<br><span style="font-size:11px;color:#5a5a70">${escapeHtml(error.message)}</span></div>`;
    }
  }

  // ── Load data ────────────────────────────────────────────────
  async function loadGraph() {
    const response = await fetch('/api/graph?limit=5000&edge_limit=10000');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    graphData = await response.json();
  }

  // The Rust API serializes enum variants as snake_case. The UI configuration
  // uses compact PascalCase names, so normalize at the network boundary and
  // keep the rest of the rendering state consistent.
  function normalizeKind(value) {
    return String(value || 'unknown')
      .split('_')
      .filter(Boolean)
      .map(part => part.charAt(0).toUpperCase() + part.slice(1))
      .join('');
  }

  // ── Build graphology graph ───────────────────────────────────
  function buildGraphology() {
    graph = new graphology.MultiDirectedGraph();
    const w = graphCanvas.clientWidth || 800;
    const h = graphCanvas.clientHeight || 600;

    graphData.nodes.forEach(n => {
      const id = String(n.id);
      const kind = normalizeKind(n.kind);
      visibleNodeTypes.add(kind);
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.sqrt(Math.random()) * Math.min(w, h) * 0.4;
      graph.addNode(id, {
        label: n.qname || n.label,
        displayLabel: n.label,
        qname: n.qname,
        kind,
        source: n.source,
        degree: n.degree,
        community: n.community,
        line_start: n.line_start,
        line_end: n.line_end,
        decorators: n.decorators,
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        size: Math.max(2, NODE_TYPE_SIZES[kind] || 3),
        color: NODE_TYPE_COLORS[kind] || '#6b7280',
      });
    });

    graphData.links.forEach(l => {
      const s = String(l.source), t = String(l.target);
      if (!graph.hasNode(s) || !graph.hasNode(t)) return;
      const edgeKind = normalizeKind(l.type || l.kind || '');
      visibleEdgeTypes.add(edgeKind);
      graph.addEdge(s, t, {
        edgeKind,
        confidence: l.confidence || 0,
        color: EDGE_TYPE_COLORS[edgeKind] || '#3a3a4a',
        size: 0.5 + (l.confidence || 0) * 1,
      });
    });
  }

  // ── Sigma render ─────────────────────────────────────────────
  function renderSigma() {
    sigma = new Sigma(graph, graphCanvas, {
      renderEdgeLabels: false,
      defaultEdgeType: 'line',
      minCameraRatio: 0.02,
      maxCameraRatio: 5,
      zIndex: true,
      nodeReducer,
      edgeReducer,
    });

    sigma.on('clickNode', ({ node }) => selectNodeById(node));
    sigma.on('enterNode', ({ node }) => { hoveredNodeId = node; showTooltip(node); sigma.refresh({ skipIndexation: true }); });
    sigma.on('leaveNode', () => { hoveredNodeId = null; hideTooltip(); sigma.refresh({ skipIndexation: true }); });
    sigma.on('clickStage', () => clearSelection());
  }

  // ── Reducers (cheap GPU-side styling, no DOM/graph mutation) ─
  function nodeReducer(node, data) {
    const res = { ...data };

    if (!visibleNodeTypes.has(data.kind)) { res.hidden = true; return res; }
    if (reachableSet && !reachableSet.has(node)) { res.hidden = true; return res; }
    if (activeCommunities.size > 0 && data.community !== undefined && !activeCommunities.has(data.community)) {
      res.hidden = true;
      return res;
    }

    if (communityColorFilterOn && data.community !== undefined) {
      res.color = COMMUNITY_PALETTE[data.community % COMMUNITY_PALETTE.length];
    }

    const focusId = selectedNodeId !== null ? selectedNodeId : hoveredNodeId;
    if (focusId !== null) {
      const connected = neighborSet(focusId);
      if (node === focusId) {
        res.zIndex = 2;
        res.highlighted = true;
      } else if (connected.has(node)) {
        res.zIndex = 1;
      } else {
        res.color = dim(res.color);
      }
    }
    return res;
  }

  function edgeReducer(edge, data) {
    const res = { ...data };
    const type = data.edgeKind || '';
    if (!visibleEdgeTypes.has(type)) { res.hidden = true; return res; }

    if (reachableSet) {
      const [s, t] = graph.extremities(edge);
      if (!reachableSet.has(s) || !reachableSet.has(t)) { res.hidden = true; return res; }
    }

    const focusId = selectedNodeId !== null ? selectedNodeId : hoveredNodeId;
    if (focusId !== null) {
      const [s, t] = graph.extremities(edge);
      if (s === focusId || t === focusId) {
        res.zIndex = 1;
        res.color = EDGE_TYPE_COLORS[type] || '#3a3a4a';
      } else {
        res.color = '#1a1a24';
      }
    }
    return res;
  }

  function neighborSet(id) {
    const s = new Set([id]);
    graph.forEachNeighbor(id, n => s.add(n));
    return s;
  }

  function dim(hex) {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || '');
    if (!m) return hex;
    const bg = { r: 10, g: 10, b: 16 };
    const mix = v => Math.round(bg.r + (v - bg.r) * 0.15);
    const r = mix(parseInt(m[1], 16)), g = mix(parseInt(m[2], 16)), b = mix(parseInt(m[3], 16));
    return `#${[r, g, b].map(x => x.toString(16).padStart(2, '0')).join('')}`;
  }

  // ── Simple force layout (no external deps, grid-bucketed repulsion) ──
  function runLayout(followCamera = false) {
    const n = graph.order;
    if (n === 0) return;
    followLayoutCamera = followCamera;
    const duration = n > 3000 ? 6000 : n > 1000 ? 4000 : n > 300 ? 2500 : 1200;
    const start = performance.now();
    const ids = graph.nodes();
    const repulsion = n > 1500 ? 400 : 800;
    const linkDistance = 60;
    const cellSize = Math.max(40, linkDistance);
    let frameCount = 0;

    function tick() {
      const elapsed = performance.now() - start;
      const t = Math.min(1, elapsed / duration);
      const cooling = 1 - t;
      if (cooling <= 0) {
        sigma.refresh();
        if (followLayoutCamera) zoomToFit(true);
        layoutFrame = null;
        return;
      }

      const pos = {};
      ids.forEach(id => { pos[id] = { x: graph.getNodeAttribute(id, 'x'), y: graph.getNodeAttribute(id, 'y') }; });
      const disp = {};
      ids.forEach(id => { disp[id] = { x: 0, y: 0 }; });

      // Bucket nodes into a spatial grid so repulsion only checks nearby
      // cells (O(n) instead of O(n^2) — needed once graphs hit thousands
      // of nodes, same class of problem the old per-frame D3/SVG sim had).
      const grid = new Map();
      ids.forEach(id => {
        const p = pos[id];
        const key = `${Math.floor(p.x / cellSize)},${Math.floor(p.y / cellSize)}`;
        let bucket = grid.get(key);
        if (!bucket) { bucket = []; grid.set(key, bucket); }
        bucket.push(id);
      });

      ids.forEach(a => {
        const pa = pos[a];
        const cx = Math.floor(pa.x / cellSize), cy = Math.floor(pa.y / cellSize);
        for (let gx = cx - 1; gx <= cx + 1; gx++) {
          for (let gy = cy - 1; gy <= cy + 1; gy++) {
            const bucket = grid.get(`${gx},${gy}`);
            if (!bucket) continue;
            for (const b of bucket) {
              if (b === a) continue;
              const pb = pos[b];
              let dx = pa.x - pb.x, dy = pa.y - pb.y;
              let d2 = dx * dx + dy * dy;
              if (d2 < 0.01) { dx = (Math.random() - 0.5); dy = (Math.random() - 0.5); d2 = 0.01; }
              const force = repulsion / d2;
              const d = Math.sqrt(d2);
              disp[a].x += (dx / d) * force;
              disp[a].y += (dy / d) * force;
            }
          }
        }
      });

      // Attraction along edges
      graph.forEachEdge((edge, attrs, s, t) => {
        const ps = pos[s], pt = pos[t];
        if (!ps || !pt) return;
        const dx = ps.x - pt.x, dy = ps.y - pt.y;
        const d = Math.max(0.01, Math.sqrt(dx * dx + dy * dy));
        const force = (d - linkDistance) * 0.02;
        const fx = (dx / d) * force, fy = (dy / d) * force;
        disp[s].x -= fx; disp[s].y -= fy;
        disp[t].x += fx; disp[t].y += fy;
      });

      // Weak centering gravity
      ids.forEach(id => {
        const p = pos[id];
        disp[id].x -= p.x * 0.005;
        disp[id].y -= p.y * 0.005;
      });

      ids.forEach(id => {
        const p = pos[id];
        const d = disp[id];
        const scale = 0.15 * cooling;
        graph.setNodeAttribute(id, 'x', p.x + d.x * scale);
        graph.setNodeAttribute(id, 'y', p.y + d.y * scale);
      });

      sigma.refresh({ skipIndexation: true });
      frameCount++;
      if (followLayoutCamera && frameCount % 15 === 0) zoomToFit(false);
      layoutFrame = requestAnimationFrame(tick);
    }

    if (layoutFrame) cancelAnimationFrame(layoutFrame);
    layoutFrame = requestAnimationFrame(tick);
  }

  // ── Events ───────────────────────────────────────────────────
  function setupEventListeners() {
    document.getElementById('zoomIn').addEventListener('click', () => {
      followLayoutCamera = false;
      if (sigma) sigma.getCamera().animatedZoom({ duration: 300 });
    });
    document.getElementById('zoomOut').addEventListener('click', () => {
      followLayoutCamera = false;
      if (sigma) sigma.getCamera().animatedUnzoom({ duration: 300 });
    });
    document.getElementById('zoomFit').addEventListener('click', () => {
      followLayoutCamera = false;
      zoomToFit();
    });
    document.getElementById('resetLayout').addEventListener('click', () => runLayout(true));
    document.getElementById('closeRightPanel').addEventListener('click', toggleRightPanel);
    document.getElementById('clearSelection').addEventListener('click', clearSelection);
    document.getElementById('toggleHelp').addEventListener('click', showHelp);
    document.getElementById('closeHelp').addEventListener('click', hideHelp);

    let searchTimeout = null;
    searchInput.addEventListener('input', e => {
      clearTimeout(searchTimeout);
      searchTimeout = setTimeout(() => performSearch(e.target.value.trim()), 150);
    });
    document.addEventListener('mousedown', e => {
      if (!searchResultsEl.contains(e.target) && e.target !== searchInput) {
        searchResultsEl.classList.add('hidden');
      }
    });

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById('tab' + btn.dataset.tab.charAt(0).toUpperCase() + btn.dataset.tab.slice(1)).classList.add('active');
      });
    });

    // Node and edge rows are rendered after listeners are registered, so use
    // delegation from their stable containers.
    document.getElementById('nodeTypeFilters').addEventListener('click', event => {
      const item = event.target.closest('.filter-item');
      if (!item) return;
      const type = item.dataset.type;
      if (visibleNodeTypes.has(type)) {
        visibleNodeTypes.delete(type);
        item.classList.remove('active');
      } else {
        visibleNodeTypes.add(type);
        item.classList.add('active');
      }
      sigma.refresh();
    });

    document.getElementById('edgeTypeFilters').addEventListener('click', event => {
      const item = event.target.closest('.filter-item');
      if (!item) return;
      const type = item.dataset.type;
      if (visibleEdgeTypes.has(type)) {
        visibleEdgeTypes.delete(type);
        item.classList.remove('active');
      } else {
        visibleEdgeTypes.add(type);
        item.classList.add('active');
      }
      applyDepthFilter();
      sigma.refresh();
    });

    // Depth filters
    document.querySelectorAll('.depth-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.depth-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentDepthFilter = parseInt(btn.dataset.depth, 10);
        applyDepthFilter();
        sigma.refresh();
      });
    });

    // File search
    fileSearch.addEventListener('input', e => renderFileTree(e.target.value.trim().toLowerCase()));

    // Resize
    window.addEventListener('resize', () => { if (sigma) sigma.resize(); });
    graphCanvas.addEventListener('pointerdown', () => { followLayoutCamera = false; });
    graphCanvas.addEventListener('wheel', () => { followLayoutCamera = false; }, { passive: true });
  }

  // ── Keyboard ─────────────────────────────────────────────────
  function setupKeyboardShortcuts() {
    document.addEventListener('keydown', e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); searchInput.focus(); }
      if (e.key === 'Escape') {
        if (!selectionBar.classList.contains('hidden')) clearSelection();
        hideHelp(); searchResultsEl.classList.add('hidden'); searchInput.blur();
      }
      if (searchResults.length > 0 && document.activeElement === searchInput) {
        if (e.key === 'ArrowDown') { e.preventDefault(); selectedIndex = Math.min(selectedIndex + 1, searchResults.length - 1); updateSearchSelection(); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); selectedIndex = Math.max(selectedIndex - 1, 0); updateSearchSelection(); }
        else if (e.key === 'Enter' && selectedIndex >= 0) { e.preventDefault(); selectNodeById(String(searchResults[selectedIndex].id)); }
      }
    });
  }

  // ── Search ───────────────────────────────────────────────────
  async function performSearch(query) {
    if (!query) { searchResultsEl.classList.add('hidden'); searchResults = []; return; }
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=10`);
      const data = await response.json();
      searchResults = data.hits.slice(0, 10);
      selectedIndex = -1;
      renderSearchResults();
    } catch (error) { console.error('Search failed:', error); }
  }

  function renderSearchResults() {
    if (searchResults.length === 0) { searchResultsEl.classList.add('hidden'); return; }
    searchResultsEl.innerHTML = searchResults
      .map((hit, i) => `<div class="search-result-item ${i === selectedIndex ? 'selected' : ''}" data-id="${hit.id}">
        <div class="search-result-name">${escapeHtml(hit.label)}</div>
        <div class="search-result-label">${hit.kind}</div></div>`).join('');
    searchResultsEl.querySelectorAll('.search-result-item').forEach(item => {
      item.addEventListener('click', () => selectNodeById(item.dataset.id));
    });
    searchResultsEl.classList.remove('hidden');
  }

  function updateSearchSelection() {
    searchResultsEl.querySelectorAll('.search-result-item').forEach((item, i) => item.classList.toggle('selected', i === selectedIndex));
    const sel = searchResultsEl.querySelector('.search-result-item.selected');
    if (sel) sel.scrollIntoView({ block: 'nearest' });
  }

  // ── Camera helpers ───────────────────────────────────────────
  function zoomToFit(animate) {
    if (!sigma || graph.order === 0) return;
    // Sigma cameras operate in normalized framed-graph coordinates, not the
    // raw layout coordinates stored on graphology nodes. With auto-rescaling,
    // the full graph is centered at (0.5, 0.5) and fits at ratio 1.
    const state = { x: 0.5, y: 0.5, ratio: 1, angle: 0 };
    if (animate) sigma.getCamera().animate(state, { duration: 400 });
    else sigma.getCamera().setState(state);
  }

  function focusOnNode(id) {
    if (!sigma) return;
    const pos = sigma.getNodeDisplayData(id);
    if (!pos) return;
    sigma.getCamera().animate({ x: pos.x, y: pos.y, ratio: 0.15 }, { duration: 500 });
  }

  // ── Tooltip ──────────────────────────────────────────────────
  function showTooltip(id) {
    const d = graph.getNodeAttributes(id);
    const pos = sigma.getNodeDisplayData(id);
    const viewport = sigma.framedGraphToViewport(pos);
    hoverTooltip.querySelector('.tooltip-name').textContent = d.qname || d.displayLabel;
    hoverTooltip.querySelector('.tooltip-meta').textContent = `${d.kind} · deg ${d.degree}`;
    hoverTooltip.style.left = `${viewport.x + 14}px`;
    hoverTooltip.style.top = `${viewport.y - 8}px`;
    hoverTooltip.classList.remove('hidden');
  }
  function hideTooltip() { hoverTooltip.classList.add('hidden'); }

  // ── Selection ────────────────────────────────────────────────
  function selectNodeById(id) {
    if (!graph.hasNode(id)) return;
    selectedNodeId = id;
    const node = graph.getNodeAttributes(id);
    selectionName.textContent = node.qname || node.displayLabel;
    selectionType.textContent = node.kind;
    selectionBar.classList.remove('hidden');
    rightPanel.classList.remove('hidden');
    renderNodeDetails(id, node);
    focusOnNode(id);
    updateFileTreeHighlight(node.source || '');
    searchResultsEl.classList.add('hidden');
    if (currentDepthFilter > 0) applyDepthFilter();
    sigma.refresh();
  }

  function clearSelection() {
    selectedNodeId = null;
    selectionBar.classList.add('hidden');
    rightPanel.classList.add('hidden');
    searchInput.value = '';
    if (currentDepthFilter > 0) applyDepthFilter();
    sigma.refresh();
  }

  // ── Depth filter (BFS reachability, applied via reducers) ────
  function applyDepthFilter() {
    if (currentDepthFilter === 0 || selectedNodeId === null) { reachableSet = null; return; }
    const reachable = new Set([selectedNodeId]);
    let frontier = [selectedNodeId];
    for (let depth = 0; depth < currentDepthFilter && frontier.length > 0; depth++) {
      const next = [];
      frontier.forEach(id => {
        graph.forEachEdge(id, (edge, attrs, s, t) => {
          const type = attrs.edgeKind || '';
          if (!visibleEdgeTypes.has(type)) return;
          const other = s === id ? t : s;
          if (!reachable.has(other)) { reachable.add(other); next.push(other); }
        });
      });
      frontier = next;
    }
    reachableSet = reachable;
  }

  // ── Node details ─────────────────────────────────────────────
  function renderNodeDetails(id, node) {
    const color = NODE_TYPE_COLORS[node.kind] || '#6b7280';
    const connections = getConnections(id);
    const inEdges = connections.in;
    const outEdges = connections.out;

    let html = `
      <div class="detail-section">
        <div class="detail-label">Type</div>
        <div class="detail-value" style="color:${color}">${node.kind}</div>
      </div>
      <div class="detail-section">
        <div class="detail-label">Name</div>
        <div class="detail-value">${escapeHtml(node.qname || node.displayLabel)}</div>
      </div>
      <div class="detail-section">
        <div class="detail-label">Display</div>
        <div class="detail-value">${escapeHtml(node.displayLabel)}</div>
      </div>`;

    if (node.source) {
      html += `<div class="detail-section">
        <div class="detail-label">File</div>
        <div class="detail-value" style="cursor:pointer"><a onclick="window.open('/api/search?q=${encodeURIComponent(node.source)}')">${escapeHtml(node.source)}</a></div>
      </div>`;
    }

    if (node.line_start !== undefined) {
      html += `<div class="detail-section">
        <div class="detail-label">Lines</div>
        <div class="detail-value">${node.line_start}${node.line_end !== undefined ? '-' + node.line_end : ''}</div>
      </div>`;
    }

    if (node.decorators && node.decorators.length > 0) {
      html += `<div class="detail-section">
        <div class="detail-label">Decorators</div>
        <div class="detail-list">${node.decorators.map(d => `<div class="detail-tag">${escapeHtml(d)}</div>`).join('')}</div>
      </div>`;
    }

    html += `<div class="detail-section">
        <div class="detail-label">Stats</div>
        <div class="detail-list">
          <div class="detail-list-item"><span>Degree</span><span style="margin-left:auto">${node.degree}</span></div>
          ${node.community !== undefined ? `<div class="detail-list-item"><span>Community</span><span style="margin-left:auto">${node.community}</span></div>` : ''}
        </div>
      </div>`;

    if (outEdges.length > 0 || inEdges.length > 0) {
      html += `<div class="detail-section">
        <div class="detail-label">Outgoing (${outEdges.length})</div>
        <div class="detail-list">${outEdges.slice(0, 20).map(e => `<div class="detail-list-item" data-id="${e.targetId}">${escapeHtml(e.targetName)} <span class="detail-tag">${escapeHtml(e.type)}</span></div>`).join('')}${outEdges.length > 20 ? `<div class="detail-tag" style="margin-top:4px">+${outEdges.length - 20} more</div>` : ''}</div>
      </div>`;
      html += `<div class="detail-section">
        <div class="detail-label">Incoming (${inEdges.length})</div>
        <div class="detail-list">${inEdges.slice(0, 20).map(e => `<div class="detail-list-item" data-id="${e.sourceId}">${escapeHtml(e.sourceName)} <span class="detail-tag">${escapeHtml(e.type)}</span></div>`).join('')}${inEdges.length > 20 ? `<div class="detail-tag" style="margin-top:4px">+${inEdges.length - 20} more</div>` : ''}</div>
      </div>`;
    }

    nodeDetails.innerHTML = html;
    nodeDetails.querySelectorAll('.detail-list-item[data-id]').forEach(item => {
      item.addEventListener('click', () => selectNodeById(item.dataset.id));
    });
  }

  function getConnections(nodeId) {
    const inEdges = [], outEdges = [];
    graph.forEachEdge(nodeId, (edge, attrs, s, t) => {
      const record = {
        type: attrs.edgeKind || '',
        sourceName: graph.hasNode(s) ? (graph.getNodeAttribute(s, 'qname') || graph.getNodeAttribute(s, 'displayLabel')) : '',
        targetName: graph.hasNode(t) ? (graph.getNodeAttribute(t, 'qname') || graph.getNodeAttribute(t, 'displayLabel')) : '',
        sourceId: s,
        targetId: t,
      };
      if (s === nodeId) outEdges.push(record);
      if (t === nodeId) inEdges.push(record);
    });
    return { in: inEdges, out: outEdges };
  }

  function toggleRightPanel() { rightPanel.classList.toggle('hidden'); }

  // ── File tree ────────────────────────────────────────────────
  function renderFileTree(filter) {
    const files = [...new Set(graph.mapNodes((id, attrs) => attrs.kind === 'File' ? (attrs.source || attrs.displayLabel) : null).filter(Boolean))].sort();
    const filtered = filter ? files.filter(f => f.toLowerCase().includes(filter)) : files;

    if (filtered.length === 0) {
      fileTree.innerHTML = `<div class="loading-state">${files.length === 0 ? 'No files' : 'No matches'}</div>`;
      return;
    }

    fileTree.innerHTML = filtered.map(file => {
      const name = file.split('/').pop() || file;
      return `<div class="tree-item" data-file="${escapeHtml(file)}">
        <span class="tree-icon">📄</span>
        <span class="tree-name">${escapeHtml(truncate(name, 30))}</span>
      </div>`;
    }).join('');

    fileTree.querySelectorAll('.tree-item').forEach(item => {
      item.addEventListener('click', () => {
        const file = item.dataset.file;
        const fileNodeIds = graph.filterNodes((id, attrs) => attrs.source === file || attrs.displayLabel === file);
        if (fileNodeIds.length > 0) {
          const preferred = fileNodeIds.find(id => ['Function', 'Method', 'Class'].includes(graph.getNodeAttribute(id, 'kind')));
          selectNodeById(preferred || fileNodeIds[0]);
        }
      });
    });
  }

  function updateFileTreeHighlight(source) {
    fileTree.querySelectorAll('.tree-item').forEach(item => {
      item.classList.toggle('active', item.dataset.file === source);
    });
  }

  // ── Legend ───────────────────────────────────────────────────
  function renderLegend() {
    const communities = [...new Set(graph.mapNodes((id, attrs) => attrs.community).filter(c => c !== undefined))].sort((a, b) => a - b);
    if (communities.length === 0) { legend.innerHTML = ''; return; }
    communities.forEach(c => activeCommunities.add(c));

    legend.innerHTML = communities.map(comm => {
      const color = COMMUNITY_PALETTE[comm % COMMUNITY_PALETTE.length];
      const count = graph.filterNodes((id, attrs) => attrs.community === comm).length;
      return `<div class="legend-item" data-community="${comm}" data-type="community">
        <div class="legend-dot" style="background:${color}"></div>
        <span>Community ${comm}</span>
        <span class="legend-count">${count}</span>
      </div>`;
    }).join('');

    legend.querySelectorAll('.legend-item').forEach(item => {
      item.addEventListener('click', () => {
        const comm = parseInt(item.dataset.community, 10);
        communityColorFilterOn = true;
        if (activeCommunities.has(comm)) {
          activeCommunities.delete(comm);
          item.classList.add('inactive');
        } else {
          activeCommunities.add(comm);
          item.classList.remove('inactive');
        }
        sigma.refresh();
      });
    });
  }

  // ── Node type filters ────────────────────────────────────────
  function renderFileTypeFilters() {
    const presentTypes = [...new Set(graph.mapNodes((id, attrs) => attrs.kind).filter(Boolean))];
    const types = [...new Set([...Object.keys(NODE_TYPE_COLORS), ...presentTypes])];
    document.getElementById('nodeTypeFilters').innerHTML = types.map(type => {
      const count = graph.filterNodes((id, attrs) => attrs.kind === type).length;
      const color = NODE_TYPE_COLORS[type] || '#6b7280';
      return `<div class="filter-item active" data-type="${type}">
        <div class="filter-dot" style="background:${color}"></div>
        <span class="filter-label">${type} (${count})</span>
        <div class="filter-toggle">✓</div>
      </div>`;
    }).join('');
  }

  // ── Edge type filters ────────────────────────────────────────
  function renderEdgeTypeFilters() {
    const edgeTypes = ['Calls', 'Contains', 'Defines', 'Imports', 'Extends', 'Implements', 'MemberOf', 'Reference'];
    const presentTypes = [...new Set(graph.mapEdges((edge, attrs) => attrs.edgeKind || '').filter(Boolean))];
    const allTypes = [...new Set([...edgeTypes, ...presentTypes])];

    document.getElementById('edgeTypeFilters').innerHTML = allTypes.map(type => {
      const color = EDGE_TYPE_COLORS[type] || '#3a3a4a';
      const count = graph.filterEdges((edge, attrs) => (attrs.edgeKind || '') === type).length;
      const active = visibleEdgeTypes.has(type);
      return `<div class="filter-item ${active ? 'active' : ''}" data-type="${type}">
        <div class="filter-edge-line" style="background:${color}"></div>
        <span class="filter-label">${type || '(none)'} (${count})</span>
        <div class="filter-toggle">✓</div>
      </div>`;
    }).join('');
  }

  // ── Stats ────────────────────────────────────────────────────
  function updateStats() {
    nodeCountEl.textContent = graph.order.toLocaleString();
    edgeCountEl.textContent = graph.size.toLocaleString();
  }

  // ── Help ─────────────────────────────────────────────────────
  function showHelp() { document.getElementById('helpModal').classList.remove('hidden'); }
  function hideHelp() { document.getElementById('helpModal').classList.add('hidden'); }

  // ── Utilities ────────────────────────────────────────────────
  function escapeHtml(s) { if (!s) return ''; return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
  function truncate(s, n) { return s.length > n ? s.slice(0, n - 3) + '…' : s; }

  // ── Start ────────────────────────────────────────────────────
  init();
})();
