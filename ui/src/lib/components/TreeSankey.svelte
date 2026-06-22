<script>
  import { onMount } from 'svelte';

  export let data = { nodes: [], links: [] };
  export let width = 900;
  export let height = 500;
  export let title = '';

  let svg;
  let hoveredNode = null;
  let selectedPath = null;

  const padding = { top: 30, right: 20, bottom: 20, left: 60 };
  const nodeWidth = 14;

  $: positions = [...new Set(data.nodes.map(n => n.position))].sort((a, b) => a - b);
  $: posCount = positions.length || 1;
  $: colWidth = (width - padding.left - padding.right) / posCount;

  $: layoutNodes = computeLayout(data.nodes, data.links, positions);
  $: layoutLinks = computeLinks(data.links, layoutNodes);

  function computeLayout(nodes, links, positions) {
    if (!nodes.length) return [];

    const maxCount = Math.max(...nodes.map(n => n.count));
    const usableHeight = height - padding.top - padding.bottom;

    // Group nodes by position
    const byPos = {};
    for (const pos of positions) {
      byPos[pos] = nodes
        .filter(n => n.position === pos)
        .sort((a, b) => b.count - a.count);
    }

    const result = [];
    for (const pos of positions) {
      const group = byPos[pos] || [];
      const totalCount = group.reduce((s, n) => s + n.count, 0);
      let y = padding.top;
      const gap = 2;

      for (const node of group) {
        const h = Math.max(4, (node.count / totalCount) * usableHeight * 0.85);
        const x = padding.left + (positions.indexOf(pos)) * colWidth;
        result.push({
          ...node,
          x, y, w: nodeWidth, h,
          color: getColor(node.position, node.name),
        });
        y += h + gap;
      }
    }
    return result;
  }

  function computeLinks(links, nodes) {
    if (!links.length || !nodes.length) return [];
    const nodeMap = {};
    for (const n of nodes) nodeMap[n.id] = n;

    return links.map(link => {
      const src = nodeMap[link.source];
      const tgt = nodeMap[link.target];
      if (!src || !tgt) return null;
      return {
        ...link,
        x1: src.x + src.w,
        y1: src.y + src.h / 2,
        x2: tgt.x,
        y2: tgt.y + tgt.h / 2,
        strokeWidth: Math.max(1, link.value / 2),
        opacity: 0.3,
      };
    }).filter(Boolean);
  }

  function getColor(pos, name) {
    const violent = ['kill', 'hit', 'punch', 'kick', 'slap', 'stab', 'murder', 'hurt', 'beat', 'attack', 'destroy', 'burn', 'smash', 'strike', 'bite', 'rip', 'tear'];
    const emotional = ['scream', 'cry', 'yell', 'shout', 'sob', 'weep'];
    const template = ['_', '__', '___', '____', '______', 'Options', 'A.'];

    const lower = (name || '').toLowerCase().trim();
    if (violent.some(v => lower.startsWith(v))) return '#e15759';
    if (emotional.some(v => lower.startsWith(v))) return '#f28e2b';
    if (template.some(v => lower.startsWith(v))) return '#bab0ac';
    if (pos === -1) return '#4e79a7';
    return '#76b7b2';
  }

  function linkPath(link) {
    const mx = (link.x1 + link.x2) / 2;
    return `M${link.x1},${link.y1} C${mx},${link.y1} ${mx},${link.y2} ${link.x2},${link.y2}`;
  }

  function handleNodeHover(node) {
    hoveredNode = node;
  }

  function handleNodeLeave() {
    hoveredNode = null;
  }
</script>

<div class="tree-sankey">
  {#if title}
    <h3>{title}</h3>
  {/if}
  <svg bind:this={svg} {width} {height}>
    <!-- Position labels -->
    {#each positions as pos, i}
      <text
        x={padding.left + i * colWidth + nodeWidth / 2}
        y={padding.top - 10}
        text-anchor="middle"
        font-size="11"
        fill="#666"
      >
        {pos === -1 ? 'prompt' : `pos ${pos}`}
      </text>
    {/each}

    <!-- Links -->
    {#each layoutLinks as link}
      <path
        d={linkPath(link)}
        stroke={hoveredNode && (link.source === hoveredNode.id || link.target === hoveredNode.id) ? '#333' : '#ccc'}
        stroke-width={link.strokeWidth}
        fill="none"
        opacity={hoveredNode && (link.source === hoveredNode.id || link.target === hoveredNode.id) ? 0.7 : 0.25}
      />
    {/each}

    <!-- Nodes -->
    {#each layoutNodes as node}
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <g
        on:mouseenter={() => handleNodeHover(node)}
        on:mouseleave={handleNodeLeave}
        style="cursor: pointer"
      >
        <rect
          x={node.x}
          y={node.y}
          width={node.w}
          height={node.h}
          fill={node.color}
          stroke={hoveredNode?.id === node.id ? '#333' : 'none'}
          stroke-width="2"
          rx="2"
        />
        {#if node.h > 12}
          <text
            x={node.x + node.w + 4}
            y={node.y + node.h / 2 + 4}
            font-size="10"
            fill="#333"
          >
            {node.name} ({node.count})
          </text>
        {/if}
      </g>
    {/each}

    <!-- Tooltip -->
    {#if hoveredNode}
      <g transform="translate({hoveredNode.x + 20}, {Math.max(40, hoveredNode.y - 10)})">
        <rect x="0" y="-15" width="140" height="35" fill="white" stroke="#ccc" rx="4" />
        <text x="5" y="0" font-size="12" font-weight="bold">{hoveredNode.name}</text>
        <text x="5" y="14" font-size="10" fill="#666">
          {hoveredNode.count} gens ({Math.round(hoveredNode.count / (data.n_gens || 100) * 100)}%)
        </text>
      </g>
    {/if}
  </svg>

  <div class="legend">
    <span class="swatch" style="background: #e15759"></span> violent
    <span class="swatch" style="background: #f28e2b"></span> emotional
    <span class="swatch" style="background: #76b7b2"></span> other
    <span class="swatch" style="background: #bab0ac"></span> template
  </div>
</div>

<style>
  .tree-sankey {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
  }
  h3 {
    margin: 0 0 8px 0;
    font-size: 14px;
    color: #333;
  }
  .legend {
    font-size: 11px;
    color: #666;
    margin-top: 4px;
  }
  .swatch {
    display: inline-block;
    width: 12px;
    height: 12px;
    margin: 0 3px 0 10px;
    border-radius: 2px;
    vertical-align: middle;
  }
</style>
