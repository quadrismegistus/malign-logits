<script lang="ts">
	import { onMount } from 'svelte';

	interface TrajRow {
		family: string;
		annotator: string;
		category: string;
		position: number;
		mean_resist: number;
		std_resist: number;
		n: number;
	}

	let rows: TrajRow[] = $state([]);
	let loading = $state(true);
	let error = $state('');
	let selectedFamily = $state('all');
	let selectedCategory = $state('all');

	let families = $derived([...new Set(rows.map(r => r.family))].sort());
	let categories = $derived([...new Set(rows.map(r => r.category))].sort());

	const CAT_COLORS: Record<string, string> = {
		sexual: '#e15759', violence: '#f28e2b', death: '#b07aa1',
		institutional: '#4e79a7', neutral: '#59a14f', power: '#edc948',
		profanity: '#9c755f', substance: '#76b7b2',
	};

	const FAMILY_COLORS: Record<string, string> = {
		'OLMo-2-0425-1B': '#4e79a7', 'Olmo-3-1025-7B': '#76b7b2',
		'Qwen2.5-7B': '#f28e2b', 'Qwen3-8B-Base': '#edc948',
		'Amber': '#b07aa1', 'Mistral-7B-v0.1': '#59a14f',
		'Yi-9B': '#ff9da7', 'pythia-6.9b': '#86bcb6',
		'deepseek-llm-7b-base': '#9c755f', 'falcon-7b': '#e15759',
	};

	let chartLines = $derived.by(() => {
		let filtered = rows;
		if (selectedFamily !== 'all') filtered = filtered.filter(r => r.family === selectedFamily);
		if (selectedCategory !== 'all') filtered = filtered.filter(r => r.category === selectedCategory);

		const byLine = selectedFamily === 'all' ? 'family' : 'category';
		const colorMap = byLine === 'family' ? FAMILY_COLORS : CAT_COLORS;

		const grouped: Record<string, Record<number, number[]>> = {};
		for (const r of filtered) {
			const key = byLine === 'family' ? r.family : r.category;
			if (!grouped[key]) grouped[key] = {};
			if (!grouped[key][r.position]) grouped[key][r.position] = [];
			grouped[key][r.position].push(r.mean_resist);
		}

		const lines: { label: string; points: { x: number; y: number }[]; color: string }[] = [];
		for (const [label, positions] of Object.entries(grouped)) {
			const points: { x: number; y: number }[] = [];
			for (let p = 0; p < 10; p++) {
				const vals = positions[p] || [];
				if (vals.length > 0) {
					points.push({ x: p, y: vals.reduce((a, b) => a + b, 0) / vals.length });
				}
			}
			lines.push({ label, points, color: colorMap[label] || '#888' });
		}
		return lines.sort((a, b) => a.label.localeCompare(b.label));
	});

	const W = 700;
	const H = 400;
	const PAD = { top: 20, right: 150, bottom: 40, left: 50 };

	let yRange = $derived.by(() => {
		const allY = chartLines.flatMap(l => l.points.map(p => p.y));
		if (allY.length === 0) return { min: -1, max: 2 };
		const min = Math.min(...allY, 0);
		const max = Math.max(...allY, 0);
		const pad = (max - min) * 0.1 || 0.5;
		return { min: min - pad, max: max + pad };
	});

	function sx(x: number): number { return PAD.left + (x / 9) * (W - PAD.left - PAD.right); }
	function sy(y: number): number {
		return PAD.top + ((yRange.max - y) / (yRange.max - yRange.min)) * (H - PAD.top - PAD.bottom);
	}

	function linePath(points: { x: number; y: number }[]): string {
		return points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x)},${sy(p.y)}`).join(' ');
	}

	function yTicks(): number[] {
		const range = yRange.max - yRange.min;
		const step = range > 4 ? 1 : range > 2 ? 0.5 : 0.25;
		const ticks: number[] = [];
		let t = Math.ceil(yRange.min / step) * step;
		while (t <= yRange.max) {
			ticks.push(t);
			t += step;
		}
		return ticks;
	}

	async function loadData() {
		try {
			let res = await fetch('/api/api/data/csv?name=resistance_trajectories&limit=20000');
			if (!res.ok) res = await fetch('/api/data/csv?name=resistance_trajectories&limit=20000');
			if (!res.ok) throw new Error('Failed to load');
			const data = await res.json();
			rows = data.rows;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	onMount(loadData);
</script>

<div class="trajectories">
	<div class="controls">
		<label class="control">
			<span>Family</span>
			<select bind:value={selectedFamily}>
				<option value="all">all families (group by family)</option>
				{#each families as f}
					<option value={f}>{f} (group by category)</option>
				{/each}
			</select>
		</label>
		<label class="control">
			<span>Category</span>
			<select bind:value={selectedCategory}>
				<option value="all">all categories</option>
				{#each categories as c}
					<option value={c}>{c}</option>
				{/each}
			</select>
		</label>
	</div>

	{#if loading}
		<div class="loading">Loading resistance trajectories...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else}
		<div class="chart-container">
			<svg viewBox="0 0 {W} {H}" class="chart">
				<!-- Zero line -->
				<line x1={PAD.left} y1={sy(0)} x2={W - PAD.right} y2={sy(0)} class="zero-line" />

				<!-- Grid -->
				{#each yTicks() as tick}
					<line x1={PAD.left} y1={sy(tick)} x2={W - PAD.right} y2={sy(tick)} class="grid-line" />
					<text x={PAD.left - 6} y={sy(tick) + 4} class="y-label">{tick > 0 ? '+' : ''}{tick.toFixed(1)}</text>
				{/each}
				{#each Array.from({length: 10}, (_, i) => i) as tick}
					<text x={sx(tick)} y={H - PAD.bottom + 16} class="x-label">{tick}</text>
				{/each}
				<text x={W / 2} y={H - 4} class="axis-title">token position</text>
				<text x={12} y={H / 2} class="axis-title" transform="rotate(-90, 12, {H / 2})">resistance (bits)</text>

				<!-- Facilitation / resistance zones -->
				<text x={PAD.left + 4} y={sy(0) - 6} class="zone-label" fill="#e1575955">← blocked</text>
				<text x={PAD.left + 4} y={sy(0) + 14} class="zone-label" fill="#59a14f55">← facilitated</text>

				<!-- Lines -->
				{#each chartLines as line}
					<path d={linePath(line.points)} fill="none" stroke={line.color} stroke-width="2" stroke-linejoin="round" />
					{#each line.points as p}
						<circle cx={sx(p.x)} cy={sy(p.y)} r="2.5" fill={line.color} />
					{/each}
					{@const last = line.points[line.points.length - 1]}
					{#if last}
						<text x={sx(last.x) + 8} y={sy(last.y) + 4} fill={line.color} class="line-label">
							{line.label}
						</text>
					{/if}
				{/each}
			</svg>
		</div>
	{/if}
</div>

<style>
	.trajectories {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.controls {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		align-items: end;
	}

	.control {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 11px;
		color: #888;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.control select {
		background: #141428;
		border: 1px solid #2a2a44;
		color: #ccc;
		padding: 6px 8px;
		border-radius: 4px;
		font-size: 12px;
		font-family: inherit;
	}

	.control select:focus {
		outline: none;
		border-color: #4e79a7;
	}

	.chart-container {
		max-width: 800px;
	}

	.chart {
		width: 100%;
		height: auto;
	}

	.grid-line {
		stroke: #1a1a2e;
		stroke-width: 1;
	}

	.zero-line {
		stroke: #444;
		stroke-width: 1;
		stroke-dasharray: 4 2;
	}

	.y-label, .x-label {
		font-size: 11px;
		fill: #888;
		font-family: 'SF Mono', monospace;
		text-anchor: end;
	}

	.x-label {
		text-anchor: middle;
	}

	.axis-title {
		font-size: 11px;
		fill: #666;
		text-anchor: middle;
	}

	.zone-label {
		font-size: 10px;
		font-style: italic;
	}

	.line-label {
		font-size: 10px;
		font-family: 'SF Mono', monospace;
	}

	.loading {
		color: #888;
		font-size: 13px;
		padding: 24px;
	}

	.error {
		color: #e15759;
		font-size: 12px;
	}
</style>
