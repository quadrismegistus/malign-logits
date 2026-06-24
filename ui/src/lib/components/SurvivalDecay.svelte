<script lang="ts">
	import { onMount } from 'svelte';

	interface DecayRow {
		family: string;
		prompt: string;
		category: string;
		annotator: string;
		prefix_len: number;
		survived: number;
		total: number;
		survival_rate: number;
	}

	let rows: DecayRow[] = $state([]);
	let loading = $state(true);
	let error = $state('');
	let selectedFamily = $state('all');
	let selectedCategory = $state('all');

	let families = $derived([...new Set(rows.map(r => r.family))].sort());
	let categories = $derived([...new Set(rows.map(r => r.category))].sort());

	const FAMILY_COLORS: Record<string, string> = {
		'OLMo-2-0425-1B': '#4e79a7', 'Olmo-3-1025-7B': '#76b7b2',
		'Qwen2.5-7B': '#f28e2b', 'Qwen3-8B-Base': '#edc948',
		'Amber': '#b07aa1', 'Mistral-7B-v0.1': '#59a14f',
		'Yi-9B': '#ff9da7', 'pythia-6.9b': '#86bcb6',
		'deepseek-llm-7b-base': '#9c755f', 'falcon-7b': '#e15759',
	};

	let chartData = $derived.by(() => {
		let filtered = rows;
		if (selectedFamily !== 'all') filtered = filtered.filter(r => r.family === selectedFamily);
		if (selectedCategory !== 'all') filtered = filtered.filter(r => r.category === selectedCategory);

		const grouped: Record<string, Record<number, number[]>> = {};
		for (const r of filtered) {
			if (!grouped[r.family]) grouped[r.family] = {};
			if (!grouped[r.family][r.prefix_len]) grouped[r.family][r.prefix_len] = [];
			grouped[r.family][r.prefix_len].push(r.survival_rate);
		}

		const lines: { family: string; points: { x: number; y: number }[]; color: string }[] = [];
		for (const [fam, prefixes] of Object.entries(grouped)) {
			const points: { x: number; y: number }[] = [];
			for (let p = 1; p <= 10; p++) {
				const vals = prefixes[p] || [];
				if (vals.length > 0) {
					const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
					points.push({ x: p, y: mean });
				}
			}
			lines.push({ family: fam, points, color: FAMILY_COLORS[fam] || '#888' });
		}
		return lines.sort((a, b) => {
			const lastA = a.points[a.points.length - 1]?.y ?? 0;
			const lastB = b.points[b.points.length - 1]?.y ?? 0;
			return lastB - lastA;
		});
	});

	const W = 700;
	const H = 400;
	const PAD = { top: 20, right: 140, bottom: 40, left: 50 };

	function sx(x: number): number { return PAD.left + ((x - 1) / 9) * (W - PAD.left - PAD.right); }
	function sy(y: number): number { return PAD.top + (1 - y) * (H - PAD.top - PAD.bottom); }

	function linePath(points: { x: number; y: number }[]): string {
		return points.map((p, i) => `${i === 0 ? 'M' : 'L'}${sx(p.x)},${sy(p.y)}`).join(' ');
	}

	async function loadData() {
		try {
			let res = await fetch('/api/api/data/csv?name=survival_decay&limit=20000');
			if (!res.ok) {
				res = await fetch('/api/data/csv?name=survival_decay&limit=20000');
			}
			if (!res.ok) throw new Error('Failed to load survival data');
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

<div class="survival">
	<div class="controls">
		<label class="control">
			<span>Family</span>
			<select bind:value={selectedFamily}>
				<option value="all">all families</option>
				{#each families as f}
					<option value={f}>{f}</option>
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
		<div class="loading">Loading survival decay data...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else}
		<div class="chart-container">
			<svg viewBox="0 0 {W} {H}" class="chart">
				<!-- Grid lines -->
				{#each [0, 0.25, 0.5, 0.75, 1.0] as tick}
					<line x1={PAD.left} y1={sy(tick)} x2={W - PAD.right} y2={sy(tick)} class="grid-line" />
					<text x={PAD.left - 6} y={sy(tick) + 4} class="y-label">{(tick * 100).toFixed(0)}%</text>
				{/each}
				{#each Array.from({length: 10}, (_, i) => i + 1) as tick}
					<text x={sx(tick)} y={H - PAD.bottom + 16} class="x-label">{tick}</text>
				{/each}
				<text x={W / 2} y={H - 4} class="axis-title">prefix length (tokens)</text>
				<text x={12} y={H / 2} class="axis-title" transform="rotate(-90, 12, {H / 2})">survival rate</text>

				<!-- Lines -->
				{#each chartData as line}
					<path d={linePath(line.points)} fill="none" stroke={line.color} stroke-width="2.5" stroke-linejoin="round" />
					{#each line.points as p}
						<circle cx={sx(p.x)} cy={sy(p.y)} r="3" fill={line.color} />
					{/each}
					<!-- Label at end -->
					{@const last = line.points[line.points.length - 1]}
					{#if last}
						<text x={sx(last.x) + 8} y={sy(last.y) + 4} fill={line.color} class="line-label">
							{line.family.replace(/-/g, ' ')} ({(last.y * 100).toFixed(0)}%)
						</text>
					{/if}
				{/each}
			</svg>
		</div>

		<div class="summary">
			{rows.length} data points across {families.length} families
			{#if selectedCategory !== 'all'} &middot; category: {selectedCategory}{/if}
		</div>
	{/if}
</div>

<style>
	.survival {
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

	.line-label {
		font-size: 10px;
		font-family: 'SF Mono', monospace;
	}

	.summary {
		font-size: 12px;
		color: #888;
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
