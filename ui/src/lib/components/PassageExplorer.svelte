<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api';
	import type { PassageMetrics } from '$lib/api';
	import ExportButton from './ExportButton.svelte';

	let data: PassageMetrics[] = $state([]);
	let loading = $state(true);
	let error = $state('');

	let xAxis = $state('token_metonymy_idx');
	let yAxis = $state('mean_surprisal');
	let colorBy = $state('family');
	let selectedPoint: PassageMetrics | null = $state(null);

	let customText = $state('');
	let customLoading = $state(false);
	let customPoint: PassageMetrics | null = $state(null);

	const METRICS = [
		{ id: 'token_metonymy_idx', label: 'Token metonymy' },
		{ id: 'metonymy_idx', label: 'Sentence metonymy' },
		{ id: 'token_diameter', label: 'Token diameter' },
		{ id: 'total_drift', label: 'Sentence diameter' },
		{ id: 'mean_drift', label: 'Mean sentence drift' },
		{ id: 'token_mean_drift', label: 'Mean token drift' },
		{ id: 'mean_surprisal', label: 'Surprisal (GPT-2)' },
		{ id: 'directedness', label: 'Sentence directedness' },
		{ id: 'token_directedness', label: 'Token directedness' },
		{ id: 'n_sentences', label: 'N sentences' },
		{ id: 'n_tokens', label: 'N tokens' },
	];

	const COLORS: Record<string, string> = {
		olmo: '#1f77b4',
		'olmo-tiny': '#aec7e8',
		qwen: '#ff7f0e',
		zephyr: '#2ca02c',
		llama: '#d62728',
		amber: '#9467bd',
		smol: '#8c564b',
		tulu: '#e377c2',
		custom: '#000000',
	};

	const LAYER_SHAPES: Record<string, string> = {
		base: 'circle',
		ego: 'square',
		superego: 'diamond',
		instruct: 'triangle',
		custom: 'star',
	};

	let container: HTMLDivElement;
	let svgEl: SVGSVGElement;

	onMount(async () => {
		try {
			const res = await api.passageMetricsCsv();
			data = res.rows;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	});

	async function analyzeCustom() {
		if (!customText.trim()) return;
		customLoading = true;
		try {
			const result = await api.passageMetrics(customText.trim());
			customPoint = { ...result, psg: customText.trim().slice(0, 200) };
			selectedPoint = customPoint;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			customLoading = false;
		}
	}

	function handleCustomKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
			analyzeCustom();
		}
	}

	function getVal(d: PassageMetrics, key: string): number {
		return (d as any)[key] ?? 0;
	}

	let filteredData = $derived.by(() => {
		return data.filter(d => {
			const x = getVal(d, xAxis);
			const y = getVal(d, yAxis);
			return isFinite(x) && isFinite(y) && x !== 0 && y !== 0;
		});
	});

	let allPoints = $derived.by(() => {
		const pts = [...filteredData];
		if (customPoint) pts.push(customPoint);
		return pts;
	});

	const W = 700;
	const H = 500;
	const PAD = { top: 30, right: 30, bottom: 50, left: 60 };

	let xRange = $derived.by(() => {
		if (allPoints.length === 0) return [0, 1];
		const vals = allPoints.map(d => getVal(d, xAxis));
		const min = Math.min(...vals);
		const max = Math.max(...vals);
		const pad = (max - min) * 0.05 || 0.1;
		return [min - pad, max + pad];
	});

	let yRange = $derived.by(() => {
		if (allPoints.length === 0) return [0, 1];
		const vals = allPoints.map(d => getVal(d, yAxis));
		const min = Math.min(...vals);
		const max = Math.max(...vals);
		const pad = (max - min) * 0.05 || 0.1;
		return [min - pad, max + pad];
	});

	function scaleX(v: number): number {
		return PAD.left + ((v - xRange[0]) / (xRange[1] - xRange[0])) * (W - PAD.left - PAD.right);
	}
	function scaleY(v: number): number {
		return H - PAD.bottom - ((v - yRange[0]) / (yRange[1] - yRange[0])) * (H - PAD.top - PAD.bottom);
	}

	function pointColor(d: PassageMetrics): string {
		if (colorBy === 'family') return COLORS[d.family] ?? '#999';
		if (colorBy === 'layer') {
			const map: Record<string, string> = {
				base: '#1f77b4', ego: '#2ca02c', superego: '#d62728', instruct: '#ff7f0e', custom: '#000'
			};
			return map[d.model] ?? '#999';
		}
		return '#666';
	}

	function pointPath(d: PassageMetrics, cx: number, cy: number, r: number): string {
		const shape = LAYER_SHAPES[d.model] ?? 'circle';
		if (shape === 'circle') {
			return `M${cx-r},${cy}a${r},${r} 0 1,0 ${r*2},0a${r},${r} 0 1,0 -${r*2},0`;
		}
		if (shape === 'square') {
			return `M${cx-r},${cy-r}h${r*2}v${r*2}h-${r*2}z`;
		}
		if (shape === 'diamond') {
			return `M${cx},${cy-r}L${cx+r},${cy}L${cx},${cy+r}L${cx-r},${cy}z`;
		}
		if (shape === 'triangle') {
			return `M${cx},${cy-r}L${cx+r},${cy+r}L${cx-r},${cy+r}z`;
		}
		// star
		const inner = r * 0.4;
		let path = '';
		for (let i = 0; i < 5; i++) {
			const a1 = (i * 72 - 90) * Math.PI / 180;
			const a2 = ((i * 72) + 36 - 90) * Math.PI / 180;
			path += `${i === 0 ? 'M' : 'L'}${cx + r * Math.cos(a1)},${cy + r * Math.sin(a1)}`;
			path += `L${cx + inner * Math.cos(a2)},${cy + inner * Math.sin(a2)}`;
		}
		return path + 'z';
	}

	function labelFor(id: string): string {
		return METRICS.find(m => m.id === id)?.label ?? id;
	}

	function xTicks(): number[] {
		const [min, max] = xRange;
		const step = niceStep(max - min, 6);
		const ticks = [];
		let v = Math.ceil(min / step) * step;
		while (v <= max) { ticks.push(v); v += step; }
		return ticks;
	}

	function yTicks(): number[] {
		const [min, max] = yRange;
		const step = niceStep(max - min, 6);
		const ticks = [];
		let v = Math.ceil(min / step) * step;
		while (v <= max) { ticks.push(v); v += step; }
		return ticks;
	}

	function niceStep(range: number, maxTicks: number): number {
		const rough = range / maxTicks;
		const pow = Math.pow(10, Math.floor(Math.log10(rough)));
		const norm = rough / pow;
		if (norm < 1.5) return pow;
		if (norm < 3.5) return 2 * pow;
		if (norm < 7.5) return 5 * pow;
		return 10 * pow;
	}

	function layerLabel(m: string): string {
		const map: Record<string, string> = { base: 'BASE', ego: 'SFT', superego: 'DPO', instruct: 'RLVR', custom: 'CUSTOM' };
		return map[m] ?? m.toUpperCase();
	}
</script>

<div class="explorer" bind:this={container}>
	<div class="controls">
		<label>
			<span>X axis</span>
			<select bind:value={xAxis}>
				{#each METRICS as m}
					<option value={m.id}>{m.label}</option>
				{/each}
			</select>
		</label>
		<label>
			<span>Y axis</span>
			<select bind:value={yAxis}>
				{#each METRICS as m}
					<option value={m.id}>{m.label}</option>
				{/each}
			</select>
		</label>
		<label>
			<span>Color</span>
			<select bind:value={colorBy}>
				<option value="family">Family</option>
				<option value="layer">Layer</option>
			</select>
		</label>
		<ExportButton {container} filename="passage_explorer" />
	</div>

	{#if loading}
		<p>Loading passage metrics...</p>
	{:else if error}
		<p class="error">{error}</p>
	{:else}
		<svg bind:this={svgEl} viewBox="0 0 {W} {H}" width={W} height={H}>
			<text x={W / 2} y={16} text-anchor="middle" font-size="13" font-weight="bold">
				Passage Explorer ({filteredData.length} passages)
			</text>

			<!-- Axes -->
			{#each xTicks() as t}
				<line x1={scaleX(t)} x2={scaleX(t)} y1={PAD.top} y2={H - PAD.bottom} stroke="#eee" />
				<text x={scaleX(t)} y={H - PAD.bottom + 16} text-anchor="middle" font-size="10" fill="#666">{t.toFixed(2)}</text>
			{/each}
			{#each yTicks() as t}
				<line x1={PAD.left} x2={W - PAD.right} y1={scaleY(t)} y2={scaleY(t)} stroke="#eee" />
				<text x={PAD.left - 6} y={scaleY(t) + 3} text-anchor="end" font-size="10" fill="#666">{t.toFixed(2)}</text>
			{/each}

			<!-- Axis labels -->
			<text x={W / 2} y={H - 6} text-anchor="middle" font-size="11" fill="#333">{labelFor(xAxis)}</text>
			<text x={14} y={H / 2} text-anchor="middle" font-size="11" fill="#333"
				transform="rotate(-90, 14, {H / 2})">{labelFor(yAxis)}</text>

			<!-- Data points -->
			{#each filteredData as d}
				{@const cx = scaleX(getVal(d, xAxis))}
				{@const cy = scaleY(getVal(d, yAxis))}
				<path
					d={pointPath(d, cx, cy, 3)}
					fill={pointColor(d)}
					opacity={selectedPoint === d ? 1 : 0.5}
					stroke={selectedPoint === d ? '#000' : 'none'}
					stroke-width={selectedPoint === d ? 2 : 0}
					cursor="pointer"
					onclick={() => { selectedPoint = d; }}
				>
					<title>{d.family} {layerLabel(d.model)} — {d.label}</title>
				</path>
			{/each}

			<!-- Custom point -->
			{#if customPoint}
				{@const cx = scaleX(getVal(customPoint, xAxis))}
				{@const cy = scaleY(getVal(customPoint, yAxis))}
				<path
					d={pointPath(customPoint, cx, cy, 6)}
					fill="#000"
					stroke="#ff0"
					stroke-width="2"
					cursor="pointer"
					onclick={() => { selectedPoint = customPoint; }}
				>
					<title>Custom passage</title>
				</path>
			{/if}

			<!-- Legend -->
			{#if colorBy === 'family'}
				{#each Object.entries(COLORS).filter(([k]) => filteredData.some(d => d.family === k) || k === 'custom' && customPoint) as [fam, color], i}
					<rect x={W - PAD.right - 80} y={PAD.top + i * 16} width={10} height={10} fill={color} />
					<text x={W - PAD.right - 66} y={PAD.top + i * 16 + 9} font-size="10">{fam}</text>
				{/each}
			{:else}
				{#each [['base', 'BASE', '#1f77b4'], ['ego', 'SFT', '#2ca02c'], ['superego', 'DPO', '#d62728'], ['instruct', 'RLVR', '#ff7f0e']] as [key, label, color], i}
					<rect x={W - PAD.right - 60} y={PAD.top + i * 16} width={10} height={10} fill={color} />
					<text x={W - PAD.right - 46} y={PAD.top + i * 16 + 9} font-size="10">{label}</text>
				{/each}
			{/if}
		</svg>

		<!-- Custom text input -->
		<div class="custom-input">
			<textarea
				bind:value={customText}
				onkeydown={handleCustomKeydown}
				placeholder="Paste a passage to see where it falls in the space (Cmd+Enter to analyze)"
				rows="3"
			></textarea>
			<button onclick={analyzeCustom} disabled={customLoading || !customText.trim()}>
				{customLoading ? 'Computing...' : 'Analyze'}
			</button>
		</div>

		<!-- Selected passage display -->
		{#if selectedPoint}
			<div class="passage-detail">
				<div class="passage-meta">
					<strong>{selectedPoint.family}</strong>
					<span class="layer-badge">{layerLabel(selectedPoint.model)}</span>
					<span>{selectedPoint.label}</span>
				</div>
				<div class="passage-metrics">
					{#each METRICS as m}
						{@const val = getVal(selectedPoint, m.id)}
						<span class="metric" class:highlight={m.id === xAxis || m.id === yAxis}>
							{m.label}: {val.toFixed(3)}
						</span>
					{/each}
				</div>
				<div class="passage-text">{selectedPoint.psg}</div>
			</div>
		{/if}
	{/if}
</div>

<style>
	.explorer {
		width: 100%;
	}
	.controls {
		display: flex;
		gap: 12px;
		align-items: center;
		margin-bottom: 8px;
		flex-wrap: wrap;
	}
	.controls label {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 12px;
	}
	.controls select {
		font-size: 12px;
		padding: 2px 4px;
	}
	svg {
		display: block;
		background: #fff;
		border: 1px solid #ddd;
		border-radius: 4px;
	}
	.custom-input {
		display: flex;
		gap: 8px;
		margin-top: 8px;
	}
	.custom-input textarea {
		flex: 1;
		font-size: 12px;
		padding: 6px;
		border: 1px solid #ccc;
		border-radius: 4px;
		resize: vertical;
		font-family: inherit;
	}
	.custom-input button {
		padding: 6px 16px;
		font-size: 12px;
		cursor: pointer;
		align-self: flex-start;
	}
	.custom-input button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.passage-detail {
		margin-top: 12px;
		padding: 12px;
		background: #f8f8f8;
		border: 1px solid #ddd;
		border-radius: 4px;
	}
	.passage-meta {
		display: flex;
		gap: 8px;
		align-items: center;
		margin-bottom: 6px;
	}
	.layer-badge {
		background: #333;
		color: #fff;
		padding: 1px 6px;
		border-radius: 3px;
		font-size: 11px;
	}
	.passage-metrics {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 14px;
		margin-bottom: 8px;
		font-size: 11px;
		color: #666;
	}
	.metric.highlight {
		color: #000;
		font-weight: bold;
	}
	.passage-text {
		font-size: 13px;
		line-height: 1.5;
		white-space: pre-wrap;
		max-height: 200px;
		overflow-y: auto;
	}
	.error {
		color: red;
	}
</style>
