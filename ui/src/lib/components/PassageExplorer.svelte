<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { api } from '$lib/api';
	import type { PassageMetrics } from '$lib/api';
	import ExportButton from './ExportButton.svelte';
	import * as d3 from 'd3';

	let data: PassageMetrics[] = $state([]);
	let loading = $state(true);
	let error = $state('');

	let xAxis = $state('mean_surprisal');
	let yAxis = $state('token_diameter');
	let colorBy: 'family' | 'layer' | 'category' = $state('family');
	let selectedPoint: PassageMetrics | null = $state(null);
	let filterFamily = $state('all');
	let filterLayer = $state('all');

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

	const FAMILY_COLORS: Record<string, string> = {
		olmo: '#1f77b4',
		'olmo-tiny': '#6baed6',
		qwen: '#ff7f0e',
		zephyr: '#2ca02c',
		llama: '#d62728',
		amber: '#9467bd',
		smol: '#8c564b',
		tulu: '#e377c2',
		custom: '#000000',
	};

	const LAYER_COLORS: Record<string, string> = {
		base: '#1f77b4',
		ego: '#2ca02c',
		superego: '#d62728',
		instruct: '#ff7f0e',
		custom: '#000000',
	};

	const CATEGORY_COLORS: Record<string, string> = {
		sexual_explicit: '#d62728',
		sexual_liminal: '#ff9896',
		violence_explicit: '#8c564b',
		violence_liminal: '#c49c94',
		death: '#7f7f7f',
		power: '#9467bd',
		profanity: '#e377c2',
		substance: '#bcbd22',
		neutral: '#17becf',
		custom: '#000000',
	};

	let container: HTMLDivElement;
	let chartDiv: HTMLDivElement;

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

	function getCategory(d: PassageMetrics): string {
		return d.label.replace(/_\d+$/, '');
	}

	function layerLabel(m: string): string {
		const map: Record<string, string> = { base: 'BASE', ego: 'SFT', superego: 'DPO', instruct: 'RLVR', custom: 'CUSTOM' };
		return map[m] ?? m.toUpperCase();
	}

	function pointColor(d: PassageMetrics): string {
		if (d.family === 'custom') return '#000';
		if (colorBy === 'family') return FAMILY_COLORS[d.family] ?? '#999';
		if (colorBy === 'layer') return LAYER_COLORS[d.model] ?? '#999';
		if (colorBy === 'category') return CATEGORY_COLORS[getCategory(d)] ?? '#999';
		return '#666';
	}

	function getVal(d: PassageMetrics, key: string): number {
		return (d as any)[key] ?? 0;
	}

	let families = $derived([...new Set(data.map(d => d.family))].sort());
	let layers = $derived([...new Set(data.map(d => d.model))].sort());

	let filteredData = $derived.by(() => {
		return data.filter(d => {
			const x = getVal(d, xAxis);
			const y = getVal(d, yAxis);
			if (!isFinite(x) || !isFinite(y)) return false;
			if (filterFamily !== 'all' && d.family !== filterFamily) return false;
			if (filterLayer !== 'all' && d.model !== filterLayer) return false;
			return true;
		});
	});

	function labelFor(id: string): string {
		return METRICS.find(m => m.id === id)?.label ?? id;
	}

	async function analyzeCustom() {
		if (!customText.trim()) return;
		customLoading = true;
		error = '';
		try {
			const result = await api.passageMetrics(customText.trim());
			customPoint = { ...result, psg: customText.trim().slice(0, 200) };
			selectedPoint = customPoint;
			await tick();
			drawChart();
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

	function drawChart() {
		if (!chartDiv || filteredData.length === 0) return;
		chartDiv.innerHTML = '';

		const margin = { top: 30, right: 20, bottom: 45, left: 55 };
		const width = 720;
		const height = 500;
		const innerW = width - margin.left - margin.right;
		const innerH = height - margin.top - margin.bottom;

		const allPoints = [...filteredData];
		if (customPoint) allPoints.push(customPoint);

		const xExtent = d3.extent(allPoints, d => getVal(d, xAxis)) as [number, number];
		const yExtent = d3.extent(allPoints, d => getVal(d, yAxis)) as [number, number];
		const xPad = (xExtent[1] - xExtent[0]) * 0.05 || 0.1;
		const yPad = (yExtent[1] - yExtent[0]) * 0.05 || 0.1;

		const x = d3.scaleLinear()
			.domain([xExtent[0] - xPad, xExtent[1] + xPad])
			.range([0, innerW]);
		const y = d3.scaleLinear()
			.domain([yExtent[0] - yPad, yExtent[1] + yPad])
			.range([innerH, 0]);

		const svg = d3.select(chartDiv)
			.append('svg')
			.attr('viewBox', `0 0 ${width} ${height}`)
			.attr('width', width)
			.attr('height', height)
			.style('background', '#fff')
			.style('border', '1px solid #ddd')
			.style('border-radius', '4px');

		// Title
		svg.append('text')
			.attr('x', width / 2)
			.attr('y', 16)
			.attr('text-anchor', 'middle')
			.attr('font-size', 13)
			.attr('font-weight', 'bold')
			.text(`Passage Explorer (${filteredData.length} passages)`);

		const g = svg.append('g')
			.attr('transform', `translate(${margin.left},${margin.top})`);

		// Axes
		g.append('g')
			.attr('transform', `translate(0,${innerH})`)
			.call(d3.axisBottom(x).ticks(8))
			.selectAll('text').attr('font-size', 10);

		g.append('g')
			.call(d3.axisLeft(y).ticks(8))
			.selectAll('text').attr('font-size', 10);

		// Axis labels
		svg.append('text')
			.attr('x', width / 2)
			.attr('y', height - 6)
			.attr('text-anchor', 'middle')
			.attr('font-size', 11)
			.attr('fill', '#333')
			.text(labelFor(xAxis));

		svg.append('text')
			.attr('x', 14)
			.attr('y', height / 2)
			.attr('text-anchor', 'middle')
			.attr('font-size', 11)
			.attr('fill', '#333')
			.attr('transform', `rotate(-90, 14, ${height / 2})`)
			.text(labelFor(yAxis));

		// Grid
		g.append('g')
			.attr('class', 'grid')
			.selectAll('line')
			.data(x.ticks(8))
			.join('line')
			.attr('x1', d => x(d)).attr('x2', d => x(d))
			.attr('y1', 0).attr('y2', innerH)
			.attr('stroke', '#f0f0f0');
		g.append('g')
			.attr('class', 'grid')
			.selectAll('line')
			.data(y.ticks(8))
			.join('line')
			.attr('x1', 0).attr('x2', innerW)
			.attr('y1', d => y(d)).attr('y2', d => y(d))
			.attr('stroke', '#f0f0f0');

		// Data points
		g.selectAll('circle.data')
			.data(filteredData)
			.join('circle')
			.attr('class', 'data')
			.attr('cx', d => x(getVal(d, xAxis)))
			.attr('cy', d => y(getVal(d, yAxis)))
			.attr('r', 3.5)
			.attr('fill', d => pointColor(d))
			.attr('opacity', 0.55)
			.attr('stroke', 'none')
			.attr('cursor', 'pointer')
			.on('click', (_e: MouseEvent, d: PassageMetrics) => {
				selectedPoint = d;
				drawChart();
			})
			.append('title')
			.text(d => `${d.family} ${layerLabel(d.model)} — ${d.label}`);

		// Highlight selected
		if (selectedPoint && selectedPoint.family !== 'custom') {
			const sx = x(getVal(selectedPoint, xAxis));
			const sy = y(getVal(selectedPoint, yAxis));
			g.append('circle')
				.attr('cx', sx).attr('cy', sy)
				.attr('r', 6)
				.attr('fill', 'none')
				.attr('stroke', '#000')
				.attr('stroke-width', 2);
		}

		// Custom point
		if (customPoint) {
			const cx = x(getVal(customPoint, xAxis));
			const cy = y(getVal(customPoint, yAxis));
			g.append('circle')
				.attr('cx', cx).attr('cy', cy)
				.attr('r', 7)
				.attr('fill', '#000')
				.attr('stroke', '#ff0')
				.attr('stroke-width', 2.5)
				.attr('cursor', 'pointer')
				.on('click', () => { selectedPoint = customPoint; drawChart(); });
		}

		// Legend
		const colorMap = colorBy === 'family' ? FAMILY_COLORS
			: colorBy === 'layer' ? LAYER_COLORS
			: CATEGORY_COLORS;
		const activeKeys = colorBy === 'family' ? families
			: colorBy === 'layer' ? layers
			: [...new Set(filteredData.map(d => getCategory(d)))].sort();
		const labelFn = colorBy === 'layer'
			? (k: string) => layerLabel(k)
			: (k: string) => k;

		const legend = svg.append('g')
			.attr('transform', `translate(${width - margin.right - 90}, ${margin.top})`);

		activeKeys.forEach((key, i) => {
			const row = legend.append('g')
				.attr('transform', `translate(0, ${i * 16})`);
			row.append('circle')
				.attr('cx', 5).attr('cy', 5).attr('r', 4)
				.attr('fill', colorMap[key] ?? '#999');
			row.append('text')
				.attr('x', 14).attr('y', 9)
				.attr('font-size', 10)
				.text(labelFn(key));
		});
	}

	$effect(() => {
		if (!loading && data.length > 0) {
			// Dependencies: re-draw when these change
			void xAxis; void yAxis; void colorBy; void filterFamily; void filterLayer;
			void filteredData;
			tick().then(drawChart);
		}
	});
</script>

<div class="explorer" bind:this={container}>
	<div class="controls">
		<label>
			<span>X</span>
			<select bind:value={xAxis}>
				{#each METRICS as m}
					<option value={m.id}>{m.label}</option>
				{/each}
			</select>
		</label>
		<label>
			<span>Y</span>
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
				<option value="category">Category</option>
			</select>
		</label>
		<label>
			<span>Family</span>
			<select bind:value={filterFamily}>
				<option value="all">all</option>
				{#each families as f}
					<option value={f}>{f}</option>
				{/each}
			</select>
		</label>
		<label>
			<span>Layer</span>
			<select bind:value={filterLayer}>
				<option value="all">all</option>
				{#each layers as l}
					<option value={l}>{layerLabel(l)}</option>
				{/each}
			</select>
		</label>
		<ExportButton {container} filename="passage_explorer" />
	</div>

	{#if loading}
		<p>Loading passage metrics...</p>
	{:else if error && !data.length}
		<p class="error">{error}</p>
	{:else}
		<div bind:this={chartDiv}></div>

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
		{#if error}
			<p class="error">{error}</p>
		{/if}

		{#if selectedPoint}
			<div class="passage-detail">
				<div class="passage-meta">
					<strong>{selectedPoint.family}</strong>
					<span class="layer-badge">{layerLabel(selectedPoint.model)}</span>
					<span class="label-text">{selectedPoint.label}</span>
				</div>
				<div class="passage-metrics">
					{#each METRICS as m}
						{@const val = getVal(selectedPoint, m.id)}
						<span class="metric" class:highlight={m.id === xAxis || m.id === yAxis}>
							{m.label}: <strong>{val.toFixed(3)}</strong>
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
		gap: 10px;
		align-items: center;
		margin-bottom: 8px;
		flex-wrap: wrap;
	}
	.controls label {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 12px;
		color: #666;
	}
	.controls select {
		font-size: 12px;
		padding: 2px 4px;
	}
	.custom-input {
		display: flex;
		gap: 8px;
		margin-top: 10px;
	}
	.custom-input textarea {
		flex: 1;
		font-size: 12px;
		padding: 6px 8px;
		border: 1px solid #ccc;
		border-radius: 4px;
		resize: vertical;
		font-family: inherit;
		line-height: 1.4;
	}
	.custom-input button {
		padding: 6px 16px;
		font-size: 12px;
		cursor: pointer;
		align-self: flex-start;
		border: 1px solid #ccc;
		border-radius: 4px;
		background: #f8f8f8;
	}
	.custom-input button:hover:not(:disabled) {
		background: #eee;
	}
	.custom-input button:disabled {
		opacity: 0.5;
		cursor: not-allowed;
	}
	.passage-detail {
		margin-top: 12px;
		padding: 12px 14px;
		background: #fafafa;
		border: 1px solid #ddd;
		border-radius: 4px;
	}
	.passage-meta {
		display: flex;
		gap: 8px;
		align-items: center;
		margin-bottom: 6px;
		font-size: 13px;
	}
	.layer-badge {
		background: #333;
		color: #fff;
		padding: 1px 6px;
		border-radius: 3px;
		font-size: 11px;
		font-weight: 600;
	}
	.label-text {
		color: #888;
	}
	.passage-metrics {
		display: flex;
		flex-wrap: wrap;
		gap: 4px 12px;
		margin-bottom: 8px;
		font-size: 11px;
		color: #888;
	}
	.metric.highlight {
		color: #000;
	}
	.passage-text {
		font-size: 13px;
		line-height: 1.5;
		white-space: pre-wrap;
		max-height: 200px;
		overflow-y: auto;
		color: #333;
	}
	.error {
		color: #d62728;
		font-size: 13px;
	}
</style>
