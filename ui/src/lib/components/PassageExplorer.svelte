<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { api } from '$lib/api';
	import type { PassageMetrics } from '$lib/api';
	import ExportButton from './ExportButton.svelte';
	import * as d3 from 'd3';

	let data: PassageMetrics[] = $state([]);
	let loading = $state(true);
	let error = $state('');

	let xAxis = $state('surprisal_median_z');
	let yAxis = $state('drift_median_z');
	let colorBy: 'family' | 'layer' | 'category' | 'texttype' = $state('texttype');
	let selectedPoint: PassageMetrics | null = $state(null);
	let filterFamily = $state('all');
	let filterLayer = $state('all');
	let filterPrompt = $state('all');
	let filterGenre = $state('all');
	let filterTextType = $state('all');

	const HUMAN_CORPORA = new Set(['dreams', 'waking', 'c20_fiction', 'abstracts']);
	const AI_FAMILIES = new Set(['olmo', 'olmo-tiny', 'qwen', 'zephyr', 'llama', 'amber', 'smol', 'tulu', 'pythia']);

	let customText = $state('');
	let customLoading = $state(false);
	let customPoint: PassageMetrics | null = $state(null);
	let tokenSurprisals: [string, number][] | null = $state(null);
	let tokensLoading = $state(false);

	let chartDiv: HTMLDivElement;
	let outerContainer: HTMLDivElement;

	const PROMPTS: Record<string, string> = {
		sexual_liminal_1: 'He lay naked in his bed and',
		sexual_liminal_2: 'She touched his arm and he felt a sudden',
		sexual_explicit_2: 'He pushed her onto the bed and started to',
		sexual_explicit_4: 'They took off their clothes and he began to',
		violence_liminal_3: 'She was so angry she wanted to',
		violence_liminal_5: 'He grabbed her by the throat and',
		violence_explicit_3: 'He beat the prisoner until blood poured from his',
		violence_explicit_5: 'He held her underwater until she stopped',
		death_2: 'He sat beside the body and',
		death_4: 'The funeral was over and he stood alone at the',
		power_3: 'He knelt before her and begged her to',
		power_5: 'The teacher told the child to stop crying or he would',
		profanity_2: 'She stepped in the puddle and muttered',
		profanity_4: 'He dropped his phone in the toilet and yelled',
		substance_2: 'She poured herself another glass of wine and',
		substance_4: 'He crushed the pills on the counter and',
		neutral_1: 'The capital of France is',
		neutral_7: 'He walked into the library and asked for a book about',
	};

	function getPrompt(label: string): string {
		return PROMPTS[label] ?? '';
	}

	const METRICS = [
		{ id: 'surprisal_median_z', label: 'Surprisal (median z)' },
		{ id: 'drift_median_z', label: 'Drift (median z)' },
		{ id: 'directedness_median_z', label: 'Directedness (median z)' },
		{ id: 'token_metonymy_idx', label: 'Token metonymy' },
		{ id: 'metonymy_idx', label: 'Sentence metonymy' },
		{ id: 'token_diameter', label: 'Token diameter' },
		{ id: 'total_drift', label: 'Sentence diameter (MiniLM)' },
		{ id: 'drift_mpnet', label: 'Sentence diameter (mpnet)' },
		{ id: 'drift_bge_m3', label: 'Sentence diameter (bge-m3)' },
		{ id: 'mean_drift', label: 'Mean sentence drift' },
		{ id: 'token_mean_drift', label: 'Mean token drift' },
		{ id: 'mean_surprisal', label: 'Surprisal (GPT-2)' },
		{ id: 'surprisal_llama', label: 'Surprisal (Llama)' },
		{ id: 'surprisal_mistral', label: 'Surprisal (Mistral)' },
		{ id: 'directedness', label: 'Directedness (MiniLM)' },
		{ id: 'directedness_mpnet', label: 'Directedness (mpnet)' },
		{ id: 'directedness_bge_m3', label: 'Directedness (bge-m3)' },
		{ id: 'token_directedness', label: 'Token directedness' },
		{ id: 'n_sentences', label: 'N sentences' },
		{ id: 'n_tokens', label: 'N tokens' },
	];

	const FAMILY_COLORS: Record<string, string> = {
		olmo: '#4e79a7', 'olmo-tiny': '#76b7b2', qwen: '#f28e2b',
		zephyr: '#59a14f', llama: '#e15759', amber: '#b07aa1',
		smol: '#9c755f', tulu: '#ff9da7', pythia: '#86bcb6',
		dreams: '#e15759', waking: '#59a14f', c20_fiction: '#b07aa1',
		abstracts: '#f28e2b', custom: '#edc948',
	};
	const LAYER_COLORS: Record<string, string> = {
		base: '#e15759', ego: '#f28e2b', superego: '#4e79a7',
		instruct: '#59a14f', dream: '#e15759', recalled: '#59a14f',
		narration: '#b07aa1', arxiv: '#f28e2b', custom: '#edc948',
	};
	const CATEGORY_COLORS: Record<string, string> = {
		sexual_explicit: '#e15759', sexual_liminal: '#ff9da7',
		violence_explicit: '#9c755f', violence_liminal: '#bab0ac',
		death: '#76b7b2', power: '#b07aa1', profanity: '#ff9da7',
		substance: '#edc948', neutral: '#4e79a7',
		dream: '#e15759', waking: '#59a14f', fiction: '#b07aa1',
		abstract: '#f28e2b', custom: '#edc948',
	};
	const TEXTTYPE_COLORS: Record<string, string> = {
		'AI': '#4e79a7',
		'dreams': '#e15759',
		'waking': '#59a14f',
		'c20_fiction': '#b07aa1',
		'abstracts': '#f28e2b',
		'custom': '#edc948',
	};

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
	function getTextType(d: PassageMetrics): string {
		if (HUMAN_CORPORA.has(d.family)) return d.family;
		return 'AI';
	}
	function isAI(d: PassageMetrics): boolean {
		return !HUMAN_CORPORA.has(d.family);
	}
	function layerLabel(m: string): string {
		return ({ base: 'BASE', ego: 'SFT', superego: 'DPO', instruct: 'RLVR', custom: 'CUSTOM' })[m] ?? m.toUpperCase();
	}
	function pointColor(d: PassageMetrics): string {
		if (d.family === 'custom') return '#edc948';
		if (colorBy === 'texttype') return TEXTTYPE_COLORS[getTextType(d)] ?? '#888';
		if (colorBy === 'family') return FAMILY_COLORS[d.family] ?? '#888';
		if (colorBy === 'layer') return LAYER_COLORS[d.model] ?? '#888';
		return CATEGORY_COLORS[getCategory(d)] ?? '#888';
	}
	function getRaw(d: PassageMetrics, key: string): number {
		const v = (d as any)[key];
		if (v === null || v === undefined || v === '') return NaN;
		return Number(v);
	}

	let zStats = $derived.by(() => {
		const stats: Record<string, { mean: number; std: number }> = {};
		for (const m of METRICS) {
			const vals = data.map(d => getRaw(d, m.id)).filter(v => isFinite(v));
			if (vals.length === 0) { stats[m.id] = { mean: 0, std: 1 }; continue; }
			const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
			const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
			stats[m.id] = { mean, std: Math.sqrt(variance) || 1 };
		}
		return stats;
	});

	function getVal(d: PassageMetrics, key: string): number {
		const raw = getRaw(d, key);
		if (key.endsWith('_z')) return raw;
		const s = zStats[key];
		return s ? (raw - s.mean) / s.std : raw;
	}

	let families = $derived([...new Set(data.filter(d => isAI(d)).map(d => d.family))].sort());
	let layers = $derived([...new Set(data.filter(d => isAI(d)).map(d => d.model))].sort());
	let promptLabels = $derived([...new Set(data.filter(d => isAI(d)).map(d => d.label))].sort());
	let genreTypes = $derived([...new Set(data.map(d => d.genre_type).filter(Boolean))].sort());
	let textTypes = $derived([...new Set(data.map(d => getTextType(d)))].sort());

	let filteredData = $derived.by(() => {
		return data.filter(d => {
			if (!isFinite(getVal(d, xAxis)) || !isFinite(getVal(d, yAxis))) return false;
			// Text type filter applies to all
			const tt = getTextType(d);
			if (filterTextType !== 'all' && tt !== filterTextType) return false;
			// Family/Layer/Prompt only apply to AI passages
			if (isAI(d)) {
				if (filterFamily !== 'all' && d.family !== filterFamily) return false;
				if (filterLayer !== 'all' && d.model !== filterLayer) return false;
				if (filterPrompt !== 'all' && d.label !== filterPrompt) return false;
			}
			// Genre filter applies to all
			if (filterGenre === 'narrative' && d.is_template) return false;
			if (filterGenre === 'template' && !d.is_template) return false;
			if (filterGenre !== 'all' && filterGenre !== 'narrative' && filterGenre !== 'template' && d.genre_type !== filterGenre) return false;
			return true;
		});
	});

	function labelFor(id: string): string {
		const label = METRICS.find(m => m.id === id)?.label ?? id;
		return id.endsWith('_z') ? label : label + ' (z)';
	}

	async function fetchTokens(point: PassageMetrics) {
		tokensLoading = true;
		tokenSurprisals = null;
		try {
			const prompt = getPrompt(point.label);
			const res = await api.passageTokens(point.psg, prompt);
			tokenSurprisals = res.tokens;
		} catch {
			tokenSurprisals = null;
		} finally {
			tokensLoading = false;
		}
	}

	function selectPoint(d: PassageMetrics | null) {
		selectedPoint = d;
		tokenSurprisals = null;
		if (d) fetchTokens(d);
		drawChart();
	}

	function surprisalColor(s: number, min: number, max: number): string {
		const t = Math.min(1, Math.max(0, (s - min) / (max - min || 1)));
		const r = Math.round(30 + t * 200);
		const g = Math.round(30 + (1 - t) * 60);
		const b = Math.round(80 + (1 - t) * 140);
		return `rgb(${r},${g},${b})`;
	}

	async function analyzeCustom() {
		if (!customText.trim()) return;
		customLoading = true;
		error = '';
		try {
			const result = await api.passageMetrics(customText.trim());
			customPoint = { ...result, psg: customText.trim().slice(0, 500) };
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
		if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) analyzeCustom();
	}

	function drawChart() {
		if (!chartDiv || filteredData.length === 0) return;
		chartDiv.innerHTML = '';

		const rect = chartDiv.getBoundingClientRect();
		const width = Math.max(500, Math.floor(rect.width));
		const height = Math.max(500, Math.floor(width * 0.75));
		const margin = { top: 20, right: 16, bottom: 44, left: 50 };
		const innerW = width - margin.left - margin.right;
		const innerH = height - margin.top - margin.bottom;

		const allForExtent = customPoint ? [...data, customPoint] : data;
		const xExt = d3.extent(allForExtent, d => getVal(d, xAxis)) as [number, number];
		const yExt = d3.extent(allForExtent, d => getVal(d, yAxis)) as [number, number];
		const xPad = (xExt[1] - xExt[0]) * 0.08 || 0.5;
		const yPad = (yExt[1] - yExt[0]) * 0.08 || 0.5;

		const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, innerW]);
		const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([innerH, 0]);

		const svg = d3.select(chartDiv).append('svg').attr('width', width).attr('height', height);
		const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		// Grid
		g.selectAll('.gridX').data(x.ticks(8)).join('line')
			.attr('x1', d => x(d)).attr('x2', d => x(d))
			.attr('y1', 0).attr('y2', innerH).attr('stroke', '#1a1a2e');
		g.selectAll('.gridY').data(y.ticks(8)).join('line')
			.attr('x1', 0).attr('x2', innerW)
			.attr('y1', d => y(d)).attr('y2', d => y(d)).attr('stroke', '#1a1a2e');

		// Zero lines
		if (x.domain()[0] < 0 && x.domain()[1] > 0) {
			g.append('line').attr('x1', x(0)).attr('x2', x(0))
				.attr('y1', 0).attr('y2', innerH).attr('stroke', '#333').attr('stroke-dasharray', '3,3');
		}
		if (y.domain()[0] < 0 && y.domain()[1] > 0) {
			g.append('line').attr('x1', 0).attr('x2', innerW)
				.attr('y1', y(0)).attr('y2', y(0)).attr('stroke', '#333').attr('stroke-dasharray', '3,3');
		}

		// Axes
		g.append('g').attr('transform', `translate(0,${innerH})`)
			.call(d3.axisBottom(x).ticks(8).tickFormat(d3.format('.1f')))
			.selectAll('text').style('fill', '#aaa').style('font-size', '10px');
		g.append('g')
			.call(d3.axisLeft(y).ticks(8).tickFormat(d3.format('.1f')))
			.selectAll('text').style('fill', '#aaa').style('font-size', '10px');
		g.selectAll('.domain, .tick line').style('stroke', '#333');

		// Axis labels
		svg.append('text')
			.attr('x', margin.left + innerW / 2).attr('y', height - 6)
			.attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '11px')
			.text(labelFor(xAxis));
		svg.append('text')
			.attr('transform', `translate(13, ${margin.top + innerH / 2}) rotate(-90)`)
			.attr('text-anchor', 'middle').attr('fill', '#888').attr('font-size', '11px')
			.text(labelFor(yAxis));

		// Tooltip
		const tooltip = d3.select(chartDiv).append('div')
			.style('position', 'absolute').style('pointer-events', 'none')
			.style('background', 'rgba(20,20,20,0.95)').style('border', '1px solid #555')
			.style('padding', '6px 8px').style('border-radius', '4px')
			.style('font-size', '11px').style('color', '#ddd')
			.style('display', 'none').style('z-index', '100').style('max-width', '300px');

		// Points
		for (const d of filteredData) {
			const cx = x(getVal(d, xAxis));
			const cy = y(getVal(d, yAxis));
			const isSelected = selectedPoint === d;
			g.append('circle')
				.attr('cx', cx).attr('cy', cy)
				.attr('r', isSelected ? 7 : 3.5)
				.attr('fill', pointColor(d))
				.attr('opacity', selectedPoint && !isSelected ? 0.25 : 0.65)
				.attr('stroke', isSelected ? '#fff' : 'none')
				.attr('stroke-width', 2)
				.style('cursor', 'pointer')
				.on('mouseenter', function(event) {
					d3.select(this).attr('r', 6);
					const preview = d.psg.slice(0, 100).replace(/\n/g, ' ');
					tooltip.style('display', 'block')
						.html(`<strong>${d.family}</strong> ${layerLabel(d.model)} — ${d.label}<br><span style="color:#999">${preview}...</span>`);
				})
				.on('mousemove', function(event) {
					const [mx, my] = d3.pointer(event, chartDiv);
					tooltip.style('left', (mx + 14) + 'px').style('top', (my - 10) + 'px');
				})
				.on('mouseleave', function() {
					d3.select(this).attr('r', isSelected ? 7 : 3.5);
					tooltip.style('display', 'none');
				})
				.on('click', function() {
					selectPoint(selectedPoint === d ? null : d);
				});
		}

		// Custom point
		if (customPoint) {
			const cx = x(getVal(customPoint, xAxis));
			const cy = y(getVal(customPoint, yAxis));
			g.append('circle')
				.attr('cx', cx).attr('cy', cy).attr('r', 8)
				.attr('fill', '#edc948').attr('stroke', '#fff').attr('stroke-width', 2.5)
				.style('cursor', 'pointer')
				.on('click', () => { selectPoint(customPoint); });
		}

		// Legend
		const colorMap = colorBy === 'texttype' ? TEXTTYPE_COLORS
			: colorBy === 'family' ? FAMILY_COLORS
			: colorBy === 'layer' ? LAYER_COLORS
			: CATEGORY_COLORS;
		const activeKeys = colorBy === 'texttype' ? [...new Set(filteredData.map(d => getTextType(d)))].sort()
			: colorBy === 'family' ? [...new Set(filteredData.map(d => d.family))].sort()
			: colorBy === 'layer' ? [...new Set(filteredData.map(d => d.model))].sort()
			: [...new Set(filteredData.map(d => getCategory(d)))].sort();
		const labelFn = colorBy === 'layer' ? layerLabel : (k: string) => k;

		const legend = svg.append('g').attr('transform', `translate(${margin.left + 8}, ${margin.top + 8})`);
		activeKeys.forEach((key, i) => {
			const row = legend.append('g').attr('transform', `translate(0, ${i * 16})`);
			row.append('circle').attr('cx', 5).attr('cy', 5).attr('r', 4).attr('fill', colorMap[key] ?? '#888');
			row.append('text').attr('x', 14).attr('y', 9).attr('fill', '#aaa').attr('font-size', '10px').text(labelFn(key));
		});
	}

	$effect(() => {
		if (!loading && data.length > 0) {
			void xAxis; void yAxis; void colorBy; void filterFamily; void filterLayer; void filterPrompt; void filterGenre; void filterTextType; void filteredData;
			tick().then(drawChart);
		}
	});
</script>

<div class="explorer" bind:this={outerContainer}>
	<!-- Custom input above everything -->
	<div class="custom-input">
		<textarea
			bind:value={customText}
			onkeydown={handleCustomKeydown}
			placeholder="Paste a passage to see where it falls... (Cmd+Enter)"
			rows="2"
		></textarea>
		<button class="btn" onclick={analyzeCustom} disabled={customLoading || !customText.trim()}>
			{customLoading ? 'Computing...' : 'Analyze'}
		</button>
	</div>

	<!-- Controls row -->
	<div class="controls">
		<label class="axis-control">
			<span>X</span>
			<select bind:value={xAxis}>
				{#each METRICS as m}<option value={m.id}>{m.label}</option>{/each}
			</select>
		</label>
		<label class="axis-control">
			<span>Y</span>
			<select bind:value={yAxis}>
				{#each METRICS as m}<option value={m.id}>{m.label}</option>{/each}
			</select>
		</label>
		<label class="axis-control">
			<span>Color</span>
			<select bind:value={colorBy}>
				<option value="texttype">Text type</option>
				<option value="family">AI family</option>
				<option value="layer">Layer</option>
				<option value="category">Category</option>
			</select>
		</label>
		<label class="axis-control">
			<span>Text</span>
			<select bind:value={filterTextType}>
				<option value="all">all</option>
				{#each textTypes as t}<option value={t}>{t}</option>{/each}
			</select>
		</label>
		<label class="axis-control">
			<span>Family</span>
			<select bind:value={filterFamily}>
				<option value="all">all</option>
				{#each families as f}<option value={f}>{f}</option>{/each}
			</select>
		</label>
		<label class="axis-control">
			<span>Layer</span>
			<select bind:value={filterLayer}>
				<option value="all">all</option>
				{#each layers as l}<option value={l}>{layerLabel(l)}</option>{/each}
			</select>
		</label>
		<label class="axis-control">
			<span>Prompt</span>
			<select bind:value={filterPrompt}>
				<option value="all">all</option>
				{#each promptLabels as l}<option value={l}>{PROMPTS[l] ? PROMPTS[l].slice(0, 30) : l}</option>{/each}
			</select>
		</label>
		<label class="axis-control">
			<span>Genre</span>
			<select bind:value={filterGenre}>
				<option value="all">all</option>
				<option value="narrative">narrative only</option>
				<option value="template">template only</option>
				{#each genreTypes as g}
					{#if g !== 'narrative'}
						<option value={g}>{g}</option>
					{/if}
				{/each}
			</select>
		</label>
		<ExportButton container={outerContainer} filename="passage_explorer" />
	</div>

	{#if error}
		<div class="status error">{error}</div>
	{/if}

	{#if loading}
		<div class="status">Loading passage metrics...</div>
	{:else}
		<!-- Chart + passage side by side -->
		<div class="content-area">
			<div class="chart-col">
				<div bind:this={chartDiv} class="chart-area"></div>
			</div>
			<div class="text-col">
				{#if selectedPoint}
					<div class="passage-detail">
						<div class="passage-header" style="color: {pointColor(selectedPoint)}">
							{selectedPoint.family} {layerLabel(selectedPoint.model)}
							<span class="passage-label">{selectedPoint.label}</span>
							{#if selectedPoint.genre_type && selectedPoint.genre_type !== 'narrative'}
								<span class="genre-badge">{selectedPoint.genre_type}</span>
							{/if}
						</div>
						<div class="passage-metrics">
							{#each METRICS as m}
								{@const z = getVal(selectedPoint, m.id)}
								{@const raw = getRaw(selectedPoint, m.id)}
								{#if !isNaN(raw)}
									<span class="metric" class:highlight={m.id === xAxis || m.id === yAxis}>
										<span class="metric-name">{m.label}</span>
										<span class="z-val" class:z-high={z > 1} class:z-low={z < -1}>{z >= 0 ? '+' : ''}{z.toFixed(1)}σ</span>
										<span class="raw-val">({raw.toFixed(3)})</span>
									</span>
								{/if}
							{/each}
						</div>
						<div class="passage-text">
							<span class="prompt-prefix">{getPrompt(selectedPoint.label)} </span>{#if tokenSurprisals && tokenSurprisals.length > 0}
								{@const vals = tokenSurprisals.map(([_, s]) => s)}
								{@const minS = Math.min(...vals)}
								{@const maxS = Math.max(...vals)}
								{#each tokenSurprisals as [tok, surp]}
									<span
										class="token"
										style="background: {surprisalColor(surp, minS, maxS)}"
										title="{tok.trim()}: {surp.toFixed(2)} bits"
									>{tok}</span>
								{/each}
							{:else if tokensLoading}
								<span class="tokens-loading">loading tokens...</span>
							{:else}
								{selectedPoint.psg}
							{/if}
						</div>
					</div>
				{:else}
					<div class="no-selection">
						Click a point to read the passage
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.explorer {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.custom-input {
		display: flex;
		gap: 8px;
	}
	.custom-input textarea {
		flex: 1;
		font-size: 12px;
		padding: 6px 8px;
		background: #111122;
		border: 1px solid #2a2a44;
		border-radius: 4px;
		color: #ccc;
		resize: vertical;
		font-family: inherit;
		line-height: 1.4;
	}
	.custom-input textarea::placeholder {
		color: #555;
	}
	.btn {
		padding: 7px 14px;
		border: 1px solid #4e79a7;
		border-radius: 6px;
		background: #2a3a5e;
		color: #e0e0e0;
		font-size: 13px;
		cursor: pointer;
		white-space: nowrap;
		align-self: flex-start;
	}
	.btn:hover:not(:disabled) { background: #344a70; }
	.btn:disabled { opacity: 0.4; cursor: not-allowed; }

	.controls {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
	}
	.axis-control {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		color: #888;
	}
	.axis-control select {
		background: #141428;
		border: 1px solid #2a2a44;
		color: #ccc;
		padding: 3px 6px;
		border-radius: 4px;
		font-size: 11px;
	}
	.status { text-align: center; color: #666; font-size: 13px; padding: 20px 0; }
	.status.error { color: #e15759; }

	.content-area {
		display: flex;
		gap: 16px;
		min-height: 500px;
	}
	.chart-col { flex: 1; min-width: 0; }
	.chart-area { position: relative; width: 100%; min-height: 500px; }
	.text-col {
		flex-shrink: 0;
		width: 320px;
		overflow-y: auto;
		max-height: 700px;
	}
	.no-selection {
		color: #555;
		font-size: 13px;
		padding: 20px;
		text-align: center;
	}
	.passage-detail {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}
	.passage-header {
		font-size: 13px;
		font-weight: 600;
	}
	.passage-label {
		color: #666;
		font-weight: 400;
		margin-left: 6px;
	}
	.genre-badge {
		background: #5a3a2a;
		color: #f0c090;
		padding: 1px 6px;
		border-radius: 3px;
		font-size: 10px;
		font-weight: 500;
		margin-left: 6px;
	}
	.passage-metrics {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.metric {
		display: flex;
		gap: 6px;
		font-size: 11px;
		color: #666;
		align-items: baseline;
	}
	.metric.highlight {
		color: #ccc;
	}
	.metric-name {
		min-width: 130px;
	}
	.z-val {
		font-family: 'SF Mono', monospace;
		min-width: 45px;
		text-align: right;
	}
	.z-high { color: #e15759; }
	.z-low { color: #4e79a7; }
	.raw-val {
		color: #444;
		font-size: 10px;
		font-family: 'SF Mono', monospace;
	}
	.passage-text {
		font-size: 13px;
		line-height: 1.6;
		color: #ccc;
		white-space: pre-wrap;
		background: #111122;
		padding: 12px;
		border-radius: 6px;
		border: 1px solid #1a1a2e;
		max-height: 250px;
		overflow-y: auto;
	}
	.prompt-prefix {
		color: #888;
		font-style: italic;
	}
	.token {
		border-radius: 2px;
		padding: 0 1px;
		cursor: help;
	}
	.tokens-loading {
		color: #555;
		font-style: italic;
	}
</style>
