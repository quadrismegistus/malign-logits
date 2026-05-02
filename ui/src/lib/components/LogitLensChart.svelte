<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import { api } from '$lib/api';
	import type { LogitLensRow } from '$lib/api';

	interface Props {
		prompt: string;
		analyzedPrompt?: string;
		onAnalyze?: () => Promise<void>;
	}

	let { prompt, analyzedPrompt = '', onAnalyze }: Props = $props();

	let allRows: LogitLensRow[] = $state([]);
	let wordSources: Record<string, string[]> = $state({});
	let loading = $state(false);
	let error = $state('');
	let loadedPrompt = $state('');
	let container: HTMLDivElement;

	let nWords = $state(8);
	let modelA = $state('base');
	let modelB = $state('dpo');
	let autoZoom = $state(true);
	let rankBy: 'output' | 'max' = $state('output');

	const MODEL_ORDER = ['base', 'sft', 'dpo', 'rlvr'];
	const COLORS = d3.schemeTableau10;

	let progressText = $state('');

	function availableModels(): string[] {
		if (!allRows.length) return MODEL_ORDER;
		return MODEL_ORDER.filter((m) => allRows.some((r) => r.model === m));
	}

	async function load() {
		if (!prompt.trim()) return;
		loading = true;
		error = '';
		progressText = 'Starting...';

		if (prompt.trim() !== analyzedPrompt && onAnalyze) {
			progressText = 'Analyzing prompt...';
			await onAnalyze();
		}

		const pollId = setInterval(async () => {
			try {
				const p = await api.progress();
				if (p.stage !== 'idle') {
					progressText = p.detail || p.stage;
				}
			} catch {}
		}, 800);
		try {
			const res = await api.logitLens(prompt.trim());
			allRows = res.rows;
			wordSources = res.word_sources;
			loadedPrompt = prompt.trim();

			const avail = availableModels();
			if (!avail.includes(modelA)) modelA = avail[0];
			if (!avail.includes(modelB)) modelB = avail[avail.length - 1];
			if (modelA === modelB && avail.length > 1) {
				modelB = avail[avail.length - 1];
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
			progressText = '';
			clearInterval(pollId);
		}
	}

	interface DiffPoint {
		layer: number;
		diff: number;
		probA: number;
		probB: number;
	}

	function computeDiffs(): Map<string, DiffPoint[]> {
		const probMapA = new Map<string, Map<number, number>>();
		const probMapB = new Map<string, Map<number, number>>();

		for (const r of allRows) {
			if (r.model === modelA) {
				if (!probMapA.has(r.word)) probMapA.set(r.word, new Map());
				probMapA.get(r.word)!.set(r.layer, r.probability);
			} else if (r.model === modelB) {
				if (!probMapB.has(r.word)) probMapB.set(r.word, new Map());
				probMapB.get(r.word)!.set(r.layer, r.probability);
			}
		}

		const maxLayer = d3.max(allRows, (r) => r.layer) ?? 32;
		const layerRange = d3.range(0, maxLayer + 1);

		const allWords = new Set([...probMapA.keys(), ...probMapB.keys()]);
		const wordDiffs = new Map<string, DiffPoint[]>();

		for (const word of allWords) {
			const mapA = probMapA.get(word);
			const mapB = probMapB.get(word);
			const points: DiffPoint[] = [];
			for (const layer of layerRange) {
				const pA = mapA?.get(layer) ?? 0;
				const pB = mapB?.get(layer) ?? 0;
				points.push({ layer, diff: pB - pA, probA: pA, probB: pB });
			}
			wordDiffs.set(word, points);
		}

		return wordDiffs;
	}

	function rankWords(wordDiffs: Map<string, DiffPoint[]>): string[] {
		const scores: [string, number][] = [];
		for (const [word, points] of wordDiffs) {
			let score: number;
			if (rankBy === 'output') {
				const last = points[points.length - 1];
				score = last ? Math.abs(last.diff) : 0;
			} else {
				score = d3.max(points, (p) => Math.abs(p.diff)) ?? 0;
			}
			scores.push([word, score]);
		}
		scores.sort((a, b) => b[1] - a[1]);
		return scores.slice(0, nWords).map(([w]) => w);
	}

	function draw() {
		if (!container || !allRows.length) return;
		container.innerHTML = '';

		const wordDiffs = computeDiffs();
		const topWords = rankWords(wordDiffs);
		if (!topWords.length) return;

		const wordColor = new Map<string, string>();
		topWords.forEach((w, i) => wordColor.set(w, COLORS[i % COLORS.length]));

		const selectedDiffs = new Map<string, DiffPoint[]>();
		let globalMin = 0;
		let globalMax = 0;
		for (const word of topWords) {
			const pts = wordDiffs.get(word);
			if (!pts) continue;
			selectedDiffs.set(word, pts);
			for (const p of pts) {
				if (p.diff < globalMin) globalMin = p.diff;
				if (p.diff > globalMax) globalMax = p.diff;
			}
		}

		const maxLayer = d3.max(allRows, (r) => r.layer) ?? 32;

		let xMin = 0;
		if (autoZoom) {
			for (const [, pts] of selectedDiffs) {
				for (const p of pts) {
					if (Math.abs(p.diff) > 0.005) {
						xMin = Math.max(0, p.layer - 2);
						break;
					}
				}
			}
		}

		const rect = container.getBoundingClientRect();
		const width = rect.width;
		const margin = { top: 24, right: 120, bottom: 44, left: 65 };
		const height = 420;
		const innerW = width - margin.left - margin.right;
		const innerH = height - margin.top - margin.bottom;

		const absMax = Math.max(Math.abs(globalMin), Math.abs(globalMax), 0.001);

		const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
		const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		const x = d3.scaleLinear().domain([xMin, maxLayer]).range([0, innerW]);
		const y = d3.scaleLinear().domain([-absMax * 1.15, absMax * 1.15]).range([innerH, 0]);

		g.append('g')
			.attr('transform', `translate(0,${innerH})`)
			.call(d3.axisBottom(x).ticks(8).tickFormat(d3.format('d')))
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '11px');

		g.append('g')
			.call(
				d3.axisLeft(y).ticks(6).tickFormat((d) => {
					const v = d as number;
					if (Math.abs(v) < 0.0005) return '0';
					return v > 0 ? `+${d3.format('.2%')(v)}` : d3.format('.2%')(v);
				})
			)
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '10px');

		g.selectAll('.domain, .tick line').style('stroke', '#333');

		g.append('line')
			.attr('x1', 0)
			.attr('x2', innerW)
			.attr('y1', y(0))
			.attr('y2', y(0))
			.attr('stroke', '#555')
			.attr('stroke-width', 1)
			.attr('stroke-dasharray', '4,4');

		svg
			.append('text')
			.attr('x', margin.left + innerW / 2)
			.attr('y', height - 6)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '11px')
			.text('Network layer');

		svg
			.append('text')
			.attr('transform', `translate(14, ${margin.top + innerH / 2}) rotate(-90)`)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '11px')
			.text(`Δ prob (${modelB.toUpperCase()} − ${modelA.toUpperCase()})`);

		g.append('text')
			.attr('x', 4)
			.attr('y', 6)
			.attr('fill', '#4e79a7')
			.attr('font-size', '10px')
			.attr('opacity', 0.5)
			.text(`↑ ${modelB.toUpperCase()} higher`);

		g.append('text')
			.attr('x', 4)
			.attr('y', innerH - 4)
			.attr('fill', '#e15759')
			.attr('font-size', '10px')
			.attr('opacity', 0.5)
			.text(`↓ ${modelA.toUpperCase()} higher`);

		const tooltip = d3
			.select(container)
			.append('div')
			.style('position', 'absolute')
			.style('pointer-events', 'none')
			.style('background', 'rgba(20,20,20,0.95)')
			.style('border', '1px solid #555')
			.style('padding', '6px 10px')
			.style('border-radius', '4px')
			.style('font-size', '11px')
			.style('color', '#ddd')
			.style('display', 'none')
			.style('z-index', '100');

		const line = d3
			.line<DiffPoint>()
			.x((d) => x(d.layer))
			.y((d) => y(d.diff))
			.curve(d3.curveMonotoneX);

		const area = d3
			.area<DiffPoint>()
			.x((d) => x(d.layer))
			.y0(y(0))
			.y1((d) => y(d.diff))
			.curve(d3.curveMonotoneX);

		for (const word of topWords) {
			const points = selectedDiffs.get(word);
			if (!points) continue;
			const color = wordColor.get(word)!;
			const visible = points.filter((p) => p.layer >= xMin);

			g.append('path').datum(visible).attr('d', area).attr('fill', color).attr('opacity', 0.06);

			const path = g
				.append('path')
				.datum(visible)
				.attr('d', line)
				.attr('fill', 'none')
				.attr('stroke', color)
				.attr('stroke-width', 2)
				.attr('opacity', 0.8);

			g.selectAll(null)
				.data(visible.filter((_, i) => i % 2 === 0))
				.enter()
				.append('circle')
				.attr('cx', (d) => x(d.layer))
				.attr('cy', (d) => y(d.diff))
				.attr('r', 2.5)
				.attr('fill', color)
				.attr('opacity', 0.7)
				.on('mouseenter', function (event, d) {
					path.attr('stroke-width', 4).attr('opacity', 1);
					const sign = d.diff >= 0 ? '+' : '';
					tooltip
						.style('display', 'block')
						.html(
							`<strong>${word}</strong><br>` +
								`layer ${d.layer}<br>` +
								`${modelA.toUpperCase()}: ${d.probA.toExponential(3)}<br>` +
								`${modelB.toUpperCase()}: ${d.probB.toExponential(3)}<br>` +
								`Δ: <strong>${sign}${(d.diff * 100).toFixed(2)}%</strong>`
						);
				})
				.on('mousemove', function (event) {
					const [mx, my] = d3.pointer(event, container);
					tooltip.style('left', mx + 14 + 'px').style('top', my - 10 + 'px');
				})
				.on('mouseleave', function () {
					path.attr('stroke-width', 2).attr('opacity', 0.8);
					tooltip.style('display', 'none');
				});

			const lastPt = visible[visible.length - 1];
			g.append('text')
				.attr('x', x(lastPt.layer) + 6)
				.attr('y', y(lastPt.diff))
				.attr('dy', '0.35em')
				.attr('fill', color)
				.attr('font-size', '11px')
				.attr('font-weight', '600')
				.text(word);
		}
	}

	$effect(() => {
		allRows;
		nWords;
		modelA;
		modelB;
		autoZoom;
		rankBy;
		draw();
	});
</script>

<div class="logit-lens">
	<div class="controls">
		<button class="btn" onclick={load} disabled={loading || !prompt.trim()}>
			{loading ? 'Computing...' : 'Run Logit Lens'}
		</button>
		<label class="compare-control">
			<select bind:value={modelA}>
				{#each availableModels() as m}
					<option value={m}>{m.toUpperCase()}</option>
				{/each}
			</select>
			<span>vs</span>
			<select bind:value={modelB}>
				{#each availableModels() as m}
					<option value={m}>{m.toUpperCase()}</option>
				{/each}
			</select>
		</label>
		<label class="slider-control">
			<span class="label-text">words</span>
			<input type="range" bind:value={nWords} min={1} max={20} />
			<span class="val">{nWords}</span>
		</label>
		<label class="compare-control">
			<span>rank by</span>
			<select bind:value={rankBy}>
				<option value="output">output layer</option>
				<option value="max">max across layers</option>
			</select>
		</label>
		<label class="check-control">
			<input type="checkbox" bind:checked={autoZoom} />
			<span>auto-zoom</span>
		</label>
	</div>
	{#if loadedPrompt && loadedPrompt !== prompt.trim()}
		<div class="stale">prompt changed — click Run to update</div>
	{/if}

	{#if loading}
		<div class="status">{progressText || 'Computing logit lens...'}</div>
	{:else if error}
		<div class="status error">{error}</div>
	{:else if !allRows.length}
		<div class="status">Click <strong>Run Logit Lens</strong> to project hidden states through the unembedding matrix at each network layer.</div>
	{/if}

	<div bind:this={container} class="chart-area"></div>
</div>

<style>
	.logit-lens {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 14px;
		flex-wrap: wrap;
	}

	.btn {
		padding: 7px 14px;
		border: 1px solid #4e79a7;
		border-radius: 6px;
		background: #2a3a5e;
		color: #e0e0e0;
		font-size: 13px;
		cursor: pointer;
		transition: all 0.15s;
		white-space: nowrap;
	}

	.btn:hover:not(:disabled) {
		background: #344a70;
	}

	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.slider-control {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		color: #888;
		white-space: nowrap;
	}

	.slider-control .label-text {
		text-align: right;
	}

	.slider-control input[type='range'] {
		width: 70px;
		accent-color: #4e79a7;
	}

	.slider-control .val {
		font-family: 'SF Mono', monospace;
		min-width: 18px;
		color: #aaa;
		font-size: 11px;
	}

	.compare-control {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		color: #888;
	}

	.compare-control select {
		background: #141428;
		border: 1px solid #2a2a44;
		color: #ccc;
		padding: 4px 8px;
		border-radius: 4px;
		font-size: 12px;
		font-family: 'SF Mono', monospace;
	}

	.compare-control select:focus {
		outline: none;
		border-color: #4e79a7;
	}

	.check-control {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 11px;
		color: #888;
		cursor: pointer;
	}

	.check-control input[type='checkbox'] {
		accent-color: #4e79a7;
	}

	.stale {
		font-size: 11px;
		color: #e2b340;
		font-style: italic;
	}

	.status {
		text-align: center;
		color: #666;
		font-size: 13px;
		padding: 20px 0;
	}

	.status.error {
		color: #e15759;
	}

	.chart-area {
		position: relative;
		width: 100%;
		min-height: 420px;
	}
</style>
