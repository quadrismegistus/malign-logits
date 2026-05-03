<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import { api } from '$lib/api';
	import type { LogitLensRow } from '$lib/api';
	import ExportButton from './ExportButton.svelte';

	interface Props {
		prompt: string;
		analyzedPrompt?: string;
		family?: string;
		onAnalyze?: () => Promise<void>;
	}

	let { prompt, analyzedPrompt = '', family = '', onAnalyze }: Props = $props();

	let allRows: LogitLensRow[] = $state([]);
	let wordSources: Record<string, string[]> = $state({});
	let loading = $state(false);
	let error = $state('');
	let loadedPrompt = $state('');
	let container: HTMLDivElement;

	let nOutput = $state(3);
	let nMax = $state(2);
	let modelA = $state('base');
	let modelB = $state('dpo');
	let autoZoom = $state(false);
	let compareAll = $state(true);
	let sharedScale = $state(true);

	const MODEL_ORDER = ['base', 'sft', 'dpo', 'rlvr'];
	const MODEL_LABELS: Record<string, string> = {
		base: 'BASE', sft: 'SFT', dpo: 'DPO', rlvr: 'RLVR'
	};
	const COLORS = d3.schemeTableau10;

	let progressText = $state('');

	function availableModels(): string[] {
		if (!allRows.length) return MODEL_ORDER;
		return MODEL_ORDER.filter((m) => allRows.some((r) => r.model === m));
	}

	function adjacentPairs(): [string, string][] {
		const avail = availableModels();
		const pairs: [string, string][] = [];
		for (let i = 0; i < avail.length - 1; i++) {
			pairs.push([avail[i], avail[i + 1]]);
		}
		return pairs;
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

	function computeDiffsFor(mA: string, mB: string): Map<string, DiffPoint[]> {
		const probMapA = new Map<string, Map<number, number>>();
		const probMapB = new Map<string, Map<number, number>>();

		for (const r of allRows) {
			if (r.model === mA) {
				if (!probMapA.has(r.word)) probMapA.set(r.word, new Map());
				probMapA.get(r.word)!.set(r.layer, r.probability);
			} else if (r.model === mB) {
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
		const outputScores: [string, number][] = [];
		const maxScores: [string, number][] = [];
		for (const [word, points] of wordDiffs) {
			const last = points[points.length - 1];
			outputScores.push([word, last ? Math.abs(last.diff) : 0]);
			maxScores.push([word, d3.max(points, (p) => Math.abs(p.diff)) ?? 0]);
		}
		outputScores.sort((a, b) => b[1] - a[1]);
		maxScores.sort((a, b) => b[1] - a[1]);
		const result: string[] = [];
		const seen = new Set<string>();
		for (const [w] of outputScores.slice(0, nOutput)) {
			if (!seen.has(w)) { result.push(w); seen.add(w); }
		}
		for (const [w] of maxScores.slice(0, nMax)) {
			if (!seen.has(w)) { result.push(w); seen.add(w); }
		}
		return result;
	}

	function drawPanel(
		parent: d3.Selection<SVGGElement, unknown, null, undefined>,
		tooltip: d3.Selection<HTMLDivElement, unknown, null, undefined>,
		mA: string, mB: string,
		topWords: string[],
		wordColor: Map<string, string>,
		panelW: number, panelH: number,
		sharedAbsMax: number | null = null,
	) {
		const wordDiffs = computeDiffsFor(mA, mB);
		const margin = { top: 28, right: 70, bottom: 44, left: 50 };
		const innerW = panelW - margin.left - margin.right;
		const innerH = panelH - margin.top - margin.bottom;

		const g = parent.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		const selectedDiffs = new Map<string, DiffPoint[]>();
		let localMin = 0, localMax = 0;
		for (const word of topWords) {
			const pts = wordDiffs.get(word);
			if (!pts) continue;
			selectedDiffs.set(word, pts);
			for (const p of pts) {
				if (p.diff < localMin) localMin = p.diff;
				if (p.diff > localMax) localMax = p.diff;
			}
		}

		const absMax = sharedAbsMax ?? Math.max(Math.abs(localMin), Math.abs(localMax), 0.001);
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

		const x = d3.scaleLinear().domain([xMin, maxLayer]).range([0, innerW]);
		const y = d3.scaleLinear().domain([-absMax * 1.15, absMax * 1.15]).range([innerH, 0]);

		g.append('g')
			.attr('transform', `translate(0,${innerH})`)
			.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('d')))
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '10px');

		g.append('g')
			.call(
				d3.axisLeft(y).ticks(5).tickFormat((d) => {
					const v = d as number;
					if (Math.abs(v) < 0.0005) return '0';
					return v > 0 ? `+${d3.format('.1%')(v)}` : d3.format('.1%')(v);
				})
			)
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '9px');

		g.selectAll('.domain, .tick line').style('stroke', '#333');

		g.append('line')
			.attr('x1', 0).attr('x2', innerW)
			.attr('y1', y(0)).attr('y2', y(0))
			.attr('stroke', '#555').attr('stroke-width', 1).attr('stroke-dasharray', '4,4');

		// Title
		parent.append('text')
			.attr('x', margin.left + innerW / 2)
			.attr('y', 16)
			.attr('text-anchor', 'middle')
			.attr('fill', '#ccc')
			.attr('font-size', '12px')
			.attr('font-weight', '600')
			.text(`${MODEL_LABELS[mA]} → ${MODEL_LABELS[mB]}`);

		// Axis labels
		parent.append('text')
			.attr('x', margin.left + innerW / 2)
			.attr('y', panelH - 4)
			.attr('text-anchor', 'middle')
			.attr('fill', '#666')
			.attr('font-size', '10px')
			.text('layer');

		g.append('text')
			.attr('x', 4).attr('y', 10)
			.attr('fill', '#4e79a7').attr('font-size', '9px').attr('opacity', 0.5)
			.text(`↑ ${MODEL_LABELS[mB]}`);

		g.append('text')
			.attr('x', 4).attr('y', innerH - 4)
			.attr('fill', '#e15759').attr('font-size', '9px').attr('opacity', 0.5)
			.text(`↓ ${MODEL_LABELS[mA]}`);

		const line = d3.line<DiffPoint>()
			.x((d) => x(d.layer))
			.y((d) => y(d.diff))
			.curve(d3.curveMonotoneX);

		const area = d3.area<DiffPoint>()
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

			const path = g.append('path')
				.datum(visible).attr('d', line)
				.attr('fill', 'none').attr('stroke', color)
				.attr('stroke-width', 2).attr('opacity', 0.8);

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
					tooltip.style('display', 'block').html(
						`<strong>${word}</strong><br>` +
						`layer ${d.layer}<br>` +
						`${MODEL_LABELS[mA]}: ${d.probA.toExponential(3)}<br>` +
						`${MODEL_LABELS[mB]}: ${d.probB.toExponential(3)}<br>` +
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
				.attr('x', x(lastPt.layer) + 5)
				.attr('y', y(lastPt.diff))
				.attr('dy', '0.35em')
				.attr('fill', color)
				.attr('font-size', '10px')
				.attr('font-weight', '600')
				.text(word);
		}
	}

	function draw() {
		if (!container || !allRows.length) return;
		container.innerHTML = '';

		const tooltip = d3.select(container)
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

		const hasTitle = !!(loadedPrompt || family);
		const titleH = hasTitle ? 36 : 0;

		function addTitle(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>, w: number) {
			if (!hasTitle) return;
			const titleParts = ['Logit Lens'];
			if (family) titleParts.push(`— ${family}`);
			svg.append('text')
				.attr('x', w / 2).attr('y', 14)
				.attr('text-anchor', 'middle')
				.attr('fill', '#ccc').attr('font-size', '13px').attr('font-weight', '600')
				.text(titleParts.join(' '));
			if (loadedPrompt) {
				svg.append('text')
					.attr('x', w / 2).attr('y', 30)
					.attr('text-anchor', 'middle')
					.attr('fill', '#777').attr('font-size', '11px').attr('font-style', 'italic')
					.text(`"${loadedPrompt}"`);
			}
		}

		if (compareAll) {
			const pairs = adjacentPairs();
			if (!pairs.length) return;

			// Collect top words across all pairs (union of output + max ranking)
			const outputScores = new Map<string, number>();
			const maxScores = new Map<string, number>();
			for (const [mA, mB] of pairs) {
				const diffs = computeDiffsFor(mA, mB);
				for (const [word, points] of diffs) {
					const last = points[points.length - 1];
					const outScore = last ? Math.abs(last.diff) : 0;
					const maxScore = d3.max(points, (p) => Math.abs(p.diff)) ?? 0;
					outputScores.set(word, Math.max(outputScores.get(word) ?? 0, outScore));
					maxScores.set(word, Math.max(maxScores.get(word) ?? 0, maxScore));
				}
			}
			const byOutput = [...outputScores.entries()].sort((a, b) => b[1] - a[1]);
			const byMax = [...maxScores.entries()].sort((a, b) => b[1] - a[1]);
			const topWords: string[] = [];
			const seen = new Set<string>();
			for (const [w] of byOutput.slice(0, nOutput)) {
				if (!seen.has(w)) { topWords.push(w); seen.add(w); }
			}
			for (const [w] of byMax.slice(0, nMax)) {
				if (!seen.has(w)) { topWords.push(w); seen.add(w); }
			}

			const wordColor = new Map<string, string>();
			topWords.forEach((w, i) => wordColor.set(w, COLORS[i % COLORS.length]));

			let globalAbsMax: number | null = null;
			if (sharedScale) {
				globalAbsMax = 0;
				for (const [mA, mB] of pairs) {
					const diffs = computeDiffsFor(mA, mB);
					for (const word of topWords) {
						const pts = diffs.get(word);
						if (!pts) continue;
						for (const p of pts) {
							globalAbsMax = Math.max(globalAbsMax!, Math.abs(p.diff));
						}
					}
				}
			}

			const rect = container.getBoundingClientRect();
			const totalW = rect.width;
			const panelW = Math.floor(totalW / pairs.length);
			const panelH = 420;

			const svg = d3.select(container).append('svg')
				.attr('width', totalW)
				.attr('height', panelH + titleH);

			addTitle(svg, totalW);

			pairs.forEach(([mA, mB], i) => {
				const panel = svg.append('g')
					.attr('transform', `translate(${i * panelW}, ${titleH})`) as d3.Selection<SVGGElement, unknown, null, undefined>;

				if (i > 0) {
					svg.append('line')
						.attr('x1', i * panelW).attr('x2', i * panelW)
						.attr('y1', titleH).attr('y2', panelH + titleH)
						.attr('stroke', '#1a1a2e').attr('stroke-width', 1);
				}

				drawPanel(panel, tooltip, mA, mB, topWords, wordColor, panelW, panelH, globalAbsMax);
			});
		} else {
			const wordDiffs = computeDiffsFor(modelA, modelB);
			const topWords = rankWords(wordDiffs);
			if (!topWords.length) return;

			const wordColor = new Map<string, string>();
			topWords.forEach((w, i) => wordColor.set(w, COLORS[i % COLORS.length]));

			const rect = container.getBoundingClientRect();
			const width = rect.width;
			const panelH = 420;

			const svg = d3.select(container).append('svg')
				.attr('width', width).attr('height', panelH + titleH);

			addTitle(svg, width);

			const panel = svg.append('g')
				.attr('transform', `translate(0, ${titleH})`) as d3.Selection<SVGGElement, unknown, null, undefined>;
			drawPanel(panel, tooltip, modelA, modelB, topWords, wordColor, width, panelH);
		}
	}

	$effect(() => {
		allRows;
		nOutput;
		nMax;
		modelA;
		modelB;
		autoZoom;
		compareAll;
		sharedScale;
		draw();
	});
</script>

<div class="logit-lens">
	<div class="controls">
		<button class="btn" onclick={load} disabled={loading || !prompt.trim()}>
			{loading ? 'Computing...' : 'Run Logit Lens'}
		</button>
		{#if !compareAll}
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
		{/if}
		<label class="slider-control">
			<span class="label-text">by output</span>
			<input type="range" bind:value={nOutput} min={0} max={15} />
			<span class="val">{nOutput}</span>
		</label>
		<label class="slider-control">
			<span class="label-text">by max</span>
			<input type="range" bind:value={nMax} min={0} max={15} />
			<span class="val">{nMax}</span>
		</label>
		<label class="check-control">
			<input type="checkbox" bind:checked={autoZoom} />
			<span>auto-zoom</span>
		</label>
		<label class="check-control">
			<input type="checkbox" bind:checked={compareAll} />
			<span>compare all</span>
		</label>
		{#if compareAll}
			<label class="check-control">
				<input type="checkbox" bind:checked={sharedScale} />
				<span>shared y</span>
			</label>
		{/if}
		<ExportButton {container} filename="logit_lens.png" />
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
