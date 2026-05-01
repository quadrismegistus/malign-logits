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

	let nDeclining = $state(1);
	let nRising = $state(1);
	let nTopPerLayer = $state(1);
	let minLayers = $state(12);

	const MODEL_ORDER = ['base', 'sft', 'dpo', 'rlvr'];
	const SOURCE_COLORS: Record<string, string> = {
		declining: '#e15759',
		rising: '#4e79a7',
		top_base: '#76b7b2',
		top_sft: '#f28e2b',
		top_dpo: '#b07aa1',
		top_rlvr: '#59a14f'
	};

	let progressText = $state('');

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
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
			progressText = '';
			clearInterval(pollId);
		}
	}

	function getVisibleTracked(): { word: string; source: string; rank: number }[] {
		if (!wordSources || !Object.keys(wordSources).length) return [];
		const buckets: Record<string, string[]> = {
			declining: [],
			rising: [],
			top_base: [],
			top_sft: [],
			top_dpo: [],
			top_rlvr: []
		};
		for (const [word, sources] of Object.entries(wordSources)) {
			for (const s of sources) {
				if (s in buckets) buckets[s].push(word);
			}
		}

		const limits: Record<string, number> = {
			declining: nDeclining,
			rising: nRising,
			top_base: nTopPerLayer,
			top_sft: nTopPerLayer,
			top_dpo: nTopPerLayer,
			top_rlvr: nTopPerLayer
		};

		const result: { word: string; source: string; rank: number }[] = [];
		const seen = new Set<string>();
		for (const [source, words] of Object.entries(buckets)) {
			const limit = limits[source] ?? 0;
			for (let i = 0; i < Math.min(words.length, limit); i++) {
				if (!seen.has(words[i])) {
					result.push({ word: words[i], source, rank: i });
					seen.add(words[i]);
				}
			}
		}
		return result;
	}

	function draw() {
		if (!container || !allRows.length || !wordSources) return;
		container.innerHTML = '';

		const tracked = getVisibleTracked();
		const trackedWords = new Set(tracked.map((t) => t.word));
		const wordPrimarySource = new Map(tracked.map((t) => [t.word, t.source]));

		const topkRows = allRows.filter((r) => r.source === 'top_k');
		const wordLayerCounts = d3.rollup(
			topkRows,
			(v) => new Set(v.map((r) => r.layer)).size,
			(r) => r.word
		);
		const frequentTopk = new Set(
			[...wordLayerCounts.entries()]
				.filter(([, count]) => count >= minLayers)
				.map(([word]) => word)
		);

		const plotWords = new Set([...trackedWords, ...frequentTopk]);
		const filtered = allRows.filter((r) => plotWords.has(r.word));
		if (!filtered.length) return;

		const allModels = [...new Set(allRows.map((r) => r.model))];
		const orderedModels = MODEL_ORDER.filter((m) => allModels.includes(m));
		if (!orderedModels.length) return;

		const trackedList = tracked.map((t) => t.word);
		const topkList = [...frequentTopk].filter((w) => !trackedWords.has(w));

		const wordColor = new Map<string, string>();
		for (const t of tracked) {
			if (!wordColor.has(t.word)) {
				wordColor.set(t.word, SOURCE_COLORS[t.source] ?? '#888888');
			}
		}
		const TOPK_PALETTE = ['#666666', '#777777', '#888888', '#555555', '#999999'];
		topkList.forEach((w, i) => {
			wordColor.set(w, TOPK_PALETTE[i % TOPK_PALETTE.length]);
		});

		const nModels = orderedModels.length;
		const rect = container.getBoundingClientRect();
		const totalWidth = rect.width;
		const panelWidth = Math.floor(totalWidth / nModels);
		const margin = { top: 32, right: 60, bottom: 36, left: 50 };
		const innerW = panelWidth - margin.left - margin.right;
		const height = 480;
		const innerH = height - margin.top - margin.bottom;
		const maxLayer = d3.max(filtered, (r) => r.layer) ?? 32;

		const svg = d3.select(container).append('svg').attr('width', totalWidth).attr('height', height);

		const tooltip = d3
			.select(container)
			.append('div')
			.attr('class', 'll-tooltip')
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

		for (let mi = 0; mi < nModels; mi++) {
			const model = orderedModels[mi];
			const modelData = filtered.filter(
				(r) => (r.model) === model
			);

			const g = svg
				.append('g')
				.attr('transform', `translate(${mi * panelWidth + margin.left},${margin.top})`);

			g.append('text')
				.attr('x', innerW / 2)
				.attr('y', -12)
				.attr('text-anchor', 'middle')
				.attr('fill', '#ccc')
				.attr('font-size', '13px')
				.attr('font-weight', '600')
				.text(model.toUpperCase());

			const x = d3.scaleLinear().domain([0, maxLayer]).range([0, innerW]);
			const allProbs = modelData.map((r) => r.probability).filter((p) => p > 0);
			const yMin = d3.min(allProbs) ?? 1e-8;
			const yMax = d3.max(allProbs) ?? 1;
			const y = d3
				.scaleLog()
				.domain([Math.max(yMin * 0.3, 1e-9), Math.min(yMax * 3, 1)])
				.range([innerH, 0])
				.clamp(true);

			g.append('g')
				.attr('transform', `translate(0,${innerH})`)
				.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('d')))
				.selectAll('text')
				.style('fill', '#aaa')
				.style('font-size', '10px');

			if (mi === 0) {
				g.append('g')
					.call(
						d3
							.axisLeft(y)
							.ticks(5)
							.tickFormat((d) => d3.format('.0e')(d as number))
					)
					.selectAll('text')
					.style('fill', '#aaa')
					.style('font-size', '10px');
			}

			g.selectAll('.domain, .tick line').style('stroke', '#333');

			if (mi === Math.floor(nModels / 2)) {
				svg
					.append('text')
					.attr('x', mi * panelWidth + margin.left + innerW / 2)
					.attr('y', height - 4)
					.attr('text-anchor', 'middle')
					.attr('fill', '#888')
					.attr('font-size', '11px')
					.text('Network layer');
			}

			const byWord = d3.group(modelData, (r) => r.word);
			const line = d3
				.line<LogitLensRow>()
				.x((d) => x(d.layer))
				.y((d) => y(Math.max(d.probability, 1e-9)))
				.curve(d3.curveMonotoneX);

			for (const [word, wordRows] of byWord) {
				const sorted = wordRows.sort((a, b) => a.layer - b.layer);
				const color = wordColor.get(word) ?? '#666666';
				const isTracked = trackedWords.has(word);

				const path = g
					.append('path')
					.datum(sorted)
					.attr('d', line)
					.attr('fill', 'none')
					.attr('stroke', color)
					.attr('stroke-width', isTracked ? 2 : 0.8)
					.attr('stroke-dasharray', isTracked ? 'none' : '3,3')
					.attr('opacity', isTracked ? 0.85 : 0.3);

				g.selectAll(null)
					.data(sorted)
					.enter()
					.append('circle')
					.attr('cx', (d) => x(d.layer))
					.attr('cy', (d) => y(Math.max(d.probability, 1e-9)))
					.attr('r', isTracked ? 2.5 : 1)
					.attr('fill', color)
					.attr('opacity', isTracked ? 0.85 : 0.25)
					.on('mouseenter', function (event, d) {
						path.attr('stroke-width', isTracked ? 3.5 : 2).attr('opacity', 1);
						const src = wordPrimarySource.get(d.word) ?? 'top-k';
						tooltip
							.style('display', 'block')
							.html(
								`<strong>${d.word}</strong> <span style="color:${color}">${src}</span><br>layer ${d.layer} &middot; p = ${d.probability.toExponential(3)}`
							);
					})
					.on('mousemove', function (event) {
						const [mx, my] = d3.pointer(event, container);
						tooltip.style('left', mx + 14 + 'px').style('top', my - 10 + 'px');
					})
					.on('mouseleave', function () {
						path
							.attr('stroke-width', isTracked ? 2 : 0.8)
							.attr('opacity', isTracked ? 0.85 : 0.3);
						tooltip.style('display', 'none');
					});

				const peak = sorted.reduce((a, b) => (b.probability > a.probability ? b : a));
				if (isTracked || peak.probability > 0.02) {
					g.append('text')
						.attr('x', x(peak.layer) + 4)
						.attr('y', y(Math.max(peak.probability, 1e-9)) - 5)
						.attr('fill', color)
						.attr('font-size', isTracked ? '10px' : '8px')
						.attr('font-weight', isTracked ? '600' : '400')
						.attr('opacity', isTracked ? 0.9 : 0.6)
						.text(word);
				}
			}
		}

		// Legend
		const legendG = svg.append('g').attr('transform', `translate(12, ${height - 18})`);
		const legendItems = [
			{ label: 'declining', color: SOURCE_COLORS.declining, dash: false },
			{ label: 'rising', color: SOURCE_COLORS.rising, dash: false },
			{ label: 'top (base)', color: SOURCE_COLORS.top_base, dash: false },
			{ label: 'top (sft)', color: SOURCE_COLORS.top_sft, dash: false },
			{ label: 'top (dpo)', color: SOURCE_COLORS.top_dpo, dash: false },
			{ label: 'internal top-k', color: '#777777', dash: true }
		];
		let lx = 0;
		for (const item of legendItems) {
			const g = legendG.append('g').attr('transform', `translate(${lx}, 0)`);
			g.append('line')
				.attr('x1', 0)
				.attr('x2', 14)
				.attr('y1', 0)
				.attr('y2', 0)
				.attr('stroke', item.color)
				.attr('stroke-width', item.dash ? 1 : 2)
				.attr('stroke-dasharray', item.dash ? '3,3' : 'none');
			const text = g
				.append('text')
				.attr('x', 18)
				.attr('y', 4)
				.attr('fill', '#888')
				.attr('font-size', '10px')
				.text(item.label);
			lx += 18 + (text.node()?.getComputedTextLength() ?? 60) + 16;
		}
	}

	$effect(() => {
		allRows;
		nDeclining;
		nRising;
		nTopPerLayer;
		minLayers;
		draw();
	});
</script>

<div class="logit-lens">
	<div class="controls">
		<button class="btn" onclick={load} disabled={loading || !prompt.trim()}>
			{loading ? 'Computing...' : 'Run Logit Lens'}
		</button>
		<label class="slider-control">
			<span class="label-text" style="color: {SOURCE_COLORS.declining}">declining</span>
			<input type="range" bind:value={nDeclining} min={0} max={15} />
			<span class="val">{nDeclining}</span>
		</label>
		<label class="slider-control">
			<span class="label-text" style="color: {SOURCE_COLORS.rising}">rising</span>
			<input type="range" bind:value={nRising} min={0} max={15} />
			<span class="val">{nRising}</span>
		</label>
		<label class="slider-control">
			<span class="label-text">top/layer</span>
			<input type="range" bind:value={nTopPerLayer} min={0} max={15} />
			<span class="val">{nTopPerLayer}</span>
		</label>
		<label class="slider-control">
			<span class="label-text" style="color: #777">top-k depth</span>
			<input type="range" bind:value={minLayers} min={1} max={25} />
			<span class="val">{minLayers}</span>
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
		gap: 12px;
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
		min-width: 50px;
		text-align: right;
	}

	.slider-control input[type='range'] {
		width: 60px;
		accent-color: #4e79a7;
	}

	.slider-control .val {
		font-family: 'SF Mono', monospace;
		min-width: 18px;
		color: #aaa;
		font-size: 11px;
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
		min-height: 480px;
	}
</style>
