<script lang="ts">
	import * as d3 from 'd3';
	import { api } from '$lib/api';
	import type { Generation } from '$lib/api';

	interface Props {
		prompt: string;
		analyzedPrompt?: string;
		onAnalyze?: () => Promise<void>;
	}

	let { prompt, analyzedPrompt = '', onAnalyze }: Props = $props();

	let generations: Generation[] = $state([]);
	let conceptAxes: string[] = $state([]);
	let loading = $state(false);
	let error = $state('');
	let progressText = $state('');
	let container: HTMLDivElement;
	let selectedGen: Generation | null = $state(null);

	let nGens = $state(5);
	let maxTokens = $state(100);
	let xAxis = $state('violent');
	let yAxis = $state('sexual');

	const MODEL_COLORS: Record<string, string> = {
		base: '#e15759',
		sft: '#f28e2b',
		dpo: '#4e79a7',
		rlvr: '#59a14f'
	};
	const MODEL_ORDER = ['base', 'sft', 'dpo', 'rlvr'];

	async function generate() {
		if (!prompt.trim()) return;
		loading = true;
		error = '';
		progressText = 'Starting...';
		selectedGen = null;

		if (prompt.trim() !== analyzedPrompt && onAnalyze) {
			progressText = 'Analyzing prompt...';
			await onAnalyze();
		}

		const pollId = setInterval(async () => {
			try {
				const p = await api.progress();
				if (p.stage !== 'idle') {
					progressText = p.detail || p.stage;
					if (p.total > 0) progressText += ` (${p.step}/${p.total})`;
				}
			} catch {}
		}, 800);

		try {
			const res = await api.generate(prompt.trim(), nGens, maxTokens);
			generations = res.generations;
			conceptAxes = res.concept_axes;
			if (conceptAxes.length >= 2) {
				xAxis = conceptAxes[0];
				yAxis = conceptAxes[1];
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
			progressText = '';
			clearInterval(pollId);
		}
	}

	function draw() {
		if (!container || !generations.length) return;
		container.innerHTML = '';

		const rect = container.getBoundingClientRect();
		const width = Math.min(rect.width, 500);
		const height = 400;
		const margin = { top: 20, right: 20, bottom: 44, left: 50 };
		const innerW = width - margin.left - margin.right;
		const innerH = height - margin.top - margin.bottom;

		const xVals = generations.map((g) => (g as Record<string, unknown>)[xAxis] as number);
		const yVals = generations.map((g) => (g as Record<string, unknown>)[yAxis] as number);

		const xExt = d3.extent(xVals) as [number, number];
		const yExt = d3.extent(yVals) as [number, number];
		const xPad = (xExt[1] - xExt[0]) * 0.15 || 0.1;
		const yPad = (yExt[1] - yExt[0]) * 0.15 || 0.1;

		const x = d3.scaleLinear().domain([xExt[0] - xPad, xExt[1] + xPad]).range([0, innerW]);
		const y = d3.scaleLinear().domain([yExt[0] - yPad, yExt[1] + yPad]).range([innerH, 0]);

		const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
		const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		g.append('g')
			.attr('transform', `translate(0,${innerH})`)
			.call(d3.axisBottom(x).ticks(6).tickFormat(d3.format('.2f')))
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '10px');

		g.append('g')
			.call(d3.axisLeft(y).ticks(6).tickFormat(d3.format('.2f')))
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '10px');

		g.selectAll('.domain, .tick line').style('stroke', '#333');

		g.append('line')
			.attr('x1', x(0)).attr('x2', x(0))
			.attr('y1', 0).attr('y2', innerH)
			.attr('stroke', '#333').attr('stroke-dasharray', '3,3');
		g.append('line')
			.attr('x1', 0).attr('x2', innerW)
			.attr('y1', y(0)).attr('y2', y(0))
			.attr('stroke', '#333').attr('stroke-dasharray', '3,3');

		svg.append('text')
			.attr('x', margin.left + innerW / 2)
			.attr('y', height - 6)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '11px')
			.text(`← less ${xAxis} · more ${xAxis} →`);

		svg.append('text')
			.attr('transform', `translate(13, ${margin.top + innerH / 2}) rotate(-90)`)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '11px')
			.text(`← less ${yAxis} · more ${yAxis} →`);

		const tooltip = d3
			.select(container)
			.append('div')
			.style('position', 'absolute')
			.style('pointer-events', 'none')
			.style('background', 'rgba(20,20,20,0.95)')
			.style('border', '1px solid #555')
			.style('padding', '8px 10px')
			.style('border-radius', '4px')
			.style('font-size', '11px')
			.style('color', '#ddd')
			.style('display', 'none')
			.style('z-index', '100')
			.style('max-width', '350px');

		for (const gen of generations) {
			const gx = (gen as Record<string, unknown>)[xAxis] as number;
			const gy = (gen as Record<string, unknown>)[yAxis] as number;
			const color = MODEL_COLORS[gen.model] ?? '#888';

			g.append('circle')
				.attr('cx', x(gx))
				.attr('cy', y(gy))
				.attr('r', selectedGen === gen ? 8 : 5)
				.attr('fill', color)
				.attr('opacity', selectedGen && selectedGen !== gen ? 0.3 : 0.8)
				.attr('stroke', selectedGen === gen ? '#fff' : 'none')
				.attr('stroke-width', 2)
				.style('cursor', 'pointer')
				.on('mouseenter', function (event) {
					d3.select(this).attr('r', 7);
					const preview = gen.text.slice(0, 120).replace(/\n/g, ' ');
					tooltip
						.style('display', 'block')
						.html(
							`<strong>${gen.model.toUpperCase()}</strong> #${gen.gen_id + 1}<br>` +
								`<span style="color:#999">${preview}...</span>`
						);
				})
				.on('mousemove', function (event) {
					const [mx, my] = d3.pointer(event, container);
					tooltip.style('left', mx + 14 + 'px').style('top', my - 10 + 'px');
				})
				.on('mouseleave', function () {
					d3.select(this).attr('r', selectedGen === gen ? 8 : 5);
					tooltip.style('display', 'none');
				})
				.on('click', function () {
					selectedGen = selectedGen === gen ? null : gen;
					draw();
				});
		}

		const models = [...new Set(generations.map((g) => g.model))];
		const ordered = MODEL_ORDER.filter((m) => models.includes(m));
		const legend = svg.append('g').attr('transform', `translate(${margin.left + 8}, ${margin.top + 8})`);
		ordered.forEach((m, i) => {
			const row = legend.append('g').attr('transform', `translate(0, ${i * 18})`);
			row.append('circle').attr('cx', 5).attr('cy', 5).attr('r', 4).attr('fill', MODEL_COLORS[m] ?? '#888');
			row.append('text').attr('x', 14).attr('y', 9).attr('fill', '#aaa').attr('font-size', '11px').text(m.toUpperCase());
		});
	}

	function textsByModel(): Record<string, Generation[]> {
		const result: Record<string, Generation[]> = {};
		for (const g of generations) {
			if (!result[g.model]) result[g.model] = [];
			result[g.model].push(g);
		}
		return result;
	}

	$effect(() => {
		generations;
		xAxis;
		yAxis;
		selectedGen;
		draw();
	});
</script>

<div class="generation">
	<div class="controls">
		<button class="btn" onclick={generate} disabled={loading || !prompt.trim()}>
			{loading ? 'Generating...' : 'Generate'}
		</button>
		<label class="slider-control">
			<span>n</span>
			<input type="range" bind:value={nGens} min={1} max={20} />
			<span class="val">{nGens}</span>
		</label>
		<label class="slider-control">
			<span>tokens</span>
			<input type="range" bind:value={maxTokens} min={25} max={200} step={25} />
			<span class="val">{maxTokens}</span>
		</label>
		{#if conceptAxes.length > 0}
			<label class="axis-control">
				<span>x</span>
				<select bind:value={xAxis}>
					{#each conceptAxes as ax}
						<option value={ax}>{ax}</option>
					{/each}
					<option value="pca_x">PCA 1</option>
					<option value="pca_y">PCA 2</option>
				</select>
			</label>
			<label class="axis-control">
				<span>y</span>
				<select bind:value={yAxis}>
					{#each conceptAxes as ax}
						<option value={ax}>{ax}</option>
					{/each}
					<option value="pca_x">PCA 1</option>
					<option value="pca_y">PCA 2</option>
				</select>
			</label>
		{/if}
	</div>

	{#if loading}
		<div class="status">{progressText || 'Generating...'}</div>
	{:else if error}
		<div class="status error">{error}</div>
	{:else if !generations.length}
		<div class="status">Click <strong>Generate</strong> to sample completions from each model layer and project onto concept axes.</div>
	{/if}

	{#if generations.length > 0}
		<div class="content-area">
			<div class="chart-col">
				<div bind:this={container} class="chart-area"></div>
			</div>
			<div class="text-col">
				{#if selectedGen}
					<div class="gen-detail">
						<div class="gen-header" style="color: {MODEL_COLORS[selectedGen.model] ?? '#888'}">
							{selectedGen.model.toUpperCase()} #{selectedGen.gen_id + 1}
						</div>
						<div class="gen-text">{selectedGen.text}</div>
					</div>
				{:else}
					{#each MODEL_ORDER.filter((m) => textsByModel()[m]) as model}
						<div class="model-group">
							<div class="model-label" style="color: {MODEL_COLORS[model] ?? '#888'}">{model.toUpperCase()}</div>
							{#each textsByModel()[model] as gen}
								<button
									class="gen-item"
									onclick={() => { selectedGen = gen; draw(); }}
								>
									<span class="gen-num">#{gen.gen_id + 1}</span>
									{gen.text.slice(0, 80).replace(/\n/g, ' ')}...
								</button>
							{/each}
						</div>
					{/each}
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.generation {
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

	.slider-control input[type='range'] {
		width: 60px;
		accent-color: #4e79a7;
	}

	.slider-control .val {
		font-family: 'SF Mono', monospace;
		min-width: 22px;
		color: #aaa;
		font-size: 11px;
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

	.status {
		text-align: center;
		color: #666;
		font-size: 13px;
		padding: 20px 0;
	}

	.status.error {
		color: #e15759;
	}

	.content-area {
		display: flex;
		gap: 16px;
		min-height: 400px;
	}

	.chart-col {
		flex-shrink: 0;
	}

	.chart-area {
		position: relative;
		width: 500px;
		min-height: 400px;
	}

	.text-col {
		flex: 1;
		overflow-y: auto;
		max-height: 500px;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.model-group {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.model-label {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 2px;
	}

	.gen-item {
		background: none;
		border: none;
		padding: 4px 8px;
		text-align: left;
		color: #999;
		font-size: 11px;
		cursor: pointer;
		border-radius: 4px;
		line-height: 1.4;
		transition: background 0.1s;
	}

	.gen-item:hover {
		background: rgba(255, 255, 255, 0.05);
		color: #ccc;
	}

	.gen-num {
		color: #666;
		font-family: 'SF Mono', monospace;
		margin-right: 4px;
	}

	.gen-detail {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.gen-header {
		font-size: 13px;
		font-weight: 600;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.gen-text {
		font-size: 13px;
		line-height: 1.6;
		color: #ccc;
		white-space: pre-wrap;
		background: #111122;
		padding: 12px;
		border-radius: 6px;
		border: 1px solid #1a1a2e;
	}
</style>
