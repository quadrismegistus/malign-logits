<script lang="ts">
	import * as d3 from 'd3';
	import { api } from '$lib/api';
	import type { ContradictionResult } from '$lib/api';
	import ExportButton from './ExportButton.svelte';

	interface Props {
		family?: string;
	}

	let { family = '' }: Props = $props();

	let results: ContradictionResult[] = $state([]);
	let loading = $state(false);
	let error = $state('');
	let progressText = $state('');
	let selectedPair = $state('');
	let container: HTMLDivElement;

	const MODEL_COLORS: Record<string, string> = {
		base: '#e15759',
		sft: '#f28e2b',
		dpo: '#4e79a7',
		rlvr: '#59a14f'
	};
	const MODEL_ORDER = ['base', 'sft', 'dpo', 'rlvr'];

	async function run() {
		loading = true;
		error = '';
		progressText = 'Starting...';
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
			const res = await api.contradiction();
			results = res.results;
			if (results.length && !selectedPair) {
				selectedPair = results[0].pair;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
			progressText = '';
			clearInterval(pollId);
		}
	}

	function pairs(): string[] {
		return [...new Set(results.map((r) => r.pair))];
	}

	function pairResults(): ContradictionResult[] {
		return results.filter((r) => r.pair === selectedPair);
	}

	function draw() {
		if (!container || !results.length || !selectedPair) return;
		container.innerHTML = '';

		const data = pairResults();
		if (!data.length) return;

		const models = MODEL_ORDER.filter((m) => data.some((d) => d.model === m));

		const hasTitle = !!(family || selectedPair);
		const rect = container.getBoundingClientRect();
		const width = Math.min(rect.width, 600);
		const height = hasTitle ? 340 : 300;
		const margin = { top: hasTitle ? 52 : 30, right: 20, bottom: 50, left: 50 };
		const innerW = width - margin.left - margin.right;
		const innerH = height - margin.top - margin.bottom;

		const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);

		if (hasTitle) {
			const titleParts = ['Contradiction'];
			if (family) titleParts.push(`— ${family}`);
			svg.append('text')
				.attr('x', width / 2).attr('y', 16)
				.attr('text-anchor', 'middle')
				.attr('fill', '#ccc').attr('font-size', '13px').attr('font-weight', '600')
				.text(titleParts.join(' '));
			if (selectedPair) {
				svg.append('text')
					.attr('x', width / 2).attr('y', 32)
					.attr('text-anchor', 'middle')
					.attr('fill', '#777').attr('font-size', '11px').attr('font-style', 'italic')
					.text(selectedPair);
			}
		}
		const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		const x = d3.scaleBand().domain(models).range([0, innerW]).padding(0.3);
		const allVals = data.flatMap((d) => [d.superposition, d.resolution]);
		const yMax = d3.max(allVals)! * 1.2;
		const y = d3.scaleLinear().domain([0, yMax]).range([innerH, 0]);

		g.append('g')
			.attr('transform', `translate(0,${innerH})`)
			.call(d3.axisBottom(x).tickFormat((d) => d.toUpperCase()))
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '12px');

		g.append('g')
			.call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.4f')))
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '10px');

		g.selectAll('.domain, .tick line').style('stroke', '#333');

		const barW = x.bandwidth() / 2 - 2;

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
			.style('max-width', '300px');

		for (const d of data) {
			const bx = x(d.model)!;
			const color = MODEL_COLORS[d.model] ?? '#888';

			g.append('rect')
				.attr('x', bx)
				.attr('y', y(d.superposition))
				.attr('width', barW)
				.attr('height', innerH - y(d.superposition))
				.attr('fill', '#4e79a7')
				.attr('opacity', 0.7)
				.on('mouseenter', function (event) {
					tooltip.style('display', 'block').html(
						`<strong>${d.model.toUpperCase()}</strong> superposition<br>` +
						`JS(AB, mean) = ${d.superposition.toFixed(5)}<br>` +
						`<span style="color:#999">Low = treats A+B additively</span>`
					);
				})
				.on('mousemove', function (event) {
					const [mx, my] = d3.pointer(event, container);
					tooltip.style('left', mx + 14 + 'px').style('top', my - 10 + 'px');
				})
				.on('mouseleave', () => tooltip.style('display', 'none'));

			g.append('rect')
				.attr('x', bx + barW + 4)
				.attr('y', y(d.resolution))
				.attr('width', barW)
				.attr('height', innerH - y(d.resolution))
				.attr('fill', '#e15759')
				.attr('opacity', 0.7)
				.on('mouseenter', function (event) {
					tooltip.style('display', 'block').html(
						`<strong>${d.model.toUpperCase()}</strong> resolution<br>` +
						`min JS(AB, A/B) = ${d.resolution.toFixed(5)}<br>` +
						`<span style="color:#999">Low = resolves toward one pole</span>`
					);
				})
				.on('mousemove', function (event) {
					const [mx, my] = d3.pointer(event, container);
					tooltip.style('left', mx + 14 + 'px').style('top', my - 10 + 'px');
				})
				.on('mouseleave', () => tooltip.style('display', 'none'));

			g.append('text')
				.attr('x', bx + x.bandwidth() / 2)
				.attr('y', y(Math.max(d.superposition, d.resolution)) - 6)
				.attr('text-anchor', 'middle')
				.attr('fill', '#aaa')
				.attr('font-size', '10px')
				.attr('font-family', 'SF Mono, monospace')
				.text(d.ratio < 1 ? `${d.ratio.toFixed(2)}×` : `${d.ratio.toFixed(1)}×`);
		}

		svg.append('text')
			.attr('x', margin.left + innerW / 2)
			.attr('y', height - 6)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '11px')
			.text('Model layer');

		svg.append('text')
			.attr('transform', `translate(14, ${margin.top + innerH / 2}) rotate(-90)`)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '11px')
			.text('JS divergence');

		const legend = svg.append('g').attr('transform', `translate(${margin.left + innerW - 140}, ${margin.top + 4})`);
		[
			{ label: 'superposition (AB ≈ mean)', color: '#4e79a7' },
			{ label: 'resolution (AB ≈ A or B)', color: '#e15759' },
		].forEach((item, i) => {
			const row = legend.append('g').attr('transform', `translate(0, ${i * 16})`);
			row.append('rect').attr('width', 10).attr('height', 10).attr('fill', item.color).attr('opacity', 0.7).attr('rx', 2);
			row.append('text').attr('x', 14).attr('y', 9).attr('fill', '#aaa').attr('font-size', '10px').text(item.label);
		});
	}

	$effect(() => {
		results;
		selectedPair;
		draw();
	});
</script>

<div class="contradiction">
	<div class="controls">
		<button class="btn" onclick={run} disabled={loading}>
			{loading ? 'Computing...' : 'Run Contradiction Analysis'}
		</button>
		{#if pairs().length > 0}
			<label class="pair-control">
				<span>pair</span>
				<select bind:value={selectedPair}>
					{#each pairs() as p}
						<option value={p}>{p}</option>
					{/each}
				</select>
			</label>
		{/if}
		<ExportButton {container} filename="contradiction.png" />
	</div>

	{#if loading}
		<div class="status">{progressText || 'Computing...'}</div>
	{:else if error}
		<div class="status error">{error}</div>
	{:else if !results.length}
		<div class="status">
			Tests whether the base model tolerates contradiction (Freud's primary process has no negation).
			Compares the combined prompt "She loved and hated him" against the average of "She loved him" and "She hated him".
			<strong>Low superposition</strong> = additive (primary process). <strong>Low resolution</strong> = picks a winner (secondary process).
		</div>
	{/if}

	{#if results.length > 0}
		<div class="content-area">
			<div bind:this={container} class="chart-area"></div>

			{#if pairResults().length > 0}
				{@const pr = pairResults()[0]}
				<div class="prompts">
					<div class="prompt-row"><span class="prompt-label">A:</span> {pr.prompt_a}</div>
					<div class="prompt-row"><span class="prompt-label">B:</span> {pr.prompt_b}</div>
					<div class="prompt-row"><span class="prompt-label">A+B:</span> {pr.prompt_ab}</div>
				</div>

				<div class="words-section">
					<div class="words-label">Most contested words (largest |prob_A - prob_B|):</div>
					<table class="words-table">
						<thead>
							<tr>
								<th>word</th>
								{#each MODEL_ORDER.filter((m) => pairResults().some((r) => r.model === m)) as model}
									<th colspan="3" style="color: {MODEL_COLORS[model]}">{model.toUpperCase()}</th>
								{/each}
							</tr>
							<tr>
								<th></th>
								{#each MODEL_ORDER.filter((m) => pairResults().some((r) => r.model === m)) as _}
									<th class="sub">A</th><th class="sub">B</th><th class="sub">A+B</th>
								{/each}
							</tr>
						</thead>
						<tbody>
							{#each pairResults()[0].contested_words as cw, wi}
								<tr>
									<td class="word-cell">{cw.word}</td>
									{#each pairResults() as mr}
										{@const w = mr.contested_words[wi]}
										{#if w}
											<td class="num">{w.prob_a.toFixed(4)}</td>
											<td class="num">{w.prob_b.toFixed(4)}</td>
											<td class="num" class:superposed={Math.abs(w.prob_ab - w.prob_mean) < Math.abs(w.prob_ab - w.prob_a) && Math.abs(w.prob_ab - w.prob_mean) < Math.abs(w.prob_ab - w.prob_b)}
												class:resolved={Math.abs(w.prob_ab - w.prob_mean) >= Math.abs(w.prob_ab - w.prob_a) || Math.abs(w.prob_ab - w.prob_mean) >= Math.abs(w.prob_ab - w.prob_b)}>
												{w.prob_ab.toFixed(4)}
											</td>
										{:else}
											<td></td><td></td><td></td>
										{/if}
									{/each}
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.contradiction {
		display: flex;
		flex-direction: column;
		gap: 12px;
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
		white-space: nowrap;
	}

	.btn:hover:not(:disabled) { background: #344a70; }
	.btn:disabled { opacity: 0.4; cursor: not-allowed; }

	.pair-control {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: #888;
	}

	.pair-control select {
		background: #141428;
		border: 1px solid #2a2a44;
		color: #ccc;
		padding: 4px 8px;
		border-radius: 4px;
		font-size: 12px;
	}

	.status {
		text-align: center;
		color: #666;
		font-size: 13px;
		padding: 20px 0;
		max-width: 600px;
		margin: 0 auto;
		line-height: 1.5;
	}

	.status.error { color: #e15759; }

	.content-area {
		display: flex;
		flex-direction: column;
		gap: 16px;
	}

	.chart-area {
		position: relative;
		min-height: 300px;
	}

	.prompts {
		display: flex;
		flex-direction: column;
		gap: 4px;
		padding: 10px 12px;
		background: #111122;
		border-radius: 6px;
		border: 1px solid #1a1a2e;
		font-size: 12px;
		color: #bbb;
	}

	.prompt-label {
		color: #666;
		font-weight: 600;
		font-family: 'SF Mono', monospace;
		margin-right: 6px;
	}

	.words-section {
		display: flex;
		flex-direction: column;
		gap: 6px;
	}

	.words-label {
		font-size: 11px;
		color: #777;
		text-transform: uppercase;
		letter-spacing: 0.3px;
	}

	.words-table {
		border-collapse: collapse;
		font-size: 11px;
		font-family: 'SF Mono', monospace;
	}

	.words-table th {
		text-align: center;
		padding: 4px 6px;
		color: #888;
		border-bottom: 1px solid #2a2a44;
		font-weight: 500;
	}

	.words-table th.sub {
		font-size: 10px;
		color: #666;
	}

	.words-table td {
		padding: 3px 6px;
		border-bottom: 1px solid #1a1a2e;
		color: #aaa;
	}

	.word-cell {
		font-weight: 600;
		color: #ccc;
	}

	.num {
		text-align: right;
	}

	.superposed {
		color: #4e79a7;
	}

	.resolved {
		color: #e15759;
	}
</style>
