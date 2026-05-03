<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import type { DisplacementResult, FormationRow } from '$lib/api';
	import ExportButton from './ExportButton.svelte';

	interface Props {
		data: DisplacementResult;
		minSim?: number;
		prompt?: string;
		family?: string;
	}

	let { data, minSim = 0.45, prompt = '', family = '' }: Props = $props();

	let container: HTMLDivElement;

	const LAYER_LABELS: Record<string, string> = {
		base: 'Base (Id)',
		ego: 'SFT (Ego)',
		superego: 'DPO (Superego)',
		instruct: 'RLVR (Ego-ideal)'
	};

	function getLayers(row: FormationRow): string[] {
		const cols = ['base'];
		if (row.ego !== undefined) cols.push('ego');
		if (row.superego !== undefined) cols.push('superego');
		if (row.instruct !== undefined) cols.push('instruct');
		return cols;
	}

	function draw() {
		if (!container || !data?.df?.length) return;
		container.innerHTML = '';

		const df = data.df;
		const layers = getLayers(df[0]);
		const labels = layers.map((l) => LAYER_LABELS[l] || l);
		const nLayers = layers.length;

		const minProb = 0.003;
		const sig = df.filter((d) => {
			const vals = layers.map((l) => (d[l] as number ?? 0));
			return Math.max(...vals) > minProb;
		});

		const hasTitle = !!(prompt || family);
		const rect = container.getBoundingClientRect();
		const width = rect.width;
		const height = Math.max(600, rect.height);
		const margin = { top: hasTitle ? 48 : 24, right: 100, bottom: 40, left: 60 };
		const innerW = width - margin.left - margin.right;
		const innerH = height - margin.top - margin.bottom;

		const svg = d3
			.select(container)
			.append('svg')
			.attr('width', width)
			.attr('height', height);

		if (hasTitle) {
			const titleParts = ['Displacement'];
			if (family) titleParts.push(`— ${family}`);
			svg.append('text')
				.attr('x', width / 2).attr('y', 16)
				.attr('text-anchor', 'middle')
				.attr('fill', '#ccc').attr('font-size', '13px').attr('font-weight', '600')
				.text(titleParts.join(' '));
			if (prompt) {
				svg.append('text')
					.attr('x', width / 2).attr('y', 32)
					.attr('text-anchor', 'middle')
					.attr('fill', '#777').attr('font-size', '11px').attr('font-style', 'italic')
					.text(`"${prompt}"`);
			}
		}

		const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

		const x = d3
			.scalePoint<string>()
			.domain(labels)
			.range([0, innerW])
			.padding(0.1);
		const allVals = sig.flatMap((d) =>
			layers.map((l) => Math.max((d[l] as number ?? 0), 1e-7))
		);
		const y = d3
			.scaleLog()
			.domain([Math.max(d3.min(allVals)! * 0.5, 1e-7), d3.max(allVals)! * 1.5])
			.range([innerH, 0]);

		g.append('g')
			.attr('transform', `translate(0,${innerH})`)
			.call(d3.axisBottom(x))
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '11px');
		g.append('g')
			.call(
				d3
					.axisLeft(y)
					.ticks(6)
					.tickFormat((d) => d3.format('.1e')(d as number))
			)
			.selectAll('text')
			.style('fill', '#aaa')
			.style('font-size', '10px');
		g.selectAll('.domain, .tick line').style('stroke', '#444');

		// Background trajectories
		for (const row of sig) {
			const vals = layers.map((l) => (row as Record<string, unknown>)[l] as number);
			g.append('path')
				.datum(vals)
				.attr(
					'd',
					d3
						.line<number>()
						.x((_, i) => x(labels[i])!)
						.y((d) => y(Math.max(d, 1e-7)))
				)
				.attr('fill', 'none')
				.attr('stroke', 'rgba(200,200,200,0.15)')
				.attr('stroke-width', 1.5);
		}

		const wordY = (word: string, layerIdx: number) => {
			const row = df.find((d) => d.word === word);
			if (!row) return innerH / 2;
			const val = (row as Record<string, unknown>)[layers[layerIdx]] as number;
			return y(Math.max(val || 1e-7, 1e-7));
		};

		const wordX = (layerIdx: number) => x(labels[layerIdx])!;

		function drawArrows(
			pairs: [string, string, number, string][],
			fromLayer: number,
			toLayer: number,
			color: string,
			label: string
		) {
			const filtered = pairs.filter((p) => p[2] >= minSim);
			const best = new Map<string, [string, string, number, string]>();
			for (const p of filtered) {
				const key = `${p[0]}->${p[1]}`;
				if (!best.has(key) || p[2] > best.get(key)![2]) {
					best.set(key, p);
				}
			}

			for (const [, pair] of best) {
				const [src, tgt, sim] = pair;
				const x1 = wordX(fromLayer) + 4;
				const y1 = wordY(src, fromLayer);
				const x2 = wordX(toLayer) - 4;
				const y2 = wordY(tgt, toLayer);

				if (isNaN(y1) || isNaN(y2)) continue;

				const alpha = 0.3 + 0.5 * ((sim - minSim) / (1 - minSim));
				g.append('line')
					.attr('x1', x1)
					.attr('y1', y1)
					.attr('x2', x2)
					.attr('y2', y2)
					.attr('stroke', color)
					.attr('stroke-width', 1 + sim * 2)
					.attr('opacity', alpha);

				// Arrowhead
				const angle = Math.atan2(y2 - y1, x2 - x1);
				const aLen = 6;
				g.append('path')
					.attr(
						'd',
						`M${x2},${y2} L${x2 - aLen * Math.cos(angle - 0.4)},${y2 - aLen * Math.sin(angle - 0.4)} L${x2 - aLen * Math.cos(angle + 0.4)},${y2 - aLen * Math.sin(angle + 0.4)} Z`
					)
					.attr('fill', color)
					.attr('opacity', alpha);
			}
		}

		if (data.sublimation?.pairs?.length && nLayers >= 3) {
			drawArrows(data.sublimation.pairs, 0, 1, '#b07aa1', 'sublimation');
		}
		if (data.repression?.pairs?.length) {
			const fromLayer = nLayers >= 3 ? 1 : 0;
			const toLayer = nLayers >= 3 ? 2 : 1;
			drawArrows(data.repression.pairs, fromLayer, toLayer, '#e15759', 'repression');
		}

		// Word labels at significant points
		const labeled = new Set<string>();
		const allPairWords = new Set<string>();
		for (const p of data.sublimation?.pairs || []) {
			if (p[2] >= minSim) {
				allPairWords.add(p[0]);
				allPairWords.add(p[1]);
			}
		}
		for (const p of data.repression?.pairs || []) {
			if (p[2] >= minSim) {
				allPairWords.add(p[0]);
				allPairWords.add(p[1]);
			}
		}

		for (const row of sig) {
			if (!allPairWords.has(row.word)) continue;
			if (labeled.has(row.word)) continue;
			labeled.add(row.word);
			const vals = layers.map((l) => (row as Record<string, unknown>)[l] as number);
			const maxIdx = vals.indexOf(Math.max(...vals));
			g.append('text')
				.attr('x', x(labels[maxIdx])!)
				.attr('y', y(Math.max(vals[maxIdx], 1e-7)) - 8)
				.attr('text-anchor', 'middle')
				.attr('fill', '#ccc')
				.attr('font-size', '9px')
				.text(row.word);
		}
	}

	$effect(() => {
		data;
		minSim;
		draw();
	});

	onMount(draw);
</script>

<div style="display: flex; justify-content: flex-end; padding: 0 4px;">
	<ExportButton {container} filename="displacement.png" />
</div>
<div bind:this={container} class="chart-container" style="position: relative; width: 100%; height: 620px;"></div>
