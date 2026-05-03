<script lang="ts">
	import { onMount } from 'svelte';
	import * as d3 from 'd3';
	import type { FormationRow } from '$lib/api';
	import ExportButton from './ExportButton.svelte';

	interface Props {
		data: FormationRow[];
		topN?: number;
		minProb?: number;
		sortBy?: 'delta' | 'mass';
		prompt?: string;
		family?: string;
	}

	let { data, topN = 60, minProb = 0.001, sortBy = 'delta', prompt = '', family = '' }: Props = $props();

	let container: HTMLDivElement;
	let hiddenShapes: Set<string> = new Set();

	const LAYER_LABELS: Record<string, string> = {
		base: 'BASE',
		sft: 'SFT',
		dpo: 'DPO',
		rlvr: 'RLVR'
	};

	function arrowLabel(vals: number[]): string {
		const arrows: string[] = [];
		for (let i = 1; i < vals.length; i++) {
			const prev = Math.max(vals[i - 1], 1e-12);
			const curr = Math.max(vals[i], 1e-12);
			const ratio = curr / prev;
			if (ratio > 1.2) arrows.push('▲');
			else if (ratio < 0.8) arrows.push('▼');
			else arrows.push('–');
		}
		return arrows.join('');
	}

	const ARROW_COLORS: Record<string, string> = {
		'▲': '#4e79a7',
		'▼': '#e15759',
		'–': '#6f6f6f',
	};

	function arrowColor(label: string): string {
		const downs = (label.match(/▼/g) || []).length;
		const ups = (label.match(/▲/g) || []).length;
		const flats = (label.match(/–/g) || []).length;
		if (downs > 0 && ups === 0) return '#e15759';
		if (ups > 0 && downs === 0) return '#4e79a7';
		if (downs > 0 && ups > 0) return '#f28e2b';
		return '#6f6f6f';
	}

	function getLayers(row: FormationRow): string[] {
		const cols = ['base'];
		if (row.sft !== undefined) cols.push('sft');
		if (row.dpo !== undefined) cols.push('dpo');
		if (row.rlvr !== undefined) cols.push('rlvr');
		return cols;
	}

	function draw() {
		if (!container || !data.length) return;
		container.innerHTML = '';

		const layers = getLayers(data[0]);
		const labels = layers.map((l) => LAYER_LABELS[l] || l);

		let filtered = data.filter((d) => {
			const vals = layers.map((l) => (d[l] as number ?? 0));
			return Math.max(...vals) > minProb;
		});

		if (sortBy === 'delta') {
			filtered = filtered
				.map((d) => {
					const vals = layers.map((l) => (d[l] as number ?? 0));
					let maxDelta = 0;
					for (let i = 1; i < vals.length; i++) {
						maxDelta = Math.max(maxDelta, Math.abs(vals[i] - vals[i - 1]));
					}
					return { ...d, _sort: maxDelta };
				})
				.sort((a, b) => b._sort - a._sort)
				.slice(0, topN);
		} else {
			filtered = filtered
				.map((d) => {
					const vals = layers.map((l) => (d[l] as number ?? 0));
					return { ...d, _sort: vals.reduce((a, b) => a + b, 0) };
				})
				.sort((a, b) => b._sort - a._sort)
				.slice(0, topN);
		}

		const hasTitle = !!(prompt || family);
		const rect = container.getBoundingClientRect();
		const width = rect.width;
		const height = Math.max(500, rect.height);
		const margin = { top: hasTitle ? 48 : 24, right: 120, bottom: 40, left: 60 };
		const innerW = width - margin.left - margin.right;
		const innerH = height - margin.top - margin.bottom;

		const svg = d3
			.select(container)
			.append('svg')
			.attr('width', width)
			.attr('height', height);

		if (hasTitle) {
			const titleParts = ['Trajectory'];
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

		const allVals = filtered.flatMap((d) =>
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

		const line = d3
			.line<number>()
			.x((_, i) => x(labels[i])!)
			.y((d) => y(Math.max(d, 1e-7)));

		const tooltip = d3
			.select(container)
			.append('div')
			.style('position', 'absolute')
			.style('pointer-events', 'none')
			.style('background', 'rgba(20,20,20,0.95)')
			.style('border', '1px solid #555')
			.style('padding', '8px 12px')
			.style('border-radius', '4px')
			.style('font-size', '12px')
			.style('color', '#ddd')
			.style('display', 'none')
			.style('z-index', '100')
			.style('max-width', '300px');

		for (const row of filtered) {
			const vals = layers.map((l) => (row as Record<string, unknown>)[l] as number);
			const shape = arrowLabel(vals);
			const color = arrowColor(shape);
			const shapeClass = `shape-${[...shape].map(c => c.charCodeAt(0)).join('_')}`;

			const wordGroup = g.append('g')
				.attr('class', shapeClass)
				.style('display', hiddenShapes.has(shape) ? 'none' : null);

			const path = wordGroup
				.append('path')
				.datum(vals)
				.attr('d', line)
				.attr('fill', 'none')
				.attr('stroke', color)
				.attr('stroke-width', 1.5)
				.attr('opacity', 0.6);

			const dots = wordGroup
				.selectAll(null)
				.data(vals)
				.enter()
				.append('circle')
				.attr('cx', (_, i) => x(labels[i])!)
				.attr('cy', (d) => y(Math.max(d, 1e-7)))
				.attr('r', 3)
				.attr('fill', color)
				.attr('opacity', 0.8);

			wordGroup
				.selectAll(null)
				.data(vals)
				.enter()
				.append('rect')
				.attr('x', (_, i) => x(labels[i])! - 15)
				.attr('y', (d) => y(Math.max(d, 1e-7)) - 10)
				.attr('width', 30)
				.attr('height', 20)
				.attr('fill', 'transparent')
				.attr('cursor', 'pointer')
				.on('mouseenter', (event) => {
					path.attr('stroke-width', 3).attr('opacity', 1);
					dots.attr('r', 5);
					const valsText = layers.map((l, i) => `${l}: ${vals[i].toExponential(3)}`).join('\n');
					tooltip
						.style('display', 'block')
						.html(
							`<strong>${row.word}</strong><br><span style="color:${color}">${shape}</span><br>${valsText.replace(/\n/g, '<br>')}`
						);
				})
				.on('mousemove', (event) => {
					const [mx, my] = d3.pointer(event, container);
					tooltip.style('left', mx + 16 + 'px').style('top', my - 10 + 'px');
				})
				.on('mouseleave', () => {
					path.attr('stroke-width', 1.5).attr('opacity', 0.6);
					dots.attr('r', 3);
					tooltip.style('display', 'none');
				});

			wordGroup.append('text')
				.attr('x', x(labels[labels.length - 1])! + 8)
				.attr('y', y(Math.max(vals[vals.length - 1], 1e-7)))
				.attr('dy', '0.35em')
				.attr('fill', color)
				.attr('font-size', '10px')
				.attr('opacity', 0.9)
				.text(row.word);
		}

		// Legend (clickable toggles)
		const allShapes = filtered.map((d) => {
			const v = layers.map((l) => (d as Record<string, unknown>)[l] as number);
			return arrowLabel(v);
		});
		const shapes = [...new Set(allShapes)].sort();
		const legend = svg
			.append('g')
			.attr('transform', `translate(${margin.left + 8}, ${margin.top + 4})`);

		shapes.forEach((shape, i) => {
			const isHidden = hiddenShapes.has(shape);
			const color = arrowColor(shape);
			const row = legend.append('g')
				.attr('transform', `translate(0, ${i * 18})`)
				.style('cursor', 'pointer')
				.on('click', () => {
					if (hiddenShapes.has(shape)) {
						hiddenShapes.delete(shape);
					} else {
						hiddenShapes.add(shape);
					}
					hiddenShapes = new Set(hiddenShapes);
					draw();
				});
			row
				.append('rect')
				.attr('width', 10)
				.attr('height', 10)
				.attr('fill', color)
				.attr('opacity', isHidden ? 0.2 : 1)
				.attr('rx', 2);
			row
				.append('text')
				.attr('x', 14)
				.attr('y', 10)
				.attr('fill', isHidden ? '#444' : '#ccc')
				.attr('font-size', '12px')
				.text(shape);
		});
	}

	$effect(() => {
		data;
		topN;
		minProb;
		sortBy;
		draw();
	});

	onMount(draw);
</script>

<div style="display: flex; justify-content: flex-end; padding: 0 4px;">
	<ExportButton {container} filename="trajectory.png" />
</div>
<div bind:this={container} class="chart-container" style="position: relative; width: 100%; height: 560px;"></div>
