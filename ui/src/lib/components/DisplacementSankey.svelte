<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api';
	import type { BeamIndex } from '$lib/api';

	let index: BeamIndex | null = $state(null);
	let loading = $state(false);
	let error = $state('');
	let selectedModel = $state('');
	let selectedPrompt = $state('');
	let depth = $state(1);
	let mode: 'beam' | 'word' = $state('word');
	let topN = $state(10);

	interface SankeyData {
		sources: Record<string, Record<string, number>>;
		model: string;
		prompt: string;
		depth: number;
		logit_probs?: Record<string, Record<string, number>>;
		nicknames?: Record<string, string>;
	}

	let sankeyData: SankeyData | null = $state(null);

	let sourceOrder = $derived.by(() => {
		if (!sankeyData) return [];
		const srcs = Object.keys(sankeyData.sources);
		const base = selectedModel.split('/').pop()?.replace(/-/g, '_') || '';
		const STAGE_ORDER = ['', 'SFT', 'DPO', 'Instr', 'Chat', 'beta', 'alpha'];
		srcs.sort((a, b) => {
			if (a === base) return -1;
			if (b === base) return 1;
			const aStage = STAGE_ORDER.findIndex(s => s && a.includes(s));
			const bStage = STAGE_ORDER.findIndex(s => s && b.includes(s));
			if (aStage !== -1 && bStage !== -1) return aStage - bStage;
			if (aStage !== -1) return -1;
			if (bStage !== -1) return 1;
			return a.localeCompare(b);
		});
		return srcs;
	});

	const STAGE_COLORS = [
		'#4e79a7', '#f28e2b', '#e15759', '#59a14f', '#b07aa1', '#edc948', '#76b7b2'
	];

	const COL_W = 100;
	const COL_GAP = 180;
	const H_PER_STAGE = 400;
	const PAD = { top: 30, bottom: 20, left: 10, right: 10 };
	let W = $derived(PAD.left + PAD.right + sourceOrder.length * COL_W + Math.max(0, sourceOrder.length - 1) * COL_GAP);

	let svgData = $derived.by(() => {
		if (!sankeyData || sourceOrder.length < 2) return null;

		const stages = sourceOrder;
		const colSpacing = COL_W + COL_GAP;
		const totalH = H_PER_STAGE;

		// Collect top-N words from each stage
		const topWords = new Set<string>();
		for (const stage of stages) {
			const dist = sankeyData.sources[stage] || {};
			const sorted = Object.entries(dist).sort(([, a], [, b]) => b - a);
			for (let i = 0; i < Math.min(topN, sorted.length); i++) {
				topWords.add(sorted[i][0]);
			}
		}

		const columns: { stage: string; tokens: { label: string; count: number; y: number; h: number; logitProb: number | null }[] }[] = [];

		for (let si = 0; si < stages.length; si++) {
			const dist = sankeyData.sources[stages[si]] || {};
			const logits = sankeyData.logit_probs?.[stages[si]] || {};
			const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
			const tokens = Object.entries(dist)
				.filter(([label]) => topWords.has(label))
				.sort(([, a], [, b]) => b - a)
				.map(([label, count]) => ({ label, count, y: 0, h: 0, logitProb: logits[label] ?? null }));

			let y = PAD.top;
			const usableH = totalH - PAD.top - PAD.bottom;
			for (const t of tokens) {
				t.h = Math.max(8, (t.count / total) * usableH);
				t.y = y;
				y += t.h + 2;
			}
			columns.push({ stage: stages[si], tokens });
		}

		// Links between adjacent columns
		const links: { x1: number; y1: number; h1: number; x2: number; y2: number; h2: number; color: string; label: string; count: number }[] = [];

		for (let ci = 0; ci < columns.length - 1; ci++) {
			const left = columns[ci];
			const right = columns[ci + 1];
			const x1 = PAD.left + ci * colSpacing + COL_W;
			const x2 = PAD.left + (ci + 1) * colSpacing;

			for (const lt of left.tokens) {
				const rt = right.tokens.find(t => t.label === lt.label);
				if (rt) {
					links.push({
						x1, y1: lt.y, h1: lt.h,
						x2, y2: rt.y, h2: rt.h,
						color: STAGE_COLORS[ci % STAGE_COLORS.length],
						label: lt.label,
						count: Math.min(lt.count, rt.count),
					});
				}
			}
		}

		return { columns, links, colSpacing };
	});

	function linkPath(link: typeof svgData extends { links: (infer T)[] } ? T : never): string {
		const midY1 = link.y1 + link.h1 / 2;
		const midY2 = link.y2 + link.h2 / 2;
		const h = Math.min(link.h1, link.h2);
		const mx = (link.x1 + link.x2) / 2;
		return `M${link.x1},${midY1 - h / 2} C${mx},${midY1 - h / 2} ${mx},${midY2 - h / 2} ${link.x2},${midY2 - h / 2} L${link.x2},${midY2 + h / 2} C${mx},${midY2 + h / 2} ${mx},${midY1 + h / 2} ${link.x1},${midY1 + h / 2} Z`;
	}

	async function loadIndex() {
		try {
			index = await api.beamIndex();
			if (index.models.length > 0) selectedModel = index.models[0];
			if (index.prompts.length > 0) selectedPrompt = index.prompts[0];
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	async function loadSankey() {
		if (!selectedModel || !selectedPrompt) return;
		loading = true;
		error = '';
		try {
			const res = await fetch(`/api/api/beam/sankey?model=${encodeURIComponent(selectedModel)}&prompt=${encodeURIComponent(selectedPrompt)}&depth=${depth}&mode=${mode}`);
			if (!res.ok) {
				const res2 = await fetch(`/api/beam/sankey?model=${encodeURIComponent(selectedModel)}&prompt=${encodeURIComponent(selectedPrompt)}&depth=${depth}&mode=${mode}`);
				if (!res2.ok) throw new Error('Failed');
				sankeyData = await res2.json();
			} else {
				sankeyData = await res.json();
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		if (selectedModel && selectedPrompt) {
			loadSankey();
		}
	});

	$effect(() => {
		mode;
		depth;
		if (selectedModel && selectedPrompt) {
			loadSankey();
		}
	});

	onMount(loadIndex);
</script>

<div class="sankey-container">
	<div class="controls">
		{#if index}
			<label class="control">
				<span>Model</span>
				<select bind:value={selectedModel}>
					{#each index.models as m}
						<option value={m}>{index.nicknames?.[m] || m.split('/').pop()}</option>
					{/each}
				</select>
			</label>
			<label class="control">
				<span>Prompt</span>
				<select bind:value={selectedPrompt}>
					{#each index.prompts as p}
						<option value={p}>{p.length > 55 ? p.slice(0, 52) + '...' : p}</option>
					{/each}
				</select>
			</label>
			<label class="control">
				<span>Mode</span>
				<select bind:value={mode}>
					<option value="beam">storyline (syntagmatic)</option>
					<option value="word">word (paradigmatic)</option>
				</select>
			</label>
			{#if mode === 'beam'}
				<label class="control">
					<span>Depth</span>
					<select bind:value={depth}>
						<option value={1}>1 token</option>
						<option value={2}>2 tokens</option>
						<option value={3}>3 tokens</option>
					</select>
				</label>
			{/if}
			<label class="control">
				<span>Top N</span>
				<select bind:value={topN}>
					<option value={5}>5</option>
					<option value={10}>10</option>
					<option value={15}>15</option>
					<option value={20}>20</option>
					<option value={30}>30</option>
					<option value={50}>all</option>
				</select>
			</label>
		{:else}
			<span class="loading-text">Loading...</span>
		{/if}
	</div>

	{#if loading}
		<div class="loading-text">Computing Sankey...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else if svgData}
		<svg viewBox="0 0 {W} {H_PER_STAGE}" class="sankey">
			<!-- Column headers -->
			{#each svgData.columns as col, ci}
				<text
					x={PAD.left + ci * svgData.colSpacing + COL_W / 2}
					y={14}
					class="stage-label"
				>{sankeyData?.nicknames?.[col.stage] || col.stage}</text>
			{/each}

			<!-- Links -->
			{#each svgData.links as link}
				<path d={linkPath(link)} fill={link.color} opacity="0.2" />
			{/each}

			<!-- Token bars -->
			{#each svgData.columns as col, ci}
				{@const x = PAD.left + ci * svgData.colSpacing}
				{#each col.tokens as tok}
					<rect
						{x}
						y={tok.y}
						width={COL_W}
						height={tok.h}
						fill={STAGE_COLORS[ci % STAGE_COLORS.length]}
						rx="3"
						opacity="0.7"
					/>
					{#if tok.h > 12}
						{#if mode === 'word'}
							<text
								x={x + COL_W / 2}
								y={tok.y + tok.h / 2 + 4}
								class="tok-label"
							>{tok.label} ({tok.count}%)</text>
						{:else}
							<text
								x={x + COL_W / 2}
								y={tok.y + tok.h / 2 + (tok.logitProb !== null ? 0 : 4)}
								class="tok-label"
							>{tok.label} ({tok.count})</text>
							{#if tok.logitProb !== null}
								<text
									x={x + COL_W / 2}
									y={tok.y + tok.h / 2 + 13}
									class="logit-label"
								>logit: {(tok.logitProb * 100).toFixed(1)}%</text>
							{/if}
						{/if}
					{/if}
				{/each}
			{/each}
		</svg>
	{:else}
		<div class="empty">Select a model and prompt to view displacement flow.</div>
	{/if}
</div>

<style>
	.sankey-container {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.controls {
		display: flex;
		gap: 12px;
		flex-wrap: wrap;
		align-items: end;
	}

	.control {
		display: flex;
		flex-direction: column;
		gap: 4px;
		font-size: 11px;
		color: #888;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.control select {
		background: #141428;
		border: 1px solid #2a2a44;
		color: #ccc;
		padding: 6px 8px;
		border-radius: 4px;
		font-size: 12px;
		font-family: inherit;
		max-width: 300px;
	}

	.control select:focus {
		outline: none;
		border-color: #4e79a7;
	}

	.sankey {
		width: 100%;
		max-width: 900px;
		height: auto;
	}

	.stage-label {
		font-size: 9px;
		fill: #aaa;
		text-anchor: middle;
		font-family: 'SF Mono', monospace;
	}

	.tok-label {
		font-size: 10px;
		fill: #fff;
		text-anchor: middle;
		font-family: 'SF Mono', monospace;
		pointer-events: none;
	}

	.logit-label {
		font-size: 8px;
		fill: #fffc;
		text-anchor: middle;
		font-family: 'SF Mono', monospace;
		font-style: italic;
		pointer-events: none;
	}

	.loading-text {
		color: #888;
		font-size: 13px;
	}

	.empty {
		color: #555;
		font-size: 13px;
		padding: 24px;
	}

	.error {
		color: #e15759;
		font-size: 12px;
	}
</style>
