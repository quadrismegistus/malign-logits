<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api';
	import type { BeamIndex, BeamStoryline } from '$lib/api';

	let index: BeamIndex | null = $state(null);
	let loading = $state(false);
	let storylines: BeamStoryline[] = $state([]);
	let selectedModel = $state('');
	let selectedPrompt = $state('');
	let selectedSource = $state('');
	let selectedAnnotator = $state('');
	let sortBy: 'rank' | 'resist' | 'prob' = $state('rank');
	let error = $state('');

	let sources: string[] = $derived(
		index && selectedModel ? (index.sources[selectedModel] || []) : []
	);

	let annotators: string[] = $derived(
		storylines.length > 0 && storylines[0].annotations
			? Object.keys(storylines[0].annotations)
			: []
	);

	let sorted: BeamStoryline[] = $derived.by(() => {
		const arr = [...storylines];
		if (sortBy === 'resist' && selectedAnnotator) {
			arr.sort((a, b) => {
				const ra = a.annotations?.[selectedAnnotator]?.total_resist ?? 0;
				const rb = b.annotations?.[selectedAnnotator]?.total_resist ?? 0;
				return rb - ra;
			});
		} else if (sortBy === 'prob') {
			arr.sort((a, b) => b.path_prob - a.path_prob);
		}
		return arr;
	});

	function resistColor(val: number): string {
		if (val > 2) return '#e15759';
		if (val > 0.5) return '#f28e2b';
		if (val > 0) return '#edc948';
		if (val > -0.5) return '#76b7b2';
		return '#59a14f';
	}

	function tokenResistColor(val: number): string {
		const clamped = Math.max(-3, Math.min(3, val));
		if (clamped > 0) {
			const t = clamped / 3;
			const r = Math.round(225 + (255 - 225) * t);
			const g = Math.round(87 + (50 - 87) * t);
			const b = Math.round(89 + (50 - 89) * t);
			return `rgb(${r},${g},${b})`;
		} else {
			const t = -clamped / 3;
			const r = Math.round(89 + (50 - 89) * t);
			const g = Math.round(161 + (200 - 161) * t);
			const b = Math.round(79 + (100 - 79) * t);
			return `rgb(${r},${g},${b})`;
		}
	}

	async function loadIndex() {
		try {
			index = await api.beamIndex();
			if (index.models.length > 0) {
				selectedModel = index.models[0];
				const srcs = index.sources[index.models[0]] || [];
				if (srcs.length > 0) selectedSource = srcs[0];
			}
			if (index.prompts.length > 0) {
				selectedPrompt = index.prompts[0];
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	async function loadStorylines() {
		if (!selectedModel || !selectedPrompt) return;
		loading = true;
		error = '';
		try {
			const res = await api.beamStorylines(selectedModel, selectedPrompt, 100, selectedSource);
			storylines = res.storylines;
			if (annotators.length > 0 && !selectedAnnotator) {
				selectedAnnotator = annotators[0];
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	$effect(() => {
		if (selectedModel && selectedPrompt && selectedSource) {
			loadStorylines();
		}
	});

	onMount(loadIndex);
</script>

<div class="beam-explorer">
	<div class="controls">
		{#if index}
			<label class="control">
				<span>Model</span>
				<select bind:value={selectedModel} onchange={() => {
					const srcs = index?.sources[selectedModel] || [];
					selectedSource = srcs[0] || '';
				}}>
					{#each index.models as m}
						<option value={m}>{index.nicknames?.[m] || m.split('/').pop()}</option>
					{/each}
				</select>
			</label>
			<label class="control">
				<span>Beams from</span>
				<select bind:value={selectedSource}>
					{#each sources as s}
						<option value={s}>{s}</option>
					{/each}
				</select>
			</label>
			<label class="control">
				<span>Prompt</span>
				<select bind:value={selectedPrompt}>
					{#each index.prompts as p}
						<option value={p}>{p.length > 60 ? p.slice(0, 57) + '...' : p}</option>
					{/each}
				</select>
			</label>
			{#if annotators.length > 0}
				<label class="control">
					<span>Annotator</span>
					<select bind:value={selectedAnnotator}>
						{#each annotators as a}
							<option value={a}>{a}</option>
						{/each}
					</select>
				</label>
			{/if}
			<label class="control">
				<span>Sort</span>
				<select bind:value={sortBy}>
					<option value="rank">beam rank</option>
					<option value="resist">resistance</option>
					<option value="prob">probability</option>
				</select>
			</label>
		{:else if error}
			<div class="error">{error}</div>
		{:else}
			<span class="loading-text">Loading beam index...</span>
		{/if}
	</div>

	{#if loading}
		<div class="loading-text">Loading storylines...</div>
	{:else if storylines.length === 0 && !error}
		<div class="empty">
			<p>No beam data found. Run <code>malign probe batch</code> or <code>python scripts/cloud_beam_annotate.py</code> to generate beam data.</p>
		</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else}
		<div class="summary">
			{storylines.length} storylines &middot;
			{selectedModel.split('/').pop()} &middot;
			&ldquo;{selectedPrompt.length > 40 ? selectedPrompt.slice(0, 37) + '...' : selectedPrompt}&rdquo;
		</div>

		<div class="storyline-list">
			{#each sorted as story, i}
				{@const ann = selectedAnnotator && story.annotations?.[selectedAnnotator]}
				<div class="storyline">
					<div class="storyline-meta">
						<span class="rank">#{story.rank + 1}</span>
						<span class="prob">{(story.path_prob * 100).toFixed(1)}%</span>
						{#if ann}
							<span class="resist" style="color: {resistColor(ann.total_resist)}">
								{ann.total_resist > 0 ? '+' : ''}{ann.total_resist.toFixed(1)}b
							</span>
						{/if}
					</div>
					<div class="tokens">
						{#each story.tokens as tok, ti}
							{@const tr = ann?.token_resist?.[ti]}
							<span
								class="token"
								style="background: {tr !== undefined ? tokenResistColor(tr) : 'transparent'}; color: {tr !== undefined && Math.abs(tr) > 1.5 ? '#fff' : '#ccc'}"
								title={tr !== undefined ? `resist: ${tr > 0 ? '+' : ''}${tr.toFixed(2)}b` : ''}
							>{tok}</span>
						{/each}
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.beam-explorer {
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

	.summary {
		font-size: 12px;
		color: #888;
		padding: 4px 0;
		border-bottom: 1px solid #1a1a2e;
	}

	.storyline-list {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.storyline {
		display: flex;
		align-items: baseline;
		gap: 12px;
		padding: 6px 8px;
		border-radius: 4px;
		transition: background 0.1s;
	}

	.storyline:hover {
		background: rgba(255, 255, 255, 0.03);
	}

	.storyline-meta {
		display: flex;
		gap: 8px;
		min-width: 120px;
		flex-shrink: 0;
		font-family: 'SF Mono', monospace;
		font-size: 11px;
	}

	.rank {
		color: #555;
		min-width: 28px;
	}

	.prob {
		color: #4e79a7;
		min-width: 40px;
		text-align: right;
	}

	.resist {
		min-width: 45px;
		text-align: right;
		font-weight: 500;
	}

	.tokens {
		display: flex;
		flex-wrap: wrap;
		gap: 1px;
		font-family: 'SF Mono', monospace;
		font-size: 12px;
	}

	.token {
		padding: 2px 4px;
		border-radius: 3px;
		white-space: pre;
		cursor: default;
	}

	.loading-text {
		color: #888;
		font-size: 13px;
	}

	.empty {
		color: #555;
		font-size: 13px;
		padding: 24px;
		text-align: center;
	}

	.empty code {
		color: #4e79a7;
		background: rgba(78, 121, 167, 0.1);
		padding: 2px 6px;
		border-radius: 3px;
		font-size: 12px;
	}

	.error {
		color: #e15759;
		font-size: 12px;
	}
</style>
