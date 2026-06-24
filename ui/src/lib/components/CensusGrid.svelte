<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api';

	interface CensusRow {
		family: string;
		prompt: string;
		dominant: string;
		base_top1: string;
		aligned_top1: string;
		base_H: number;
		aligned_H: number;
	}

	let rows: CensusRow[] = $state([]);
	let loading = $state(true);
	let error = $state('');
	let selectedPrompt = $state('all');
	let colorBy: 'dominant' | 'entropy' = $state('dominant');

	let families = $derived([...new Set(rows.map(r => r.family))].sort());
	let prompts = $derived([...new Set(rows.map(r => r.prompt))].sort());

	let filtered = $derived(
		selectedPrompt === 'all' ? rows : rows.filter(r => r.prompt === selectedPrompt)
	);

	const COLORS: Record<string, string> = {
		transparent: '#59a14f',
		repression: '#e15759',
		foreclosure: '#4e79a7',
		reaction_formation: '#f28e2b',
		return_of_repressed: '#b07aa1',
		de_foreclosure: '#76b7b2',
	};

	const LABELS: Record<string, string> = {
		transparent: 'T',
		repression: 'R',
		foreclosure: 'F',
		reaction_formation: 'RF',
		return_of_repressed: 'RR',
		de_foreclosure: 'DF',
	};

	function cellColor(row: CensusRow): string {
		if (colorBy === 'dominant') {
			return COLORS[row.dominant] || '#666';
		}
		const delta = row.aligned_H - row.base_H;
		if (delta < -1) return '#e15759';
		if (delta < -0.3) return '#f28e2b';
		if (delta < 0.3) return '#888';
		if (delta < 1) return '#76b7b2';
		return '#59a14f';
	}

	function entropyArrow(row: CensusRow): string {
		const delta = row.aligned_H - row.base_H;
		if (delta < -0.3) return '↓';
		if (delta > 0.3) return '↑';
		return '→';
	}

	async function loadData() {
		try {
			const res = await fetch('/api/api/data/csv?name=circuit_census_grid_final');
			if (!res.ok) {
				const res2 = await fetch('/api/data/csv?name=circuit_census_grid_final');
				if (!res2.ok) throw new Error('Failed to load census data');
				const data = await res2.json();
				rows = data.rows;
			} else {
				const data = await res.json();
				rows = data.rows;
			}
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	onMount(loadData);
</script>

<div class="census">
	<div class="controls">
		<label class="control">
			<span>Prompt</span>
			<select bind:value={selectedPrompt}>
				<option value="all">all prompts</option>
				{#each prompts as p}
					<option value={p}>{p}</option>
				{/each}
			</select>
		</label>
		<label class="control">
			<span>Color by</span>
			<select bind:value={colorBy}>
				<option value="dominant">mechanism</option>
				<option value="entropy">entropy change</option>
			</select>
		</label>
		<div class="legend">
			{#each Object.entries(COLORS) as [key, color]}
				<span class="legend-item">
					<span class="swatch" style="background: {color}"></span>
					{key.replace(/_/g, ' ')}
				</span>
			{/each}
		</div>
	</div>

	{#if loading}
		<div class="loading">Loading census data...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else if selectedPrompt === 'all'}
		<div class="grid-container">
			<table class="grid">
				<thead>
					<tr>
						<th class="family-header"></th>
						{#each prompts as p}
							<th class="prompt-header">{p}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each families as fam}
						<tr>
							<td class="family-cell">{fam}</td>
							{#each prompts as p}
								{@const row = rows.find(r => r.family === fam && r.prompt === p)}
								{#if row}
									<td class="data-cell" style="background: {cellColor(row)}22; border-left: 3px solid {cellColor(row)}">
										<div class="cell-label">{LABELS[row.dominant] || '?'}</div>
										<div class="cell-tokens">
											{row.base_top1} → {row.aligned_top1}
										</div>
										<div class="cell-entropy">
											H: {row.base_H.toFixed(1)} {entropyArrow(row)} {row.aligned_H.toFixed(1)}
										</div>
									</td>
								{:else}
									<td class="data-cell empty-cell">—</td>
								{/if}
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<div class="list-view">
			{#each filtered.sort((a, b) => a.family.localeCompare(b.family)) as row}
				<div class="list-row" style="border-left: 4px solid {cellColor(row)}">
					<div class="list-family">{row.family}</div>
					<div class="list-mechanism" style="color: {cellColor(row)}">{row.dominant.replace(/_/g, ' ')}</div>
					<div class="list-shift">
						<span class="base-token">{row.base_top1}</span>
						<span class="arrow">→</span>
						<span class="aligned-token">{row.aligned_top1}</span>
					</div>
					<div class="list-entropy">
						H: {row.base_H.toFixed(1)} {entropyArrow(row)} {row.aligned_H.toFixed(1)}
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.census {
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
	}

	.control select:focus {
		outline: none;
		border-color: #4e79a7;
	}

	.legend {
		display: flex;
		gap: 10px;
		flex-wrap: wrap;
		font-size: 11px;
		color: #888;
		margin-left: auto;
	}

	.legend-item {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.swatch {
		width: 10px;
		height: 10px;
		border-radius: 2px;
		display: inline-block;
	}

	.grid-container {
		overflow-x: auto;
	}

	.grid {
		border-collapse: separate;
		border-spacing: 2px;
		width: 100%;
	}

	.family-header {
		width: 120px;
	}

	.prompt-header {
		text-align: center;
		font-size: 12px;
		color: #aaa;
		font-weight: 500;
		text-transform: capitalize;
		padding: 6px 8px;
	}

	.family-cell {
		font-size: 12px;
		color: #ccc;
		font-family: 'SF Mono', monospace;
		padding: 6px 8px;
		white-space: nowrap;
	}

	.data-cell {
		padding: 6px 8px;
		border-radius: 4px;
		vertical-align: top;
		min-width: 120px;
	}

	.empty-cell {
		color: #444;
		text-align: center;
	}

	.cell-label {
		font-size: 13px;
		font-weight: 600;
		margin-bottom: 2px;
	}

	.cell-tokens {
		font-size: 11px;
		font-family: 'SF Mono', monospace;
		color: #ccc;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.cell-entropy {
		font-size: 10px;
		color: #888;
		font-family: 'SF Mono', monospace;
		margin-top: 2px;
	}

	.list-view {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.list-row {
		display: flex;
		align-items: center;
		gap: 16px;
		padding: 8px 12px;
		border-radius: 4px;
		background: rgba(255, 255, 255, 0.02);
	}

	.list-row:hover {
		background: rgba(255, 255, 255, 0.04);
	}

	.list-family {
		font-family: 'SF Mono', monospace;
		font-size: 12px;
		color: #ccc;
		min-width: 120px;
	}

	.list-mechanism {
		font-size: 12px;
		font-weight: 500;
		min-width: 150px;
	}

	.list-shift {
		font-family: 'SF Mono', monospace;
		font-size: 12px;
		min-width: 180px;
	}

	.base-token {
		color: #aaa;
	}

	.arrow {
		color: #555;
		margin: 0 4px;
	}

	.aligned-token {
		color: #e0e0e0;
	}

	.list-entropy {
		font-family: 'SF Mono', monospace;
		font-size: 11px;
		color: #888;
	}

	.loading {
		color: #888;
		font-size: 13px;
		padding: 24px;
	}

	.error {
		color: #e15759;
		font-size: 12px;
	}
</style>
