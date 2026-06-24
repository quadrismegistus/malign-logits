<script lang="ts">
	import { onMount } from 'svelte';

	interface CorrRow {
		family_1: string;
		family_2: string;
		spearman_r: number;
		p_value: number;
	}

	let rows: CorrRow[] = $state([]);
	let loading = $state(true);
	let error = $state('');

	let families = $derived([...new Set(rows.map(r => r.family_1))].sort());

	function getCorr(f1: string, f2: string): CorrRow | undefined {
		return rows.find(r => r.family_1 === f1 && r.family_2 === f2);
	}

	function cellColor(r: number): string {
		if (r >= 0.8) return 'rgba(89, 161, 79, 0.8)';
		if (r >= 0.5) return 'rgba(89, 161, 79, 0.5)';
		if (r >= 0.3) return 'rgba(89, 161, 79, 0.25)';
		if (r >= 0.1) return 'rgba(89, 161, 79, 0.1)';
		if (r >= -0.1) return 'rgba(128, 128, 128, 0.1)';
		if (r >= -0.3) return 'rgba(225, 87, 89, 0.15)';
		if (r >= -0.5) return 'rgba(225, 87, 89, 0.3)';
		return 'rgba(225, 87, 89, 0.5)';
	}

	function textColor(r: number): string {
		if (Math.abs(r) >= 0.3) return '#e0e0e0';
		return '#999';
	}

	function shortName(f: string): string {
		return f.replace(/-7[Bb].*/, '').replace(/-v0.*/, '').replace('deepseek-llm', 'DeepSeek').replace('-8B-Base', '');
	}

	async function loadData() {
		try {
			let res = await fetch('/api/api/data/csv?name=cross_family_beam_correlation');
			if (!res.ok) res = await fetch('/api/data/csv?name=cross_family_beam_correlation');
			if (!res.ok) throw new Error('Failed to load correlation data');
			const data = await res.json();
			rows = data.rows;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	onMount(loadData);
</script>

<div class="heatmap-container">
	<h3 class="title">Cross-Family Survival Correlation</h3>
	<p class="subtitle">Spearman r of 10-token survival rates across 71 prompts. Near-zero = families don't agree on which prompts to align.</p>

	{#if loading}
		<div class="loading">Loading correlation data...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else}
		<div class="grid-scroll">
			<table class="corr-grid">
				<thead>
					<tr>
						<th class="corner"></th>
						{#each families as f}
							<th class="col-header"><span>{shortName(f)}</span></th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each families as f1}
						<tr>
							<td class="row-header">{shortName(f1)}</td>
							{#each families as f2}
								{@const c = getCorr(f1, f2)}
								{@const r = c?.spearman_r ?? 0}
								{@const p = c?.p_value ?? 1}
								<td
									class="cell"
									class:diagonal={f1 === f2}
									style="background: {f1 === f2 ? '#1a1a2e' : cellColor(r)}; color: {textColor(r)}"
									title="{f1} vs {f2}: r={r.toFixed(3)}, p={p.toFixed(4)}"
								>
									{#if f1 === f2}
										—
									{:else}
										{r > 0 ? '+' : ''}{r.toFixed(2)}
										{#if p < 0.05}<span class="sig">*</span>{/if}
									{/if}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<div class="legend">
			<span class="leg-item"><span class="leg-swatch" style="background: rgba(89,161,79,0.5)"></span> r &gt; 0.3</span>
			<span class="leg-item"><span class="leg-swatch" style="background: rgba(89,161,79,0.1)"></span> 0 &lt; r &lt; 0.3</span>
			<span class="leg-item"><span class="leg-swatch" style="background: rgba(128,128,128,0.1)"></span> r &asymp; 0</span>
			<span class="leg-item"><span class="leg-swatch" style="background: rgba(225,87,89,0.3)"></span> r &lt; -0.1</span>
			<span class="leg-item"><span class="sig-label">*</span> p &lt; 0.05</span>
		</div>

		<div class="interpretation">
			Mean off-diagonal |r| = {(rows.filter(r => r.family_1 !== r.family_2).reduce((a, r) => a + Math.abs(r.spearman_r), 0) / rows.filter(r => r.family_1 !== r.family_2).length).toFixed(3)}
			— families show near-zero agreement on which prompts to align.
		</div>
	{/if}
</div>

<style>
	.heatmap-container {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.title {
		margin: 0;
		font-size: 16px;
		font-weight: 600;
		color: #e0e0e0;
	}

	.subtitle {
		margin: 0;
		font-size: 12px;
		color: #888;
	}

	.grid-scroll {
		overflow-x: auto;
	}

	.corr-grid {
		border-collapse: separate;
		border-spacing: 2px;
	}

	.corner {
		width: 100px;
	}

	.col-header {
		text-align: center;
		padding: 4px 2px;
		min-width: 64px;
	}

	.col-header span {
		font-size: 10px;
		color: #aaa;
		font-family: 'SF Mono', monospace;
		writing-mode: vertical-lr;
		transform: rotate(180deg);
		display: inline-block;
		white-space: nowrap;
	}

	.row-header {
		font-size: 11px;
		color: #ccc;
		font-family: 'SF Mono', monospace;
		padding: 4px 8px;
		text-align: right;
		white-space: nowrap;
	}

	.cell {
		text-align: center;
		font-family: 'SF Mono', monospace;
		font-size: 11px;
		padding: 6px 4px;
		border-radius: 3px;
		min-width: 56px;
		cursor: default;
	}

	.cell.diagonal {
		color: #444 !important;
	}

	.sig {
		color: #edc948;
		font-weight: 700;
		font-size: 13px;
	}

	.legend {
		display: flex;
		gap: 14px;
		font-size: 11px;
		color: #888;
		flex-wrap: wrap;
	}

	.leg-item {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.leg-swatch {
		width: 14px;
		height: 14px;
		border-radius: 2px;
		display: inline-block;
	}

	.sig-label {
		color: #edc948;
		font-weight: 700;
		font-size: 13px;
	}

	.interpretation {
		font-size: 12px;
		color: #aaa;
		font-style: italic;
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
