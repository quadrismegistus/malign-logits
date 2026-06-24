<script lang="ts">
	import { onMount } from 'svelte';

	interface ShiftRow {
		family: string;
		token: string;
		base: number;
		aligned: number;
		delta: number;
	}

	let rows: ShiftRow[] = $state([]);
	let loading = $state(true);
	let error = $state('');
	let selectedFamily = $state('all');
	let sortBy: 'delta' | 'base' | 'alpha' = $state('delta');
	let showDelta = $state(true);

	let families = $derived([...new Set(rows.map(r => r.family))].sort());
	let tokens = $derived([...new Set(rows.map(r => r.token))].sort());

	let filtered = $derived.by(() => {
		let f = selectedFamily === 'all' ? rows : rows.filter(r => r.family === selectedFamily);
		if (sortBy === 'delta') {
			f = [...f].sort((a, b) => a.delta - b.delta);
		} else if (sortBy === 'base') {
			f = [...f].sort((a, b) => b.base - a.base);
		} else {
			f = [...f].sort((a, b) => a.token.localeCompare(b.token));
		}
		return f;
	});

	let maxProb = $derived(Math.max(...rows.map(r => Math.max(r.base, r.aligned)), 0.001));

	function barWidth(val: number): string {
		return `${Math.max(1, (val / maxProb) * 100)}%`;
	}

	function deltaColor(delta: number): string {
		if (delta < -0.001) return '#e15759';
		if (delta > 0.001) return '#59a14f';
		return '#888';
	}

	async function loadData() {
		try {
			const res = await fetch('/api/api/data/csv?name=f21_token_shifts_multi');
			let data;
			if (!res.ok) {
				const res2 = await fetch('/api/data/csv?name=f21_token_shifts_multi');
				if (!res2.ok) throw new Error('Failed to load token shifts data');
				data = await res2.json();
			} else {
				data = await res.json();
			}
			rows = data.rows;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
		}
	}

	onMount(loadData);
</script>

<div class="token-shifts">
	<div class="controls">
		<label class="control">
			<span>Family</span>
			<select bind:value={selectedFamily}>
				<option value="all">all families</option>
				{#each families as f}
					<option value={f}>{f}</option>
				{/each}
			</select>
		</label>
		<label class="control">
			<span>Sort</span>
			<select bind:value={sortBy}>
				<option value="delta">displacement (delta)</option>
				<option value="base">base probability</option>
				<option value="alpha">alphabetical</option>
			</select>
		</label>
		<label class="control toggle">
			<input type="checkbox" bind:checked={showDelta} />
			<span>show delta</span>
		</label>
	</div>

	{#if loading}
		<div class="loading">Loading token shifts...</div>
	{:else if error}
		<div class="error">{error}</div>
	{:else if selectedFamily === 'all'}
		<div class="summary">{rows.length} shifts across {families.length} families, {tokens.length} tokens</div>
		<div class="grid-view">
			<table class="shift-grid">
				<thead>
					<tr>
						<th class="token-header">token</th>
						{#each families as f}
							<th class="family-header">{f}</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each tokens as tok}
						<tr>
							<td class="token-cell">{tok}</td>
							{#each families as fam}
								{@const row = rows.find(r => r.family === fam && r.token === tok)}
								{#if row}
									<td class="shift-cell">
										<div class="bar-pair">
											<div class="bar base-bar" style="width: {barWidth(row.base)}"></div>
											<div class="bar aligned-bar" style="width: {barWidth(row.aligned)}"></div>
										</div>
										{#if showDelta}
											<span class="delta" style="color: {deltaColor(row.delta)}">
												{row.delta > 0 ? '+' : ''}{(row.delta * 100).toFixed(2)}%
											</span>
										{/if}
									</td>
								{:else}
									<td class="shift-cell empty">—</td>
								{/if}
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<div class="summary">{filtered.length} tokens for {selectedFamily}</div>
		<div class="list-view">
			{#each filtered as row}
				<div class="shift-row">
					<span class="row-token">{row.token}</span>
					<div class="row-bars">
						<div class="bar-container">
							<span class="bar-label">base</span>
							<div class="bar base-bar" style="width: {barWidth(row.base)}"></div>
							<span class="bar-val">{(row.base * 100).toFixed(2)}%</span>
						</div>
						<div class="bar-container">
							<span class="bar-label">aligned</span>
							<div class="bar aligned-bar" style="width: {barWidth(row.aligned)}"></div>
							<span class="bar-val">{(row.aligned * 100).toFixed(2)}%</span>
						</div>
					</div>
					<span class="row-delta" style="color: {deltaColor(row.delta)}">
						{row.delta > 0 ? '+' : ''}{(row.delta * 100).toFixed(3)}%
					</span>
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.token-shifts {
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

	.toggle {
		flex-direction: row;
		align-items: center;
		gap: 6px;
		cursor: pointer;
	}

	.toggle input {
		accent-color: #4e79a7;
	}

	.summary {
		font-size: 12px;
		color: #888;
		border-bottom: 1px solid #1a1a2e;
		padding-bottom: 8px;
	}

	.grid-view {
		overflow-x: auto;
	}

	.shift-grid {
		border-collapse: separate;
		border-spacing: 2px;
		width: 100%;
	}

	.token-header {
		text-align: left;
		font-size: 12px;
		color: #aaa;
		font-weight: 500;
		padding: 4px 8px;
		min-width: 70px;
	}

	.family-header {
		text-align: center;
		font-size: 11px;
		color: #aaa;
		font-weight: 500;
		padding: 4px 8px;
		min-width: 100px;
	}

	.token-cell {
		font-family: 'SF Mono', monospace;
		font-size: 12px;
		color: #ccc;
		padding: 4px 8px;
		font-weight: 500;
	}

	.shift-cell {
		padding: 3px 6px;
		vertical-align: middle;
	}

	.shift-cell.empty {
		color: #444;
		text-align: center;
	}

	.bar-pair {
		display: flex;
		flex-direction: column;
		gap: 1px;
	}

	.bar {
		height: 6px;
		border-radius: 2px;
		min-width: 1px;
	}

	.base-bar {
		background: #4e79a7;
	}

	.aligned-bar {
		background: #e15759;
	}

	.delta {
		font-family: 'SF Mono', monospace;
		font-size: 10px;
	}

	.list-view {
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.shift-row {
		display: flex;
		align-items: center;
		gap: 16px;
		padding: 6px 8px;
		border-radius: 4px;
	}

	.shift-row:hover {
		background: rgba(255, 255, 255, 0.03);
	}

	.row-token {
		font-family: 'SF Mono', monospace;
		font-size: 13px;
		color: #ccc;
		font-weight: 500;
		min-width: 80px;
	}

	.row-bars {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.bar-container {
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.bar-label {
		font-size: 10px;
		color: #666;
		min-width: 42px;
		text-align: right;
	}

	.bar-val {
		font-family: 'SF Mono', monospace;
		font-size: 10px;
		color: #888;
		min-width: 50px;
	}

	.row-delta {
		font-family: 'SF Mono', monospace;
		font-size: 12px;
		font-weight: 500;
		min-width: 70px;
		text-align: right;
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
