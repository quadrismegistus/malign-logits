<script lang="ts">
	import DataTable from './DataTable.svelte';

	const API_BASE = import.meta.env.DEV ? '/api' : '';

	const DATASETS = [
		{ id: 'displacement_agreement', label: 'Displacement Agreement (7071 rows)', limit: 7100 },
		{ id: 'bits_resistance', label: 'Bits of Resistance (32K rows)', limit: 5000 },
		{ id: 'logit_lens_datadriven', label: 'Logit Lens (405K rows)', limit: 5000 },
		{ id: 'system_prompt_effect', label: 'System Prompt Effect (286 rows)', limit: 500 },
		{ id: 'cross_family_resistance', label: 'Cross-Family Resistance (578 rows)', limit: 1000 },
		{ id: 'category_displacement', label: 'Category Displacement (732 rows)', limit: 1000 },
		{ id: 'word_annotations', label: 'Word Annotations (2975 rows)', limit: 3000 },
		{ id: 'resistance_trajectories', label: 'Resistance Trajectories (7586 rows)', limit: 8000 },
	];

	let selectedDataset = $state(DATASETS[0].id);
	let data: Record<string, unknown>[] = $state([]);
	let totalRows = $state(0);
	let columns: string[] = $state([]);
	let loading = $state(false);
	let error = $state('');

	// Filters
	let filterCategory = $state('all');
	let filterDirection = $state('all');
	let filterFamily = $state('');
	let filterWord = $state('');
	let minAgreement = $state(0);
	let minBits = $state(0);

	let categories = $derived([...new Set(data.map(r => r.category).filter(Boolean))].sort() as string[]);
	let directions = $derived([...new Set(data.map(r => r.direction).filter(Boolean))].sort() as string[]);

	let filtered = $derived.by(() => {
		let rows = data;
		if (filterCategory !== 'all') rows = rows.filter(r => r.category === filterCategory);
		if (filterDirection !== 'all') rows = rows.filter(r => r.direction === filterDirection);
		if (filterFamily) rows = rows.filter(r => String(r.family || r.prompt_key || '').toLowerCase().includes(filterFamily.toLowerCase()));
		if (filterWord) rows = rows.filter(r => String(r.word || '').toLowerCase().includes(filterWord.toLowerCase()));
		if (minAgreement > 0) {
			rows = rows.filter(r => {
				const af = Number(r.agreement_filtered || r.agreement_all || r.agreement || 0);
				return af >= minAgreement;
			});
		}
		if (minBits > 0) {
			rows = rows.filter(r => {
				const b = Math.abs(Number(r.mean_bits_filtered || r.mean_bits || r.bits_resistance || 0));
				return b >= minBits;
			});
		}
		return rows;
	});

	async function loadDataset() {
		const ds = DATASETS.find(d => d.id === selectedDataset);
		if (!ds) return;
		loading = true;
		error = '';
		try {
			const res = await fetch(`${API_BASE}/data/csv?name=${ds.id}&limit=${ds.limit}`);
			const json = await res.json();
			data = json.rows || [];
			totalRows = json.total || data.length;
			columns = json.columns || [];
		} catch (e) {
			error = String(e);
			data = [];
		}
		loading = false;
	}

	$effect(() => {
		loadDataset();
	});

	function resetFilters() {
		filterCategory = 'all';
		filterDirection = 'all';
		filterFamily = '';
		filterWord = '';
		minAgreement = 0;
		minBits = 0;
	}
</script>

<div class="explorer">
	<div class="controls">
		<div class="control-row">
			<label>
				Dataset:
				<select bind:value={selectedDataset} onchange={loadDataset}>
					{#each DATASETS as ds}
						<option value={ds.id}>{ds.label}</option>
					{/each}
				</select>
			</label>
			<span class="count">
				{#if loading}Loading...{:else}{filtered.length} / {totalRows} rows{/if}
			</span>
		</div>

		<div class="filter-row">
			{#if categories.length}
				<label>Category:
					<select bind:value={filterCategory}>
						<option value="all">all</option>
						{#each categories as cat}
							<option value={cat}>{cat}</option>
						{/each}
					</select>
				</label>
			{/if}

			{#if directions.length}
				<label>Direction:
					<select bind:value={filterDirection}>
						<option value="all">all</option>
						{#each directions as dir}
							<option value={dir}>{dir}</option>
						{/each}
					</select>
				</label>
			{/if}

			<label>Word: <input type="text" bind:value={filterWord} placeholder="filter..." /></label>
			<label>Family: <input type="text" bind:value={filterFamily} placeholder="filter..." /></label>

			<label>Min agreement:
				<input type="number" bind:value={minAgreement} min="0" max="100" step="5" style="width:60px" />%
			</label>

			<label>Min |bits|:
				<input type="number" bind:value={minBits} min="0" step="0.5" style="width:60px" />
			</label>

			<button onclick={resetFilters}>Reset</button>
		</div>
	</div>

	{#if error}
		<div class="error">{error}</div>
	{:else if !loading}
		<DataTable data={filtered} maxRows={500} sortKey={columns.includes('agreement_filtered') ? 'agreement_filtered' : columns.includes('mean_bits') ? 'mean_bits' : ''} />
	{/if}
</div>

<style>
	.explorer {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}
	.controls {
		display: flex;
		flex-direction: column;
		gap: 8px;
		padding: 12px;
		background: rgba(255,255,255,0.03);
		border-radius: 8px;
	}
	.control-row, .filter-row {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-wrap: wrap;
	}
	.count {
		color: #888;
		font-size: 13px;
	}
	label {
		font-size: 12px;
		color: #aaa;
		display: flex;
		align-items: center;
		gap: 4px;
	}
	select, input[type="text"], input[type="number"] {
		background: #1a1a2e;
		color: #ddd;
		border: 1px solid #333;
		border-radius: 4px;
		padding: 4px 8px;
		font-size: 12px;
	}
	button {
		background: #333;
		color: #ccc;
		border: 1px solid #444;
		border-radius: 4px;
		padding: 4px 12px;
		font-size: 12px;
		cursor: pointer;
	}
	button:hover { background: #444; }
	.error { color: #e15759; padding: 8px; }
</style>
