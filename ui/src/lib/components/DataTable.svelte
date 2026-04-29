<script lang="ts">
	interface Props {
		data: Record<string, unknown>[];
		maxRows?: number;
		sortKey?: string;
		sortDesc?: boolean;
	}

	let { data, maxRows = 200, sortKey = '', sortDesc = true }: Props = $props();

	let currentSort = $state(sortKey);
	let currentDesc = $state(sortDesc);

	function toggleSort(col: string) {
		if (currentSort === col) {
			currentDesc = !currentDesc;
		} else {
			currentSort = col;
			currentDesc = true;
		}
	}

	let columns = $derived(data.length ? Object.keys(data[0]) : []);

	let sorted = $derived.by(() => {
		if (!data.length) return [];
		let rows = [...data];
		if (currentSort && columns.includes(currentSort)) {
			rows.sort((a, b) => {
				const va = a[currentSort] as number;
				const vb = b[currentSort] as number;
				if (typeof va === 'number' && typeof vb === 'number') {
					return currentDesc ? vb - va : va - vb;
				}
				return currentDesc
					? String(vb).localeCompare(String(va))
					: String(va).localeCompare(String(vb));
			});
		}
		return rows.slice(0, maxRows);
	});

	function fmt(v: unknown): string {
		if (v === null || v === undefined) return '';
		if (typeof v === 'number') {
			if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(3);
			if (Math.abs(v) < 1) return v.toFixed(5);
			return v.toFixed(2);
		}
		return String(v);
	}

	function cellColor(col: string, v: unknown): string {
		if (typeof v !== 'number') return '';
		if (col.includes(' - ') || col === 'delta' || col === 'repression') {
			if (v > 0.001) return 'color: #59a14f';
			if (v < -0.001) return 'color: #e15759';
		}
		return '';
	}
</script>

<div class="table-wrap">
	<table>
		<thead>
			<tr>
				{#each columns as col}
					<th onclick={() => toggleSort(col)} class:sorted={currentSort === col}>
						{col}
						{#if currentSort === col}
							<span class="arrow">{currentDesc ? '▼' : '▲'}</span>
						{/if}
					</th>
				{/each}
			</tr>
		</thead>
		<tbody>
			{#each sorted as row}
				<tr>
					{#each columns as col}
						<td style={cellColor(col, row[col])}>{fmt(row[col])}</td>
					{/each}
				</tr>
			{/each}
		</tbody>
	</table>
</div>

<style>
	.table-wrap {
		overflow-x: auto;
		max-height: 600px;
		overflow-y: auto;
	}
	table {
		border-collapse: collapse;
		width: 100%;
		font-size: 12px;
		font-family: 'SF Mono', 'Cascadia Code', monospace;
	}
	th {
		position: sticky;
		top: 0;
		background: #1a1a2e;
		padding: 6px 10px;
		text-align: left;
		cursor: pointer;
		user-select: none;
		white-space: nowrap;
		border-bottom: 1px solid #333;
		color: #aaa;
		font-weight: 500;
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	th:hover {
		color: #ddd;
	}
	th.sorted {
		color: #e2b340;
	}
	.arrow {
		font-size: 8px;
		margin-left: 4px;
	}
	td {
		padding: 4px 10px;
		border-bottom: 1px solid #222;
		white-space: nowrap;
		color: #ccc;
	}
	tr:hover td {
		background: rgba(255, 255, 255, 0.04);
	}
	td:first-child {
		font-weight: 500;
		color: #e8e8e8;
	}
</style>
