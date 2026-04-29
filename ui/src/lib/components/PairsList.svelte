<script lang="ts">
	interface Props {
		sublimation: [string, string, number, string][];
		repression: [string, string, number, string][];
		minSim?: number;
	}

	let { sublimation = [], repression = [], minSim = 0.4 }: Props = $props();

	function best(
		pairs: [string, string, number, string][]
	): { source: string; target: string; sim: number; layer: string }[] {
		const map = new Map<string, { source: string; target: string; sim: number; layer: string }>();
		for (const [source, target, sim, layer] of pairs) {
			if (sim < minSim) continue;
			const key = `${source}->${target}`;
			if (!map.has(key) || sim > map.get(key)!.sim) {
				map.set(key, { source, target, sim, layer });
			}
		}
		return [...map.values()].sort((a, b) => b.sim - a.sim);
	}
</script>

<div class="pairs">
	{#if sublimation.length > 0}
		<div class="section">
			<h4>Sublimation (base → ego)</h4>
			<div class="pair-list">
				{#each best(sublimation) as pair}
					<div class="pair">
						<span class="source">{pair.source}</span>
						<span class="arrow">→</span>
						<span class="target">{pair.target}</span>
						<span class="sim">{pair.sim.toFixed(3)}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	{#if repression.length > 0}
		<div class="section">
			<h4>Repression (ego → superego)</h4>
			<div class="pair-list">
				{#each best(repression) as pair}
					<div class="pair">
						<span class="source">{pair.source}</span>
						<span class="arrow">→</span>
						<span class="target">{pair.target}</span>
						<span class="sim">{pair.sim.toFixed(3)}</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.pairs {
		display: flex;
		flex-direction: column;
		gap: 24px;
	}
	h4 {
		margin: 0 0 8px 0;
		color: #aaa;
		font-size: 13px;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}
	.pair-list {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}
	.pair {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 4px 8px;
		border-radius: 3px;
		font-size: 13px;
		font-family: 'SF Mono', monospace;
	}
	.pair:hover {
		background: rgba(255, 255, 255, 0.05);
	}
	.source {
		color: #e15759;
		min-width: 120px;
		text-align: right;
	}
	.arrow {
		color: #555;
	}
	.target {
		color: #59a14f;
		min-width: 120px;
	}
	.sim {
		color: #888;
		margin-left: auto;
		font-size: 11px;
	}
</style>
