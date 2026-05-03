<script lang="ts">
	import { exportPng } from '$lib/export';

	interface Props {
		container: HTMLDivElement | undefined;
		filename?: string;
	}

	let { container, filename = 'chart.png' }: Props = $props();
</script>

<button
	class="export-btn"
	title="Save as PNG (300dpi)"
	onclick={() => container && exportPng(container, filename)}
>
	PNG
</button>

<style>
	.export-btn {
		padding: 4px 10px;
		border: 1px solid #444;
		border-radius: 4px;
		background: #1a1a2e;
		color: #888;
		font-size: 11px;
		font-family: 'SF Mono', monospace;
		cursor: pointer;
		transition: all 0.15s;
	}

	.export-btn:hover {
		border-color: #4e79a7;
		color: #ccc;
	}
</style>
