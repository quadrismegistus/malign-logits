<script lang="ts">
	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href="data:," />
</svelte:head>

{@render children()}
