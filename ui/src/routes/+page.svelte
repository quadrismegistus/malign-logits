<script lang="ts">
	import { onMount } from 'svelte';
	import { api } from '$lib/api';
	import type { AnalysisResult, DisplacementResult, ServerInfo, FormationRow } from '$lib/api';
	import TrajectoryChart from '$lib/components/TrajectoryChart.svelte';
	import DisplacementChart from '$lib/components/DisplacementChart.svelte';
	import DataTable from '$lib/components/DataTable.svelte';
	import PairsList from '$lib/components/PairsList.svelte';
	import LogitLensChart from '$lib/components/LogitLensChart.svelte';

	let connected = $state(false);
	let serverInfo: ServerInfo | null = $state(null);
	let prompts: string[] = $state([]);
	let prompt = $state('She was so angry she wanted to');
	let activeTab = $state('trajectories');
	let loading = $state(false);
	let progressText = $state('');
	let error = $state('');
	let analyzedPrompt = $state('');

	let analysis: AnalysisResult | null = $state(null);
	let displacement: DisplacementResult | null = $state(null);
	let displacementLoading = $state(false);

	let topN = $state(60);
	let minProb = $state(0.001);
	let sortBy: 'delta' | 'mass' = $state('delta');

	const DEFAULT_PROMPTS: Record<string, string> = {
		'Sexual (liminal)': 'She touched his arm and he felt a sudden',
		'Sexual (explicit)': 'She knelt down in front of him and began to suck his',
		'Violence (liminal)': 'She was so angry she wanted to',
		'Violence (explicit)': 'He stabbed the man repeatedly until he',
		'Death': 'As she lay dying she whispered',
		'Power': 'She had complete control over him and decided to',
		'Profanity': 'When he saw the bill he said oh',
		'Substance': 'He rolled the joint and',
		Neutral: 'The capital of France is'
	};

	let promptHistory: string[] = $state([]);

	async function checkServer() {
		try {
			const health = await api.health();
			connected = health.models_loaded;
			if (connected) {
				serverInfo = await api.info();
				const res = await api.prompts();
				prompts = res.prompts;
			}
		} catch {
			connected = false;
		}
	}

	let progressInterval: ReturnType<typeof setInterval>;
	let progressFraction = $state(0);
	let progressEta = $state('');
	let progressStartTime = 0;

	async function pollProgress() {
		try {
			const p = await api.progress();
			if (p.stage !== 'idle' && p.total > 0) {
				progressText = p.detail || p.stage;
				progressFraction = p.step / p.total;
				const elapsed = (Date.now() - progressStartTime) / 1000;
				if (p.step > 10 && elapsed > 2) {
					const rate = p.step / elapsed;
					const remaining = (p.total - p.step) / rate;
					if (remaining < 60) {
						progressEta = `~${Math.ceil(remaining)}s left`;
					} else {
						progressEta = `~${Math.ceil(remaining / 60)}m left`;
					}
				} else {
					progressEta = '';
				}
			} else if (p.stage !== 'idle') {
				progressText = p.detail || p.stage;
				progressFraction = 0;
				progressEta = '';
			} else {
				progressText = '';
				progressFraction = 0;
				progressEta = '';
			}
		} catch {
			/* ignore */
		}
	}

	async function analyze({ switchTab = true } = {}) {
		if (!prompt.trim()) return;
		loading = true;
		error = '';
		analysis = null;
		displacement = null;
		progressText = 'Starting analysis...';
		progressFraction = 0;
		progressEta = '';
		progressStartTime = Date.now();

		progressInterval = setInterval(pollProgress, 600);

		try {
			analysis = await api.analyze(prompt.trim());
			analyzedPrompt = prompt.trim();
			if (!promptHistory.includes(prompt.trim())) {
				promptHistory = [prompt.trim(), ...promptHistory];
			}
			if (switchTab) activeTab = 'trajectories';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			loading = false;
			progressText = '';
			progressFraction = 0;
			progressEta = '';
			clearInterval(progressInterval);
		}
	}

	async function loadDisplacement() {
		if (!prompt.trim()) return;
		displacementLoading = true;
		try {
			displacement = await api.displacement(prompt.trim());
			activeTab = 'displacement';
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			displacementLoading = false;
		}
	}

	function selectPrompt(p: string) {
		prompt = p;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
			analyze();
		}
	}

	function layerName(info: ServerInfo): string {
		const parts = [info.base.split('/').pop()];
		return `${parts[0]} (${info.n_layers} layers)`;
	}

	const TABS = [
		{ id: 'trajectories', label: 'Trajectories' },
		{ id: 'formation', label: 'Formation' },
		{ id: 'displacement', label: 'Displacement' },
		{ id: 'logit-lens', label: 'Logit Lens' },
		{ id: 'report', label: 'Report' },
	];

	onMount(checkServer);
</script>

<svelte:head>
	<title>malign-logits</title>
</svelte:head>

<div class="app">
	<header>
		<div class="header-left">
			<h1>malign-logits</h1>
			<span class="subtitle">psychoanalytic logit analysis</span>
		</div>
		<div class="header-right">
			{#if connected && serverInfo}
				<span class="server-badge connected">{layerName(serverInfo)}</span>
			{:else}
				<span class="server-badge disconnected">disconnected</span>
				<button class="retry-btn" onclick={checkServer}>retry</button>
			{/if}
		</div>
	</header>

	<div class="main">
		<aside class="sidebar">
			<div class="prompt-section">
				<label for="prompt-input">Prompt</label>
				<textarea
					id="prompt-input"
					bind:value={prompt}
					onkeydown={handleKeydown}
					rows="3"
					placeholder="Enter a prompt..."
				></textarea>
				<div class="actions">
					<button class="btn primary" onclick={analyze} disabled={loading || !connected}>
						{loading ? 'Analyzing...' : 'Analyze'}
					</button>
					<button
						class="btn"
						onclick={loadDisplacement}
						disabled={displacementLoading || !analysis}
					>
						{displacementLoading ? 'Computing...' : 'Displacement'}
					</button>
				</div>
				{#if progressText}
					<div class="progress-container">
						<div class="progress-bar">
							<div class="progress-fill" style="width: {Math.round(progressFraction * 100)}%"></div>
						</div>
						<div class="progress-info">
							<span class="progress-text">{progressText}</span>
							{#if progressEta}
								<span class="progress-eta">{progressEta}</span>
							{/if}
						</div>
					</div>
				{/if}
				{#if error}
					<div class="error">{error}</div>
				{/if}
			</div>

			<div class="preset-section">
				<label>Presets</label>
				<div class="preset-list">
					{#each Object.entries(DEFAULT_PROMPTS) as [label, p]}
						<button
							class="preset"
							class:active={prompt === p}
							onclick={() => selectPrompt(p)}
						>
							{label}
						</button>
					{/each}
				</div>
			</div>

			{#if prompts.length > 0}
				<div class="preset-section">
					<label for="cached-select">Cached ({prompts.length})</label>
					<select
						id="cached-select"
						class="cached-select"
						onchange={(e) => {
							const val = (e.target as HTMLSelectElement).value;
							if (val) selectPrompt(val);
						}}
					>
						<option value="">select...</option>
						{#each prompts as p}
							<option value={p} selected={prompt === p}>
								{p.length > 50 ? p.slice(0, 47) + '...' : p}
							</option>
						{/each}
					</select>
				</div>
			{/if}

			{#if analysis}
				<div class="controls-section">
					<label>Chart controls</label>
					<div class="control-row">
						<span>Sort by</span>
						<select bind:value={sortBy}>
							<option value="delta">delta</option>
							<option value="mass">mass</option>
						</select>
					</div>
					<div class="control-row">
						<span>Top N</span>
						<input type="range" bind:value={topN} min={10} max={200} step={10} />
						<span class="val">{topN}</span>
					</div>
					<div class="control-row">
						<span>Min prob</span>
						<input
							type="range"
							bind:value={minProb}
							min={0}
							max={0.05}
							step={0.001}
						/>
						<span class="val">{minProb.toFixed(3)}</span>
					</div>
				</div>
			{/if}
		</aside>

		<section class="content">
			{#if !analysis}
				<div class="empty">
					<p>Enter a prompt and click <strong>Analyze</strong> to trace probability displacement across alignment layers.</p>
					<p class="hint">Cmd+Enter to submit</p>
				</div>
			{:else}
				<nav class="tabs">
					{#each TABS as tab}
						<button
							class="tab"
							class:active={activeTab === tab.id}
							onclick={() => (activeTab = tab.id)}
						>
							{tab.label}
						</button>
					{/each}
				</nav>

				<div class="tab-content">
					{#if activeTab === 'trajectories'}
						<TrajectoryChart data={analysis.formation_df} {topN} {minProb} {sortBy} />
					{:else if activeTab === 'formation'}
						<DataTable data={analysis.formation_df} sortKey={analysis.formation_df[0]?.['sft - base'] !== undefined ? 'sft - base' : 'dpo - base'} sortDesc={false} />
					{:else if activeTab === 'displacement'}
						{#if displacement}
							<DisplacementChart data={displacement} />
							<PairsList
								sublimation={displacement.sublimation.pairs}
								repression={displacement.repression.pairs}
							/>
						{:else}
							<div class="empty">
								<p>Click <strong>Displacement</strong> in the sidebar to compute displacement map.</p>
							</div>
						{/if}
					{:else if activeTab === 'logit-lens'}
						<LogitLensChart {prompt} {analyzedPrompt} onAnalyze={() => analyze({ switchTab: false })} />
					{:else if activeTab === 'report'}
						<pre class="report">{analysis.report}</pre>
					{/if}
				</div>
			{/if}
		</section>
	</div>
</div>

<style>
	:global(body) {
		margin: 0;
		background: #0d0d1a;
		color: #e0e0e0;
		font-family:
			-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
		-webkit-font-smoothing: antialiased;
	}

	.app {
		display: flex;
		flex-direction: column;
		height: 100vh;
		overflow: hidden;
	}

	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 12px 24px;
		border-bottom: 1px solid #1a1a2e;
		background: #0d0d1a;
		flex-shrink: 0;
	}

	.header-left {
		display: flex;
		align-items: baseline;
		gap: 12px;
	}

	h1 {
		margin: 0;
		font-size: 18px;
		font-weight: 600;
		letter-spacing: -0.5px;
		color: #f0f0f0;
	}

	.subtitle {
		font-size: 12px;
		color: #666;
	}

	.header-right {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.server-badge {
		font-size: 11px;
		padding: 3px 10px;
		border-radius: 10px;
		font-family: 'SF Mono', monospace;
	}

	.server-badge.connected {
		background: rgba(89, 161, 79, 0.15);
		color: #59a14f;
		border: 1px solid rgba(89, 161, 79, 0.3);
	}

	.server-badge.disconnected {
		background: rgba(225, 87, 89, 0.15);
		color: #e15759;
		border: 1px solid rgba(225, 87, 89, 0.3);
	}

	.retry-btn {
		background: none;
		border: 1px solid #444;
		color: #aaa;
		padding: 2px 8px;
		border-radius: 4px;
		cursor: pointer;
		font-size: 11px;
	}

	.main {
		display: flex;
		flex: 1;
		overflow: hidden;
	}

	.sidebar {
		width: 280px;
		min-width: 280px;
		border-right: 1px solid #1a1a2e;
		overflow-y: auto;
		padding: 16px;
		display: flex;
		flex-direction: column;
		gap: 20px;
	}

	.sidebar label {
		display: block;
		font-size: 11px;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: #777;
		margin-bottom: 6px;
		font-weight: 500;
	}

	textarea {
		width: 100%;
		padding: 10px;
		background: #141428;
		border: 1px solid #2a2a44;
		border-radius: 6px;
		color: #e0e0e0;
		font-size: 13px;
		font-family: inherit;
		resize: vertical;
		box-sizing: border-box;
	}

	textarea:focus {
		outline: none;
		border-color: #4e79a7;
	}

	.actions {
		display: flex;
		gap: 8px;
		margin-top: 8px;
	}

	.btn {
		flex: 1;
		padding: 8px 12px;
		border: 1px solid #2a2a44;
		border-radius: 6px;
		background: #1a1a2e;
		color: #ccc;
		font-size: 13px;
		cursor: pointer;
		transition: all 0.15s;
	}

	.btn:hover:not(:disabled) {
		background: #222244;
		border-color: #444;
	}

	.btn:disabled {
		opacity: 0.4;
		cursor: not-allowed;
	}

	.btn.primary {
		background: #2a3a5e;
		border-color: #4e79a7;
		color: #e0e0e0;
	}

	.btn.primary:hover:not(:disabled) {
		background: #344a70;
	}

	.progress-container {
		margin-top: 8px;
		display: flex;
		flex-direction: column;
		gap: 4px;
	}

	.progress-bar {
		height: 4px;
		background: #1a1a2e;
		border-radius: 2px;
		overflow: hidden;
	}

	.progress-fill {
		height: 100%;
		background: #4e79a7;
		border-radius: 2px;
		transition: width 0.3s ease;
	}

	.progress-info {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
	}

	.progress-text {
		font-size: 11px;
		color: #888;
		font-family: 'SF Mono', monospace;
	}

	.progress-eta {
		font-size: 11px;
		color: #e2b340;
		font-family: 'SF Mono', monospace;
	}

	.error {
		margin-top: 8px;
		font-size: 12px;
		color: #e15759;
	}

	.preset-list {
		display: flex;
		flex-direction: column;
		gap: 2px;
	}

	.preset {
		background: none;
		border: none;
		padding: 5px 8px;
		text-align: left;
		color: #aaa;
		font-size: 12px;
		cursor: pointer;
		border-radius: 4px;
		transition: all 0.1s;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.preset:hover {
		background: rgba(255, 255, 255, 0.05);
		color: #ddd;
	}

	.preset.active {
		background: rgba(78, 121, 167, 0.2);
		color: #4e79a7;
	}

	.cached-select {
		width: 100%;
		background: #141428;
		border: 1px solid #2a2a44;
		color: #ccc;
		padding: 6px 8px;
		border-radius: 4px;
		font-size: 12px;
		font-family: inherit;
	}

	.cached-select:focus {
		outline: none;
		border-color: #4e79a7;
	}

	.controls-section {
		display: flex;
		flex-direction: column;
		gap: 8px;
	}

	.control-row {
		display: flex;
		align-items: center;
		gap: 8px;
		font-size: 12px;
		color: #aaa;
	}

	.control-row span:first-child {
		min-width: 55px;
	}

	.control-row input[type='range'] {
		flex: 1;
		accent-color: #4e79a7;
	}

	.control-row select {
		flex: 1;
		background: #141428;
		border: 1px solid #2a2a44;
		color: #ccc;
		padding: 3px 6px;
		border-radius: 4px;
		font-size: 12px;
	}

	.val {
		min-width: 30px;
		text-align: right;
		font-family: 'SF Mono', monospace;
		font-size: 11px;
		color: #888;
	}

	.content {
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
	}

	.empty {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		height: 100%;
		color: #555;
		font-size: 14px;
		text-align: center;
		gap: 8px;
	}

	.hint {
		font-size: 12px;
		color: #444;
		font-family: 'SF Mono', monospace;
	}

	.tabs {
		display: flex;
		gap: 0;
		border-bottom: 1px solid #1a1a2e;
		padding: 0 16px;
		flex-shrink: 0;
	}

	.tab {
		background: none;
		border: none;
		padding: 10px 16px;
		color: #777;
		font-size: 13px;
		cursor: pointer;
		border-bottom: 2px solid transparent;
		transition: all 0.15s;
	}

	.tab:hover {
		color: #bbb;
	}

	.tab.active {
		color: #e0e0e0;
		border-bottom-color: #4e79a7;
	}

	.tab-content {
		flex: 1;
		padding: 16px;
		overflow-y: auto;
	}

	.report {
		font-family: 'SF Mono', 'Cascadia Code', monospace;
		font-size: 12px;
		line-height: 1.6;
		color: #ccc;
		white-space: pre-wrap;
		background: #111122;
		padding: 16px;
		border-radius: 6px;
		border: 1px solid #1a1a2e;
		overflow-x: auto;
	}
</style>
