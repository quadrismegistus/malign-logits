const BASE = import.meta.env.DEV ? '/api' : '';

async function post<T>(path: string, body: Record<string, unknown> = {}): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) throw new Error(`${path}: ${res.status}`);
	return res.json();
}

async function get<T>(path: string): Promise<T> {
	const res = await fetch(`${BASE}${path}`);
	if (!res.ok) throw new Error(`${path}: ${res.status}`);
	return res.json();
}

export interface ServerInfo {
	base: string;
	n_layers: number;
	sft?: string;
	dpo?: string;
	rlvr?: string;
}

export interface FormationRow {
	word: string;
	base: number;
	sft?: number;
	dpo?: number;
	rlvr?: number;
	'sft - base'?: number;
	'dpo - sft'?: number;
	'dpo - base'?: number;
	'rlvr - dpo'?: number;
	trajectory: string;
	[key: string]: unknown;
}

export interface RepressionRow {
	word: string;
	[key: string]: unknown;
}

export interface AnalysisResult {
	status: string;
	layers: string[];
	report: string;
	formation_df: FormationRow[];
	repression_df: RepressionRow[];
}

export interface DisplacementPair {
	source: string;
	target: string;
	sim: number;
	layer: string;
}

export interface DisplacementResult {
	sublimation: {
		source: string[];
		target: string[];
		pairs: [string, string, number, string][];
	};
	repression: {
		source: string[];
		target: string[];
		pairs: [string, string, number, string][];
	};
	df: FormationRow[];
}

export interface Progress {
	stage: string;
	detail: string;
	step: number;
	total: number;
}

export interface LogitLensRow {
	layer: number;
	word: string;
	probability: number;
	source: string;
	model: string;
}

export interface LogitLensResult {
	rows: LogitLensRow[];
	word_sources: Record<string, string[]>;
}

export interface Generation {
	model: string;
	text: string;
	gen_id: number;
	pca_x: number;
	pca_y: number;
	violent: number;
	sexual: number;
	compliant: number;
}

export interface GenerateResult {
	generations: Generation[];
	concept_axes: string[];
	pca_variance: number[];
}

export interface ContestedWord {
	word: string;
	prob_a: number;
	prob_b: number;
	prob_ab: number;
	prob_mean: number;
}

export interface ContradictionResult {
	pair: string;
	prompt_a: string;
	prompt_b: string;
	prompt_ab: string;
	model: string;
	superposition: number;
	resolution: number;
	ratio: number;
	contested_words: ContestedWord[];
}

export interface PassageMetrics {
	family: string;
	label: string;
	model: string;
	psg: string;
	prompt?: string;
	n_sentences: number;
	mean_drift: number;
	total_drift: number;
	directedness: number;
	mean_surprisal: number;
	surprisal_llama?: number;
	surprisal_mistral?: number;
	metonymy_idx: number;
	token_diameter: number;
	token_mean_drift: number;
	token_directedness: number;
	token_metonymy_idx: number;
	n_tokens: number;
	is_template?: boolean;
	genre_type?: string;
	[key: string]: unknown;
}

export interface BeamAnnotation {
	token_resist: number[];
	total_resist: number;
	mean_resist: number;
}

export interface BeamStoryline {
	rank: number;
	text: string;
	tokens: string[];
	path_prob: number;
	log_prob: number;
	base_token_probs?: number[];
	annotations?: Record<string, BeamAnnotation>;
}

export interface BeamIndex {
	models: string[];
	prompts: string[];
	sources: Record<string, string[]>;
	nicknames: Record<string, string>;
	source_nicknames: Record<string, string>;
}

export const api = {
	health: () => get<{ status: string; models_loaded: boolean; data_only?: boolean }>('/health'),
	info: () => get<ServerInfo>('/info'),
	progress: () => get<Progress>('/progress'),
	prompts: () => get<{ prompts: string[] }>('/prompts'),
	analyze: (prompt: string, top_k = 200) => post<AnalysisResult>('/analyze', { prompt, top_k }),
	displacement: (prompt: string, layers?: number[]) =>
		post<DisplacementResult>('/displacement_map', { prompt, ...(layers ? { layers } : {}) }),
	topWords: (layer: string, prompt: string, top_k = 200) =>
		post<{ words: Record<string, number> }>('/top_words', { layer, prompt, top_k }),
	perplexity: (layer: string, prompt: string) =>
		post<{ perplexity: number }>('/perplexity', { layer, prompt }),
	logitLens: (prompt: string) => post<LogitLensResult>('/logit_lens', { prompt }),
	generate: (prompt: string, n = 5, max_tokens = 100, temperature = 1.0) =>
		post<GenerateResult>('/generate', { prompt, n, max_tokens, temperature }),
	contradiction: () =>
		post<{ results: ContradictionResult[] }>('/contradiction', {}),
	passageMetricsCsv: () =>
		post<{ rows: PassageMetrics[] }>('/passage-metrics-csv', {}),
	passageMetrics: (text: string) =>
		post<PassageMetrics>('/passage-metrics', { text }),
	passageTokens: (psg: string, prompt = '', model_id = '', gen_prompt = '', idx = 0) =>
		post<{ tokens: [string, number][]; sentences?: { drift: number; tokens: [string, number][] }[] }>('/passage-tokens', { psg, prompt, model_id, gen_prompt, idx }),
	beamIndex: () => get<BeamIndex>('/api/beam/index'),
	beamStorylines: (model: string, prompt: string, n = 50, source = '') =>
		get<{ storylines: BeamStoryline[]; model: string; prompt: string }>(
			`/api/beam/storylines?model=${encodeURIComponent(model)}&prompt=${encodeURIComponent(prompt)}&n=${n}${source ? `&source=${encodeURIComponent(source)}` : ''}`
		),
};
