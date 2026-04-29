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
	ego?: string;
	superego?: string;
	instruct?: string;
}

export interface FormationRow {
	word: string;
	base: number;
	ego?: number;
	superego?: number;
	instruct?: number;
	'ego - base'?: number;
	'superego - ego'?: number;
	'superego - base'?: number;
	'instruct - superego'?: number;
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

export const api = {
	health: () => get<{ status: string; models_loaded: boolean }>('/health'),
	info: () => get<ServerInfo>('/info'),
	progress: () => get<Progress>('/progress'),
	prompts: () => get<{ prompts: string[] }>('/prompts'),
	analyze: (prompt: string, top_k = 200) => post<AnalysisResult>('/analyze', { prompt, top_k }),
	displacement: (prompt: string, layers?: number[]) =>
		post<DisplacementResult>('/displacement_map', { prompt, ...(layers ? { layers } : {}) }),
	topWords: (layer: string, prompt: string, top_k = 200) =>
		post<{ words: Record<string, number> }>('/top_words', { layer, prompt, top_k }),
	perplexity: (layer: string, prompt: string) =>
		post<{ perplexity: number }>('/perplexity', { layer, prompt })
};
