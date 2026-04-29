import adapter from '@sveltejs/adapter-static';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	compilerOptions: {
		runes: ({ filename }) => (filename.split(/[/\\]/).includes('node_modules') ? undefined : true)
	},
	kit: {
		adapter: adapter({
			pages: '../malign_logits/ui_dist',
			assets: '../malign_logits/ui_dist',
			fallback: 'index.html'
		}),
		paths: {
			relative: true
		}
	}
};

export default config;
