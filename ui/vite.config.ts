import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const API_PORT = process.env.API_PORT || '8421';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		host: '0.0.0.0',
		proxy: {
			'/api': {
				target: `http://127.0.0.1:${API_PORT}`,
				rewrite: (path) => path.replace(/^\/api/, '')
			}
		}
	}
});
