/**
 * Export a D3-rendered SVG from a container element as a high-res PNG.
 */
export function exportPng(container: HTMLElement, filename = 'chart.png', scale = 4) {
	const svg = container.querySelector('svg');
	if (!svg) return;

	const clone = svg.cloneNode(true) as SVGSVGElement;

	clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
	clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');

	// Inline computed styles so the rasterized version matches the screen
	const original = svg.querySelectorAll('*');
	const cloned = clone.querySelectorAll('*');
	for (let i = 0; i < original.length; i++) {
		const computed = getComputedStyle(original[i]);
		const el = cloned[i] as SVGElement;
		for (const prop of ['fill', 'stroke', 'stroke-width', 'opacity', 'font-size',
			'font-family', 'font-weight', 'text-anchor', 'dominant-baseline']) {
			const val = computed.getPropertyValue(prop);
			if (val) el.style.setProperty(prop, val);
		}
	}

	const w = svg.width.baseVal.value || svg.getBoundingClientRect().width;
	const h = svg.height.baseVal.value || svg.getBoundingClientRect().height;

	const blob = new Blob([new XMLSerializer().serializeToString(clone)], {
		type: 'image/svg+xml;charset=utf-8'
	});
	const url = URL.createObjectURL(blob);
	const img = new Image();

	img.onload = () => {
		const canvas = document.createElement('canvas');
		canvas.width = w * scale;
		canvas.height = h * scale;
		const ctx = canvas.getContext('2d')!;
		ctx.fillStyle = '#0a0a1a';
		ctx.fillRect(0, 0, canvas.width, canvas.height);
		ctx.scale(scale, scale);
		ctx.drawImage(img, 0, 0, w, h);
		URL.revokeObjectURL(url);

		canvas.toBlob((b) => {
			if (!b) return;
			const a = document.createElement('a');
			a.href = URL.createObjectURL(b);
			a.download = filename;
			a.click();
			URL.revokeObjectURL(a.href);
		}, 'image/png');
	};

	img.src = url;
}
