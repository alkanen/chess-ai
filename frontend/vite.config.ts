/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs, so the same build works under any path prefix. The Python
  // server adds <base href="{prefix}/"> to index.html at runtime.
  base: './',
  build: {
    outDir: '../src/chess_ai/web/static',
    emptyOutDir: true,
  },
  server: {
    // For `npm run dev`: forward API calls to `chess-ai serve` running with an empty
    // path prefix.
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
  },
});
