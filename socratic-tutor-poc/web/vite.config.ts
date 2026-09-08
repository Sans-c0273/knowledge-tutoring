import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// FastAPI serves web/dist in production; in dev we proxy /api to the local backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // SSE passes through unbuffered; FastAPI must send `Cache-Control: no-cache`.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
  },
});
