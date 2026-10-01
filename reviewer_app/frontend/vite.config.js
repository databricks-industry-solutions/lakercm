import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },

  // Offline component tests. jsdom, not a browser: no Databricks, no Lakebase, no
  // workspace credentials, and nothing to download on a CI runner.
  //
  // Why not Playwright, which was the first choice. This dev image is Ubuntu
  // 20.04, which Playwright dropped ("does not support chromium on
  // ubuntu20.04-x64"), and the last release that still supports 20.04 hangs
  // silently in its ESM loader on Node 24 — verified by bisecting down to a
  // one-line spec with no webServer. That leaves only "author browser tests you
  // cannot execute and iterate through CI", which is how a broken test stub
  // reached CI earlier in this work. vitest runs here, so these are verified
  // before they are pushed.
  //
  // The honest limit: jsdom has no layout engine. getComputedStyle returns
  // declared values, never computed geometry, so the side-by-side review split
  // and the assistant gutter WIDTH cannot be asserted here — only that the
  // classes and elements that drive them are present. Real geometry needs a
  // browser on a supported OS.
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.js'],
    css: false,
    include: ['src/**/*.test.{js,jsx}'],
    restoreMocks: true,
    clearMocks: true,
  },
})
