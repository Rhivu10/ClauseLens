import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'
import clauseLensApi from './server/api.js'

export default defineConfig({
  plugins: [react(), tailwindcss(), clauseLensApi()],
  resolve: {
    alias: { '@': '/src' },
  },
  server: {
    // uploads and downloaded reports are not app code
    watch: { ignored: ['**/server/work/**', '**/server/data/**'] },
  },
})
