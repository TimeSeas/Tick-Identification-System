import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
  proxy: {
    '/api':'http://localhost:8000'   // 前端所有 /api 请求转发到后端
  }
},
  resolve:{
    alias:{
      '@':path.resolve(__dirname,'src')
    }
  }
})


