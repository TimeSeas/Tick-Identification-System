/**
 * 前端路由：分析页（主页）与历史页
 * 
 */
import { createRouter, createWebHistory } from 'vue-router'
import AnalyzeView from '../views/AnalyzeView.vue'

const router = createRouter({
  history: createWebHistory(),   // HTML5 history 模式（生产环境需 nginx 配合）
  routes: [
    {
      path: '/',
      name: 'analyze',
      component: AnalyzeView,
      meta: { title: '蜱虫识别' },
    },
    {
      path: '/history',
      name: 'history',
      component: () => import('../views/HistoryView.vue'),  // 懒加载：访问时才下载该页代码
      meta: { title: '历史记录' },
    },
  ],
})

export default router
