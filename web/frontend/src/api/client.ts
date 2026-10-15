/**
 * API 层：axios 实例 + 统一响应解包 + 错误提示
 * 所有请求走这个文件，页面组件不直接碰 axios
 */
import axios from 'axios'
import { ElMessage } from 'element-plus'
import type { AnalyzeResult, ApiResponse, HistoryDetail, HistoryListData } from '../types'

const http = axios.create({
  baseURL: '/api/v1',        // 开发环境由 Vite 代理转发到后端 8000
  timeout: 30000,
})

// 响应拦截器：解包统一响应 {code, msg, data}
http.interceptors.response.use(
  (resp) => {
    const body = resp.data as ApiResponse
    if (body.code !== 200) {                       // 后端约定的成功码是 200
      ElMessage.error(body.msg || '请求失败')
      return Promise.reject(new Error(body.msg))
    }
    return body.data as never                  // 直接返回 data，调用方拿到干净数据
  },
  (error) => {
    const message = error.code === 'ECONNABORTED' || error.code === 'ETIMEDOUT'
      ? '分析等待时间较长，请先查看识别档案确认是否完成，再尝试重试。'
      : error.response?.data?.msg || '无法连接识别服务，请检查网络后重试。'
    ElMessage.error(message)
    return Promise.reject(new Error(message))
  },
)

/** 上传图片做蜱虫分析 */
export async function analyzeImage(file: File): Promise<AnalyzeResult> {
  const form = new FormData()
  form.append('file', file)                     // 字段名与后端 UploadFile 参数一致
  // 注意：FormData 上传不能手动设 Content-Type，axios 会自动带 boundary
  const resp = await http.post('/analyze', form, { timeout: 120000 })
  return resp as unknown as AnalyzeResult
}

/** 历史列表（分页） */
export async function getHistory(page = 1, size = 10): Promise<HistoryListData> {
  const resp = await http.get('/history', { params: { page, size } })
  return resp as unknown as HistoryListData
}

/** 历史详情（含标注图与完整检测结果） */
export async function getHistoryDetail(id: number): Promise<HistoryDetail> {
  const resp = await http.get(`/history/${id}`)
  return resp as unknown as HistoryDetail
}
