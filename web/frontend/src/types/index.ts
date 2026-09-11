/**
 * 全局 TS 类型定义：与后端 app/schemas.py 的响应结构一一对应
 * 后端统一响应包 {code, msg, data}，code === 200 表示成功
 */

export interface ApiResponse<T = unknown> {
  code: number
  msg: string
  data: T
}

/** 历史列表单条摘要（后端 /history 列表接口） */
export interface HistorySummary {
  id: number
  filename: string
  request_id: string
  created_at: string
  detection_count: number
}

/** 历史列表响应 data */
export interface HistoryListData {
  items: HistorySummary[]
  total: number
  page: number
  size: number
}

/** 历史详情（后端 /history/{id} 接口）：完整 AnalyzeResult + 记录元信息 */
export interface HistoryDetail extends AnalyzeResult {
  record_id: number
  filename: string
  created_at: string
}

/** 单个蜱虫检测结果 */
export interface Detection {
  bbox: [number, number, number, number]  // [x1, y1, x2, y2]
  score: number                            // M1 检测置信度
  species_slug: string                     // 学名 slug；uncertain 时为 "uncertain"
  species_cn: string                       // 中文名
  confidence: number                       // M2 分类置信度（0-100）
  status: 'confirmed' | 'uncertain'        // 分物种阈值决策结果
}

/** POST /api/v1/analyze 返回的 data 字段 */
export interface AnalyzeResult {
  request_id: string
  detections: Detection[]
  risk_advice: string
  result_image_base64: string   // data:image/jpeg;base64,xxx 可直接用于 <img :src>
}
