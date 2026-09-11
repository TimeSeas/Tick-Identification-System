<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getHistory, getHistoryDetail } from '../api/client'
import type { HistoryDetail, HistorySummary } from '../types'

// ===== 列表状态 =====
const loading = ref(false)
const items = ref<HistorySummary[]>([])
const total = ref(0)
const page = ref(1)
const size = 10

async function loadList() {
  loading.value = true
  try {
    const data = await getHistory(page.value, size)
    items.value = data.items
    total.value = data.total
  } finally {
    loading.value = false
  }
}

onMounted(loadList)   // 进入页面自动加载第一页

// ===== 详情弹窗 =====
const dialogVisible = ref(false)
const detailLoading = ref(false)
const detail = ref<HistoryDetail | null>(null)

async function showDetail(row: HistorySummary) {
  dialogVisible.value = true
  detailLoading.value = true
  detail.value = null
  try {
    detail.value = await getHistoryDetail(row.id)
  } finally {
    detailLoading.value = false
  }
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString()   // UTC ISO 字符串 → 本地时间显示
}
</script>

<template>
  <div class="page">
    <h3>分析历史</h3>
    <el-table :data="items" v-loading="loading" border stripe>
      <el-table-column prop="id" label="ID" width="70" />
      <el-table-column prop="filename" label="文件名" min-width="220" show-overflow-tooltip />
      <el-table-column prop="request_id" label="请求 ID" width="140" />
      <el-table-column label="分析时间" width="180">
        <template #default="{ row }">{{ formatTime(row.created_at) }}</template>
      </el-table-column>
      <el-table-column prop="detection_count" label="检出数" width="90" />
      <el-table-column label="操作" width="110">
        <template #default="{ row }">
          <el-button size="small" type="primary" plain @click="showDetail(row)">查看详情</el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 分页 -->
    <div style="margin-top: 12px; display: flex; justify-content: flex-end">
      <el-pagination
        v-model:current-page="page"
        :page-size="size"
        :total="total"
        layout="total, prev, pager, next"
        @current-change="loadList"
      />
    </div>

    <!-- 详情弹窗 -->
    <el-dialog v-model="dialogVisible" title="分析详情" width="760px" destroy-on-close>
      <div v-loading="detailLoading" style="min-height: 120px">
        <template v-if="detail">
          <img :src="detail.result_image_base64" style="max-width: 100%; max-height: 420px; display: block; margin: 0 auto" />
          <el-alert :title="detail.risk_advice" type="warning" :closable="false" style="margin-top: 12px" />
          <div style="margin-top: 12px">
            <el-tag v-for="(d, i) in detail.detections" :key="i" style="margin-right: 8px"
                    :type="d.status === 'confirmed' ? 'success' : 'warning'">
              {{ d.species_cn }} {{ d.confidence }}%
              {{ d.status === 'confirmed' ? '已确认' : '待确认' }}
            </el-tag>
          </div>
          <div style="margin-top: 8px; color: #909399; font-size: 13px">
            {{ detail.filename }} · 请求 {{ detail.request_id }} · {{ formatTime(detail.created_at) }}
          </div>
        </template>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.page {
  max-width: 1000px;
  margin: 0 auto;
}
</style>
