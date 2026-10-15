<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { getHistory, getHistoryDetail } from '../api/client'
import type { HistoryDetail, HistorySummary } from '../types'
const loading = ref(false), error = ref(''), page = ref(1), total = ref(0)
const items = ref<HistorySummary[]>([])
const size = 10
const dialogVisible = ref(false), detailLoading = ref(false), detailError = ref('')
const detail = ref<HistoryDetail | null>(null)
async function loadList() {
  loading.value = true; error.value = ''
  try { const data = await getHistory(page.value, size); items.value = data.items; total.value = data.total }
  catch { error.value = '档案暂时无法加载，请重试。' }
  finally { loading.value = false }
}
async function showDetail(row: HistorySummary) {
  dialogVisible.value = true; detailLoading.value = true; detail.value = null; detailError.value = ''
  try { detail.value = await getHistoryDetail(row.id) }
  catch { detailError.value = '本条记录暂时无法读取，请关闭后重试。' }
  finally { detailLoading.value = false }
}
function formatTime(iso: string) {
  // 接口返回的历史时间没有后缀，数据库按UTC存储。
  const utc = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(iso) ? iso : `${iso}Z`
  return new Date(utc).toLocaleString('zh-CN', { hour12: false })
}
onMounted(loadList)
</script>
<template>
  <section class="page-intro history-intro">
    <div>
      <p class="eyebrow">
        OBSERVATION ARCHIVE / 识别档案
      </p>
      <h1>
        每一次识别，
        <br />
        都有迹可循
        <span>
          。
        </span>
      </h1>
      <p class="intro-copy">
        回看图像、定位结果与蜱种结论，留存完整观察记录。
      </p>
    </div>
    <div class="archive-total">
      <strong>
        {{ total.toString().padStart(2, '0') }}
      </strong>
      <span>
        条识别记录
      </span>
    </div>
  </section>
  <section class="panel archive-panel">
    <div class="panel-heading">
      <h2>
        全部记录
      </h2>
      <button class="button subtle" :disabled="loading" @click="loadList">
        {{ loading ? '正在加载…' : '刷新记录 ↻' }}
      </button>
    </div>
    <p v-if="error" class="inline-error" role="alert">
      {{ error }}
      <button class="text-button" @click="loadList">
        重新加载
      </button>
    </p>
    <div class="table-scroll">
      <table class="archive-table" :aria-busy="loading">
        <thead>
          <tr>
            <th>
              记录 / 编号
            </th>
            <th>
              图像名称
            </th>
            <th>
              识别时间
            </th>
            <th>
              检出实例
            </th>
            <th>
              <span class="visually-hidden">
                操作
              </span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in items" :key="row.id">
            <td>
              <span class="record-number">
                #{{ String(row.id).padStart(3, '0') }}
              </span>
              <small class="record-request">
                {{ row.request_id }}
              </small>
            </td>
            <td class="filename-cell">
              {{ row.filename }}
            </td>
            <td>
              {{ formatTime(row.created_at) }}
            </td>
            <td>
              <span class="count-pill" :class="{ empty: !row.detection_count }">
                {{ row.detection_count }} 只
              </span>
            </td>
            <td>
              <button class="text-button" @click="showDetail(row)">
                查看详情 ↗
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-if="!items.length && !loading && !error" class="archive-empty">
        <span>
          ⌑
        </span>
        <h3>
          还没有识别记录
        </h3>
        <p>
          完成一次图像识别后，结果会保存在这里。
        </p>
        <router-link to="/" class="button primary">
          开始第一次识别 →
        </router-link>
      </div>
    </div>
    <div class="archive-pagination">
      <span>
        第 {{ page }} 页 · 每页 {{ size }} 条
      </span>
      <el-pagination v-model:current-page="page" :page-size="size" :total="total" layout="prev, pager, next" @current-change="loadList" />
    </div>
  </section>
  <p class="archive-footnote">
    历史记录保留当次模型输出；不同版本模型的结果可能存在差异。
  </p>
  <el-dialog v-model="dialogVisible" title="识别记录详情" width="min(960px, 94vw)" destroy-on-close class="record-dialog">
    <div v-loading="detailLoading" style="min-height: 160px">
      <p v-if="detailError" class="inline-error" role="alert">
        {{ detailError }}
      </p>
      <template v-if="detail">
        <div class="detail-heading">
          <div>
            <p class="eyebrow">
              RECORD #{{ String(detail.record_id).padStart(3, '0') }}
            </p>
            <h3>
              {{ detail.filename }}
            </h3>
            <span class="muted">
              {{ formatTime(detail.created_at) }}
            </span>
          </div>
          <span class="detail-count">
            {{ detail.detections.length }}
            <small>
              只候选实例
            </small>
          </span>
        </div>
        <div class="detail-image">
          <img :src="detail.result_image_base64" alt="历史识别标注图" />
        </div>
        <div class="detail-detections">
          <div v-for="(d, i) in detail.detections" :key="i">
            <span class="specimen-index">
              #{{ i + 1 }}
            </span>
            <strong>
              {{ d.status === 'uncertain' && d.candidate_cn ? `疑似${d.candidate_cn}` : d.species_cn }}
            </strong>
            <span>
              {{ d.confidence.toFixed(1) }}%
            </span>
            <span class="badge" :class="d.status">
              {{ d.status === 'confirmed' ? '已确认' : '待确认' }}
            </span>
            <small v-if="d.threshold !== undefined">确认参考阈值 {{ d.threshold.toFixed(1) }}%</small>
            <small v-if="d.status === 'uncertain' && d.top_candidates?.length">候选参考：{{ d.top_candidates.map(c => `${c.cn} ${c.confidence.toFixed(1)}%`).join('；') }}</small>
          </div>
          <p v-if="!detail.detections.length">
            未检出蜱虫
            <span v-if="detail.insect">
              · 粗分类候选：{{ detail.insect.cn }}（{{ detail.insect.confidence.toFixed(1) }}%）
            </span>
          </p>
        </div>
        <div class="detail-footer">
          <span class="request-id">
            请求编号 {{ detail.request_id }}
          </span>
          <a class="button outline" :href="detail.result_image_base64" :download="`标注结果_${detail.filename}`">
            下载标注图 ↓
          </a>
        </div>
      </template>
    </div>
  </el-dialog>
</template>
