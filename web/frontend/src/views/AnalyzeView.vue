<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { analyzeImage } from '../api/client'
import type { AnalyzeResult } from '../types'
const picker = ref<HTMLInputElement | null>(null)
const file = ref<File | null>(null)
const originalUrl = ref('')
const loading = ref(false)
const dragging = ref(false)
const error = ref('')
const result = ref<AnalyzeResult | null>(null)
const viewMode = ref<'original' | 'annotated'>('original')
const previewSrc = computed(() => viewMode.value === 'annotated' && result.value ? result.value.result_image_base64 : originalUrl.value)
const count = computed(() => result.value?.detections.length ?? 0)
const confirmed = computed(() => result.value?.detections.filter(d => d.status === 'confirmed').length ?? 0)
const summary = computed(() => !result.value ? '等待图像分析' : count.value ? `检出 ${count.value} 个蜱虫候选实例` : result.value.insect?.slug === 'tick' ? '疑似蜱虫，尚未定位' : '本次未检出蜱虫')
function choose(candidate?: File) {
  if (!candidate || loading.value) return
  if (!['image/jpeg', 'image/png', 'image/bmp', 'image/webp'].includes(candidate.type)) { ElMessage.warning('请选择 JPG、PNG、BMP 或 WebP 图片'); return }
  if (candidate.size > 10 * 1024 * 1024) { ElMessage.warning('图片大小不能超过 10 MB'); return }
  if (originalUrl.value) URL.revokeObjectURL(originalUrl.value)
  originalUrl.value = URL.createObjectURL(candidate)
  file.value = candidate; result.value = null; error.value = ''; viewMode.value = 'original'
}
function onChange(event: Event) {
  const input = event.target as HTMLInputElement
  choose(input.files?.[0]); input.value = ''
}
function drop(event: DragEvent) { dragging.value = false; choose(event.dataTransfer?.files[0]) }
async function analyze() {
  if (!file.value || loading.value) return
  loading.value = true; error.value = ''
  try { result.value = await analyzeImage(file.value); viewMode.value = 'annotated' }
  catch (cause) { error.value = cause instanceof Error ? cause.message : '本次分析未完成，请稍后重试。' }
  finally { loading.value = false }
}
onUnmounted(() => { if (originalUrl.value) URL.revokeObjectURL(originalUrl.value) })
</script>
<template>
  <section class="page-intro">
    <div>
      <p class="eyebrow">
        IMAGE IDENTIFICATION / 图像识别
      </p>
      <h1>
        从一张照片，
        <br />
        认识眼前的蜱虫
        <span>
          。
        </span>
      </h1>
      <p class="intro-copy">
        上传清晰照片，查看蜱虫数量、逐只位置与蜱种结果。
      </p>
    </div>
    <div class="intro-index">
      <span>
        01 — 03
      </span>
      <p>
        观察 · 定位 · 识别
      </p>
      <div class="index-rule">
      </div>
      <small>
        保留不确定性，让结果有据可查
      </small>
    </div>
  </section>
  <div class="analysis-layout">
    <section class="panel image-panel" aria-labelledby="image-heading">
      <div class="panel-heading">
        <h2 id="image-heading">
          <span class="step">
            01
          </span>
          图像工作区
        </h2>
        <span class="muted">
          {{ file ? '已载入图像' : '等待上传' }}
        </span>
      </div>
      <input ref="picker" class="visually-hidden" type="file" accept="image/jpeg,image/png,image/bmp,image/webp" aria-label="上传待识别图片" :disabled="loading" @change="onChange" />
      <div v-if="!file" class="upload-zone" :class="{ dragging }" @dragover.prevent="dragging = true" @dragleave.prevent="dragging = false" @drop.prevent="drop">
        <div class="focus-frame" aria-hidden="true">
          <svg viewBox="0 0 80 80" fill="none">
            <path d="M25 10H10v15M55 10h15v15M10 55v15h15M70 55v15H55" stroke="currentColor" stroke-width="2"/>
            <ellipse cx="40" cy="41" rx="10" ry="15" stroke="currentColor" stroke-width="2"/>
            <path d="M34 25l-7-9m19 9 7-9M30 33l-12-5m32 5 12-5M29 43H15m36 0h14M31 53l-12 9m30-9 12 9" stroke="currentColor" stroke-width="2"/>
          </svg>
        </div>
        <h3>
          将照片拖到这里
        </h3>
        <p>
          或从设备中选择一张图片
        </p>
        <button class="button primary" @click="picker?.click()">
          选择图片
          <span aria-hidden="true">
            ↗
          </span>
        </button>
        <small>
          JPG / PNG / BMP / WebP · 最大 10 MB
        </small>
      </div>
      <div v-else class="image-stage">
        <img :src="previewSrc" alt="当前图像预览" />
        <div v-if="loading" class="image-loading">
          <span class="spinner">
          </span>
          <strong>
            正在识别图像
          </strong>
          <small>
            正在定位目标并逐只分析，复杂图片可能需要约两分钟
          </small>
        </div>
      </div>
      <div v-if="file" class="image-toolbar">
        <div class="file-name">
          <span class="file-dot">
          </span>
          <span>
            {{ file.name }}
          </span>
          <small>
            {{ (file.size / 1024 / 1024).toFixed(2) }} MB
          </small>
        </div>
        <div class="toolbar-actions">
          <button class="button subtle" :disabled="loading" @click="picker?.click()">
            更换图片
          </button>
          <div v-if="result" class="segmented">
            <button :class="{ selected: viewMode === 'original' }" @click="viewMode = 'original'">
              原图
            </button>
            <button :class="{ selected: viewMode === 'annotated' }" @click="viewMode = 'annotated'">
              标注图
            </button>
          </div>
        </div>
      </div>
      <div class="workspace-footer">
        <span>
          <i class="status-dot" :class="{ busy: loading }">
          </i>
          {{ loading ? '正在分析，请勿重复提交' : file ? '照片已就绪' : '清晰、对焦准确的照片有助于识别' }}
        </span>
        <button class="button primary" :disabled="!file || loading" @click="analyze">
          {{ loading ? '正在分析…' : result ? '重新分析' : '开始识别' }}
          <span v-if="!loading" aria-hidden="true">
            →
          </span>
        </button>
      </div>
      <p v-if="error" class="inline-error" role="alert">
        {{ error }}
      </p>
    </section>
    <aside class="panel result-panel" aria-labelledby="result-heading" aria-live="polite">
      <div class="panel-heading">
        <h2 id="result-heading">
          <span class="step">
            02
          </span>
          识别概览
        </h2>
        <span class="tiny-label">
          RESULT
        </span>
      </div>
      <div class="result-overview">
        <p class="eyebrow">
          {{ result ? '本次识别' : '蜱虫候选实例' }}
        </p>
        <div class="count-number">
          {{ result ? String(count).padStart(2, '0') : '—' }}
          <span>
            只
          </span>
        </div>
        <h3>
          {{ summary }}
        </h3>
        <p>
          {{ result ? (count ? `${confirmed} 只蜱种已确认，${count - confirmed} 只待确认` : '没有定位结果不代表照片中一定不存在蜱虫') : '分析后将展示数量、位置和逐只识别结果。' }}
        </p>
      </div>
      <div v-if="!result" class="result-checklist">
        <div>
          <span>
            01
          </span>
          <p>
            <strong>
              定位目标
            </strong>
            <small>
              在图片中标出每个候选实例
            </small>
          </p>
        </div>
        <div>
          <span>
            02
          </span>
          <p>
            <strong>
              辨识蜱种
            </strong>
            <small>
              分别展示中文名、学名与置信度
            </small>
          </p>
        </div>
        <div>
          <span>
            03
          </span>
          <p>
            <strong>
              保留待确认结果
            </strong>
            <small>
              证据不足时提示补充清晰照片
            </small>
          </p>
        </div>
      </div>
      <div v-else class="result-meta">
        <div>
          <span>
            已确认蜱种
          </span>
          <strong>
            {{ confirmed }}
          </strong>
        </div>
        <div>
          <span>
            待人工确认
          </span>
          <strong>
            {{ count - confirmed }}
          </strong>
        </div>
        <div v-if="result.insect && !count">
          <span>
            粗分类候选
          </span>
          <strong>
            {{ result.insect.cn }}
          </strong>
        </div>
        <div v-if="result.insect && !count">
          <span>
            分类置信度
          </span>
          <strong>
            {{ result.insect.confidence.toFixed(1) }}%
          </strong>
        </div>
        <p v-if="result.insect?.status === 'uncertain' && !count" class="candidate-note">粗分类尚待确认，请补拍清晰照片。{{ result.insect.threshold !== undefined ? `确认参考阈值 ${result.insect.threshold.toFixed(1)}%` : '' }}</p>
        <p v-if="result.insect?.top_candidates?.length && !count" class="candidate-note">候选参考：{{ result.insect.top_candidates.map(c => `${c.cn} ${c.confidence.toFixed(1)}%`).join('；') }}</p>
        <a class="button outline full-width" :href="result.result_image_base64" :download="`标注结果_${file?.name ?? 'tick.jpg'}`">
          下载标注结果 ↓
        </a>
        <p class="request-id">
          请求编号 {{ result.request_id }}
        </p>
      </div>
      <div class="soft-note">
        <span>
          i
        </span>
        <p>
          识别结果仅供辅助鉴定。低置信度结果标记为待确认，建议补拍或请专业人员复核。
        </p>
      </div>
    </aside>
  </div>
  <section v-if="result && count" class="panel instances-panel">
    <div class="panel-heading">
      <h2>
        <span class="step">
          03
        </span>
        逐只识别结果
      </h2>
      <span class="muted">
        共 {{ count }} 个实例 · 与标注框对应
      </span>
    </div>
    <div class="instance-grid">
      <article v-for="(d, i) in result.detections" :key="i" class="instance-card">
        <div class="instance-top">
          <span class="specimen-index">
            #{{ String(i + 1).padStart(2, '0') }}
          </span>
          <span class="badge" :class="d.status">
            {{ d.status === 'confirmed' ? '已确认' : '待确认' }}
          </span>
        </div>
        <h3>
          {{ d.status === 'uncertain' && d.candidate_cn ? `疑似${d.candidate_cn}` : d.species_cn }}
        </h3>
        <p class="latin">
          {{ (d.candidate_slug || (d.species_slug === 'uncertain' ? 'Species undetermined' : d.species_slug)).replaceAll('_', ' ') }}
        </p>
        <div class="confidence-row">
          <span>
            蜱种分类置信度
          </span>
          <strong>
            {{ d.confidence.toFixed(1) }}%
          </strong>
        </div>
        <div class="confidence-track">
          <i :style="{ width: `${d.confidence}%` }">
          </i>
        </div>
        <div class="instance-details">
          <span v-if="d.threshold !== undefined">
            确认参考阈值
            <strong>{{ d.threshold.toFixed(1) }}%</strong>
          </span>
          <span>
            检测置信度
            <strong>
              {{ (d.score * 100).toFixed(1) }}%
            </strong>
          </span>
          <span>
            位置
            <code>
              {{ d.bbox.map(v => Math.round(v)).join(', ') }}
            </code>
          </span>
        </div>
        <div v-if="d.status === 'uncertain'" class="candidate-note">
          <p>{{ d.decision_reason || '现有证据不足以确认蜱种，请补拍清晰照片或人工复核。' }}</p>
          <p v-if="d.top_candidates?.length">候选参考：{{ d.top_candidates.map(c => `${c.cn} ${c.confidence.toFixed(1)}%`).join('；') }}</p>
        </div>
      </article>
    </div>
  </section>
  <section v-else-if="!result" class="photo-tips">
    <span class="tiny-label">
      拍摄建议
    </span>
    <p>
      <strong>
        靠近一点
      </strong>
      让目标尽量占据画面主体
    </p>
    <p>
      <strong>
        对焦清晰
      </strong>
      保留体表纹理与形态细节
    </p>
    <p>
      <strong>
        完整入镜
      </strong>
      避免遮挡与过强的反光
    </p>
  </section>
</template>
