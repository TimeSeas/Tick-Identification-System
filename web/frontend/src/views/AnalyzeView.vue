<script setup lang="ts">
import { ref, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { analyzeImage } from '../api/client'
import type { AnalyzeResult } from '../types'

// ===== 状态 =====
const loading = ref(false)                          // 请求中：按钮转圈
const file = ref<File | null>(null)                 // 用户选的原图文件
const result = ref<AnalyzeResult | null>(null)      // 分析结果（含标注图 base64）
const viewMode = ref<'original' | 'annotated'>('original')  // 预览切换：原图 / 标注图

// 原图预览 URL：File 对象不能直接给 <img>，用 createObjectURL 生成浏览器本地地址
// （这个地址只在当前页面有效，不需要上传就有预览——所以"选了图立刻能看到"）
const originalUrl = ref('')
function onFileChange(uploadFile: { raw: File }) {
  file.value = uploadFile.raw
  result.value = null                    // 换了新图，清掉旧结果
  viewMode.value = 'original'            // 回到原图视图
  if (originalUrl.value) URL.revokeObjectURL(originalUrl.value)  // 释放旧地址，防内存泄漏
  originalUrl.value = URL.createObjectURL(uploadFile.raw)
}

async function doAnalyze() {
  if (!file.value) {
    ElMessage.warning('请先选择图片')
    return
  }
  loading.value = true
  try {
    result.value = await analyzeImage(file.value)
    viewMode.value = 'annotated'         // 出结果后自动切到标注图
  } catch {
    // 错误提示已由 api 层统一弹出
  } finally {
    loading.value = false
  }
}

// 当前预览的图：原图走本地 URL，标注图走后端返回的 base64
//const previewSrc = computed(...)
//定义一个计算属性​ previewSrc。计算属性的值会根据依赖的响应式数据自动变化。
//viewMode.value === 'annotated'判断是不是标注模式，为真的话，运行&&后面代码 
//result.value存在，取标注图，不存在取原图
const previewSrc = computed(() =>
  viewMode.value === 'annotated' && result.value
    ? result.value.result_image_base64
    : originalUrl.value,
)

// 下载文件名：原文件名前加"标注结果_"前缀
const downloadName = computed(() => {
  const base = file.value?.name ?? 'tick'
  return `标注结果_${base}`
})

// 检测列表的行数据：给每个结果补一个展示用的序号与 bbox 文本
const detectionRows = computed(() =>
  (result.value?.detections ?? []).map((d, i) => ({
    index: i + 1,
    ...d,
    bboxText: `[${d.bbox.join(', ')}]`,
  })),
)
//判断有没有出结果
const hasResult = computed(() => result.value !== null)
const hasDetections = computed(() => (result.value?.detections.length ?? 0) > 0)
</script>

<template>
  <div class="page">
    <!-- 操作区 -->
    <div class="toolbar">
      <el-upload :auto-upload="false" :show-file-list="false"
                 accept="image/*" :on-change="onFileChange">
        <el-button>选择图片</el-button>
      </el-upload>
      <el-button type="primary" :loading="loading" @click="doAnalyze">开始分析</el-button>

      <!-- 预览切换：有结果后才显示标注图选项 -->
      <el-radio-group v-if="hasResult" v-model="viewMode" style="margin-left: 12px">
        <el-radio-button value="original">原图</el-radio-button>
        <el-radio-button value="annotated">标注图</el-radio-button>
      </el-radio-group>

      <!-- 下载标注图：base64 数据可直接作为href下载链接 -->
      <el-button v-if="hasResult" style="margin-left: 12px"
                 tag="a" :href="result!.result_image_base64"
                 :download="downloadName" type="success" plain>
        下载标注图
      </el-button>
    </div>

    <!-- 图片预览区 -->
    <div class="preview">
      <img v-if="previewSrc" :src="previewSrc" alt="预览图" />
      <el-empty v-else description="选择图片开始分析" />
    </div>

    <!-- 结果区 -->
    <template v-if="hasResult">
      <!-- 风险建议 -->
      <el-alert :title="result!.risk_advice" type="warning" :closable="false" style="margin-top: 16px" />

      <!-- 识别结果列表 -->
      <div style="margin-top: 16px">
        <h4>识别结果</h4>
        <el-table v-if="hasDetections" :data="detectionRows" border stripe>
          <el-table-column prop="index" label="#" width="50" />
          <el-table-column prop="species_cn" label="蜱种（中文名）" min-width="140" />
          <el-table-column prop="species_slug" label="学名" min-width="160" />
          <el-table-column label="检测置信度" width="110">
            <template #default="{ row }">{{ (row.score * 100).toFixed(1) }}%</template>
          </el-table-column>
          <el-table-column label="分类置信度" width="110">
            <template #default="{ row }">{{ row.confidence.toFixed(1) }}%</template>
          </el-table-column>
          <el-table-column label="状态" width="100">
            <template #default="{ row }">
              <el-tag :type="row.status === 'confirmed' ? 'success' : 'warning'">
                {{ row.status === 'confirmed' ? '已确认' : '待确认' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="bboxText" label="位置 (x1,y1,x2,y2)" min-width="180" />
        </el-table>
        <el-empty v-else description="未检测到蜱虫" :image-size="60" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.page {
  max-width: 900px;
  margin: 0 auto;
}
.toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.preview {
  margin-top: 16px;
  text-align: center;
}
.preview img {
  max-width: 100%;
  max-height: 520px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
}
</style>
