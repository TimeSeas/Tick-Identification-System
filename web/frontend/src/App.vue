<script setup lang="ts">
// 根组件：全局布局 = 顶部导航 + 路由出口
</script>

<template>
  <el-container>
    <el-header class="header">
      <h2>🕷 蜱虫识别与叮咬风险提示系统</h2>
      <nav>
        <router-link to="/">分析</router-link>
        <router-link to="/history">历史</router-link>
      </nav>
    </el-header>
    <el-main>
      <router-view />
    </el-main>
  </el-container>
</template>

<style scoped>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #dcdfe6;
}
.header nav {
  display: flex;
  gap: 20px;
}
.header a {
  color: #409eff;
  text-decoration: none;
  font-size: 15px;
}
/* 当前路由高亮：router-link 激活时自带 router-link-active 类 */
.header a.router-link-active {
  color: #1f2d3d;
  font-weight: bold;
}
</style>
