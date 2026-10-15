<template>
  <div class="app-shell">
    <header class="site-header">
      <router-link to="/" class="brand" aria-label="系统首页">
        <span class="brand-icon" aria-hidden="true">
          ✳
        </span>
        <span>
          <strong>
            基于Mask R-CNN的蜱虫识别系统的设计与实现
          </strong>
          <small>
            GRADUATION PROJECT · TICK IDENTIFICATION
          </small>
        </span>
      </router-link>
      <nav class="main-nav" aria-label="主导航">
        <router-link to="/" exact-active-class="active">
          图像识别
        </router-link>
        <router-link to="/history" active-class="active">
          识别档案
        </router-link>
      </nav>
    </header>
    <main class="main-content">
      <router-view />
    </main>
    <footer class="site-footer">
      <span>
        蜱识 · 让每一次观察更清晰
      </span>
      <span>
        数量统计 / 逐只定位 / 蜱种识别
      </span>
    </footer>
  </div>
</template>
