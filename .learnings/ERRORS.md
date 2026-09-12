# Errors

Command failures and integration errors.

---

## [ERR-20260912-001] uvicorn_port_bind

**Logged**: 2026-09-12T07:00:00Z
**Priority**: medium
**Status**: resolved
**Area**: backend

### Summary
后端启动报 [Errno 10048] address already in use (127.0.0.1:8000)

### Error
```
ERROR: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000): 通常每个套接字地址(协议/网络地址/端口)只允许使用一次。
```

### Context
- 旧终端里的 uvicorn（旧模型）还占着 8000，新进程模型加载成功后绑定失败退出
- 排查：`netstat -ano | grep ":8000"`（无输出=端口已释放）；旧进程退出后直接重启即可

### Suggested Fix
启动前先 `netstat -ano | grep ":8000"` 查占用；有旧进程先 taskkill /PID <pid> /F，或等旧进程退出后重试

### Metadata
- Reproducible: yes
- Related Files: web/backend
- Recurrence-Count: 1
- First-Seen: 2026-09-12

### Resolution
- **Resolved**: 2026-09-12T07:05:00Z
- **Notes**: 旧后端退出后重试启动成功

---

## [ERR-20260912-002] curl_missing_file_stale_json

**Logged**: 2026-09-12T08:00:00Z
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
回归测试循环里输入图不存在时 curl 失败但 -o 目标文件保留上一次结果，导致读到"过期"响应，误判某 case 未生效

### Context
- 菜青虫测试图路径写错（Cabbage_worm 不存在，实际是 Cabbage Loopers），curl 上传失败，analyze_test.json 还是蓟马图的结果 → 打印出与上一条完全相同的输出
- 教训：循环测试脚本里先 `[ -f "$img" ] || continue` 校验文件存在；curl 加 `-f` 让 HTTP/上传失败返回非零

### Suggested Fix
批量回归测试：文件存在性校验 + curl -f + 每次循环前 rm 旧响应文件

### Metadata
- Reproducible: yes
- Related Files: 无
- Recurrence-Count: 1
- First-Seen: 2026-09-12

### Resolution
- **Resolved**: 2026-09-12T08:10:00Z
- **Notes**: 后续回归循环已加 [ -f ] 校验

---
