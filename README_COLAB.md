# Colab Listening Video Generator (方案 B)

在 Google Colab 上运行的英语听力练习视频生成管线。基于 `tjgenerators-b-english-listening-video` skill，适配 Colab Linux 环境。

## 文件清单

```
colab_listening_b/
├── listening_video_colab.ipynb   # Colab notebook（6 个 cell）
├── mcp_client.py                 # MCP 客户端（token 参数传入，非文件读取）
├── llm_client.py                 # LLM 脚本生成（无 Windows 代码）
├── tts_engine.py                 # TTS 引擎（无 HF mirror，Colab 直连 HuggingFace）
├── timeline.py                   # 时间轴 + SRT 生成（不变）
├── grouping_b.py                 # 方案 B 分组逻辑（不变）
├── video_compose.py              # FFmpeg 合成（字体路径自动检测 Win/Linux）
└── pipeline.py                   # 主流程（--mcp-token 参数，无 stdout reconfigure）
```

## 与 Windows skill 的差异

| 项目 | Windows skill | Colab 版本 |
|------|---------------|-----------|
| MCP Token | 从 `~/.codely-cli/mcp-oauth-tokens.json` 读取 | `--mcp-token` 参数传入 |
| 字体路径 | `C:\Windows\Fonts\msyhbd.ttc` 等 | `/usr/share/fonts/truetype/noto/` 等（apt 安装） |
| HF Mirror | `HF_ENDPOINT=https://hf-mirror.com` | 不需要（Colab 直连） |
| stdout | `sys.stdout.reconfigure(encoding="utf-8")` | 不需要（Linux 默认 UTF-8） |
| GPU | CPU only | Colab GPU 可用（Kokoro 自动检测） |

## 使用方法

1. 打开 [Google Colab](https://colab.research.google.com)，上传 `listening_video_colab.ipynb`
2. 运行 Cell 1 安装依赖（~3min）
3. 将本目录所有 `.py` 文件上传到 Colab 的 `/content/listening_b/`（用左侧文件浏览器）
4. 运行 Cell 2 验证文件
5. 在 Cell 3 填入 MCP OAuth token（从本地 `mcp-oauth-tokens.json` 获取）
6. 运行 Cell 4 生成视频（~15-25min）
7. 运行 Cell 5 下载结果

## 获取 MCP Token

在本地 Windows 机器运行：
```python
import json
tokens = json.load(open(r'C:\Users\Administrator\.codely-cli\mcp-oauth-tokens.json'))
token = next(t['token']['accessToken'] for t in tokens if t['serverName'] == 'TJGenerators')
print(token)
```
