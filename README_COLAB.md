# Colab Listening Video Generator (方案 B)

在 Google Colab 上运行的英语听力练习视频生成管线（全流程单 Cell）。基于 `tjgenerators-b-english-listening-video` skill，适配 Colab Linux 环境。

## 流程

```
LLM 脚本生成 (SenseNova DeepSeek V4 Flash)
  → 并行: frontier 图片生成 + Kokoro/edge-tts 配音
  → clip_0 场景空镜与 TTS 并行生成
  → 方案 B 分组（连续行合并为一个 Seedance2 视频片段，时长按组音频动态计算 4-15s）
  → 时间轴 + SRT（统一时间轴保证音画同步）
  → Pillow 静态帧 + 字幕（跟讀章节 EN-only 帧 / EN+ZH 帧）
  → FFmpeg 合成 → loudnorm → 4K 升采样（可 --no-4k 跳过）
```

## 文件清单

```
colab_listening_b/
├── listening_video_colab.ipynb   # Colab notebook（单 Cell：安装→挂载→clone→运行→下载）
├── requirements.txt              # Python 依赖
├── mcp_client.py                 # MCP 客户端（多 token 轮换，积分耗尽自动切换）
├── llm_client.py                 # LLM 脚本生成（original 结构）
├── tts_engine.py                 # TTS（Kokoro EN + edge-tts ZH，两步 loudnorm）
├── timeline.py                   # 时间轴 + SRT 生成
├── grouping_b.py                 # 方案 B 分组逻辑
├── video_compose.py              # FFmpeg 合成（original 4 章节结构）
├── thumbnail_gen.py              # YouTube 缩略图 + 元数据
├── topic_manager.py              # 随机选题 + 防重复
├── topics.json                   # 选题库（23 类 247 个主题）
├── pipeline.py                   # 主入口（checkpoint 断点续传）
└── quest/                        # quest 任务钩子慢速听力结构（全图片，任意场景）
```

## 视频结构

- **original**（4 章节）：标题卡 → 沉浸式情境對話 → 跟讀練習（1EN×3→ZH→1EN→sil ×9 段/行）→ Outro
- **static**（original 结构的全图片版）：不生成 Seedance 视频，逐行 frontier 图片 + TTS，最省积分
- **quest**（任务钩子慢速听力，全图片版，结构可复制到任意场景）：标题卡 → 旁白钩子（"我会说得很慢"+ 布置听力任务，答案在影片里）→ 三幕慢速對話（buildup 需求铺垫 char_a+char_b → core 核心实战 char_a+char_c 服务人员 → review 评价复盘 char_a+char_b）→ 旁白闭环 CTA（重复问题 + 评论区用英文作答 + 订阅）。A1-A2 句长≤10 词、目标词重复≥3 次、语速 -25%、段间停顿 5s。**无跟读/词汇/测验章节**。默认 48 行（约 9-11 分钟）；换任何 topic（机场登机、酒店入住、职场面试…）结构不变。

## 使用方法

1. 打开 [Google Colab](https://colab.research.google.com)，上传 `listening_video_colab.ipynb`
2. 运行 Cell（~3min 安装依赖 + 15-25min 生成）
3. 在 Cell 内填入：
   - `MCP_TOKENS`：TJGenerators OAuth token 数组（支持多个，积分耗尽自动轮换）
   - `SENSENOVA_API_KEY`：SenseNova API key
4. 可调参数：`TOPIC`（空=随机选题）、`CEFR`、`NUM_LINES`、`STRUCTURE`（original/static/quest）、`RESUME`

产物输出到 Google Drive `/content/drive/MyDrive/listening_videos/<YouTube标题>/`，包含：
`videos/*.mp4`（720p + 4K）、`thumbnail.jpg`、`script.json`、`youtube_metadata.json`、`images/`、`clips/`、`audio/`、`subtitles/`。

## 断点续传

- 每步完成后写入 `checkpoint.json`（步骤：step0_script → step2_images_tts → step3_video → step4_timeline → step4.5_thumbnail → step5_compose → step6_4k）
- `--resume`：定位最近未完成的 checkpoint，跳过已完成步骤；视频 clip 按**单个文件**复用（部分完成不浪费积分）
- 全部完成后自动删除 checkpoint；下次 `--resume` 会扫描子目录找未完成的运行，找不到则开新选题
- **主题防重复**：`used_topics.json` 默认写在输出目录（Drive 上持久化，不受 Colab 重装影响）；topic 只在脚本成功保存后才标记 used

## 获取 MCP Token

在本地 Windows 机器运行：
```python
import json
tokens = json.load(open(r'C:\Users\Administrator\.codely-cli\mcp-oauth-tokens.json'))
token = next(t['token']['accessToken'] for t in tokens if t['serverName'] == 'TJGenerators')
print(token)
```

## 常用参数

| 参数 | 说明 |
|------|------|
| `--topic` | 指定主题（缺省从 topics.json 随机） |
| `--cefr` | A1-C2（默认 A2） |
| `--num-lines` | 对话行数（默认：quest 48 / 其他 18） |
| `--pad` | 段间停顿秒数（默认：quest 5.0 / 其他 0.4） |
| `--structure` | `original` / `static` / `quest` |
| `--mcp-tokens` | 多 token 逗号分隔 |
| `--resume` | 断点续传 |
| `--no-4k` | 跳过 4K 升采样 |
| `--upscale-timeout` | 4K 编码超时（默认 3600s） |

## 与 Windows skill 的差异

| 项目 | Windows skill | Colab 版本 |
|------|---------------|-----------|
| MCP Token | 从 `~/.codely-cli/mcp-oauth-tokens.json` 读取 | `--mcp-tokens` 参数传入 |
| 字体路径 | `C:\Windows\Fonts\msyhbd.ttc` 等 | `/usr/share/fonts/truetype/noto/` + DejaVu（IPA） |
| HF Mirror | `HF_ENDPOINT=https://hf-mirror.com` | 不需要（Colab 直连） |
| stdout | `sys.stdout.reconfigure(encoding="utf-8")` | 不需要（Linux 默认 UTF-8） |
| GPU | CPU only | Colab GPU 可用（Kokoro 自动检测） |
