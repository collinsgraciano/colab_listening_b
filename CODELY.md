

## Codely Structured Memories

### User

### Feedback
- [2026-08-12 11:15:55] [feedback] pipeline.py 中 subprocess 必须在模块级导入，不能仅在函数内 `import subprocess as _sp`（局部别名不影响外部作用域）。**Why:** _get_audio_duration 内的局部 import 导致 main() 中的 subprocess.run 调用 NameError 崩溃。**How to apply:** Python 模块中所有函数共享的模块级 import 必须放在文件顶部，函数内 import 只在需要延迟加载重依赖（如 kokoro）时使用。
- [2026-08-12 11:15:58] [feedback] get_zh_voice 必须根据 speaker gender 返回不同 edge-tts voice（male→zh-CN-YunxiNeural, female→zh-CN-XiaoxiaoNeural），不能硬编码女声。**Why:** 之前 get_zh_voice 始终返回 zh-CN-XiaoxiaoNeural，导致男性角色中文翻译配音为女声，与英文男声 am_adam 不一致。**How to apply:** 在 tts_engine.py 中，get_zh_voice 读 script 的 char_a_gender/char_b_gender 字段决定 voice；synth_chinese 必须使用传入的 voice 参数而非内部硬编码。
- [2026-08-12 12:41:15] [feedback] 每次修改完代码后都要自动 commit + push 到 GitHub。**Why:** 用户要求代码变更后立即同步到远程仓库，不要等用户提醒。**How to apply:** 在 colab_listening_b 项目中，完成任何代码修改后，自动执行 git add + commit + push origin main，无需用户额外指示。

### Project
- [2026-08-12 11:15:51] [project] colab_listening_b: Colab 上运行的英语听力练习视频生成管线（方案 B）。流程：LLM 脚本(SenseNova)→frontier 图片+Kokoro/edge-tts TTS→方案 B 分组(连续行合并为一个视频片段)→Seedance2 视频→FFmpeg+Pillow 合成。结构：Ch1 标题→Ch2 沉浸式情境對話(grouped)→Ch3 跟讀練習(1EN×3→1ZH→1EN→sil)→Ch4 outro。18 行对话，繁体中文，720p 16:9。**Why:** 用户要 Colab 版本脱离 Windows 依赖。**How to apply:** pipeline.py 是主入口，--mcp-token 传入 OAuth token；grouping_b.py 按音频时长合并连续行；video_compose.py 的 compose_listening 用 group_info/line_to_group 跳过已处理的行。
### Reference

