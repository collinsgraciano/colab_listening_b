

## Codely Structured Memories

### User

### Feedback
- [2026-08-12 11:15:55] [feedback] pipeline.py 中 subprocess 必须在模块级导入，不能仅在函数内 `import subprocess as _sp`（局部别名不影响外部作用域）。**Why:** _get_audio_duration 内的局部 import 导致 main() 中的 subprocess.run 调用 NameError 崩溃。**How to apply:** Python 模块中所有函数共享的模块级 import 必须放在文件顶部，函数内 import 只在需要延迟加载重依赖（如 kokoro）时使用。
- [2026-08-12 11:15:58] [feedback] get_zh_voice 必须根据 speaker gender 返回不同 edge-tts voice（male→zh-CN-YunxiNeural, female→zh-CN-XiaoxiaoNeural），不能硬编码女声。**Why:** 之前 get_zh_voice 始终返回 zh-CN-XiaoxiaoNeural，导致男性角色中文翻译配音为女声，与英文男声 am_adam 不一致。**How to apply:** 在 tts_engine.py 中，get_zh_voice 读 script 的 char_a_gender/char_b_gender 字段决定 voice；synth_chinese 必须使用传入的 voice 参数而非内部硬编码。
- [2026-08-12 12:41:15] [feedback] 每次修改完代码后都要自动 commit + push 到 GitHub。**Why:** 用户要求代码变更后立即同步到远程仓库，不要等用户提醒。**How to apply:** 在 colab_listening_b 项目中，完成任何代码修改后，自动执行 git add + commit + push origin main，无需用户额外指示。
- [2026-08-13 01:37:17] [feedback] f-string 中 JSON schema/示例的花括号必须用双花括号转义（{{ }} 而非 { }），否则 Python 将其解释为格式说明符导致 ValueError: Invalid format specifier。**Why:** llm_client.py 和 enhanced/llm_client_enhanced.py 的 thumbnail_icons 字段示例 `[{"en": string, "zh": string}]` 在 f-string 中未转义，Colab 运行时崩溃。**How to apply:** 在 f-string 模板中写 JSON schema 时，所有字面花括号都要写成 {{ }}，包括描述行、示例行和 schema 行。
- [2026-08-13 15:18:38] Seedance2 video API duration parameter must be an integer between 4 and 15 (or -1). Values outside this range cause [InvalidParameter] error. **Why:** Group B pipeline calculates group_dur = round(total_audio + n_lines*pad), which can exceed 15s for groups with 4+ lines. **How to apply:** always clamp: `group_dur = max(4, min(group_dur, 15))` before passing to generate_video.
- [2026-08-14 15:05:01] Kokoro TTS loudnorm 单次模式对极安静音频（RMS -37dB）效果不好。**Why:** dialogue_1/8/13/15 等文件 loudnorm 后仍在 -37dB，比正常文件低 ~18dB。**How to apply:** tts_engine.py _loudnorm 改为两步：(1) loudnorm 线性归一化到 -16dB；(2) volumedetect 检测实际 mean_volume，若仍低于 -20dB 则计算差值用 volume={boost}dB+alimiter 补偿。
- [2026-08-15 00:08:51] argparse 以数字开头的长选项（如 --4k-timeout）生成的 dest 是 "4k_timeout"，不是合法 Python 标识符，只能通过 args.__dict__ 访问。**Why:** argparse 把内部连字符转下划线推导 dest，前导数字导致属性访问非法。**How to apply:** CLI flag 一律以字母开头（如 --upscale-timeout，或 --no-4k 配显式 dest="no_4k"）。
- [2026-08-16 01:12:19] 图片/视频混排合成时，overlay PNG 尺寸绝不能硬编码——必须按 ffprobe 探测底视频实际分辨率渲染，且所有段先统一 scale+pad 归一化到目标画布，否则画布小于 overlay 时字幕会被裁出画面（static 模式字幕超框的根因：frontier 图片原生分辨率直接 -loop 1 进段）。**Why:** ffmpeg overlay 以第一个输入为画布，超出画布的 overlay 像素直接丢弃。**How to apply:** video_compose.py 已有 _VF_NORM + _probe_resolution 可复用；字幕/文字定位一律自下而上堆叠（底边固定 margin），不要自上而下累加。
- [2026-08-17 13:44:22] [feedback] pipeline.py 中 write_file 工具误把中文元数据/注释写入文件头会彻底破坏 .py 文件（SyntaxError: invalid character）。**Why:** 在尝试用 write_file 做字节级替换时，工具把 Markdown 验证注释当作文件内容写入了 pipeline.py，导致文件头变成中文非 Python 内容。**How to apply:** (1) 修改现有 .py 文件时优先用 replace 工具做精确替换，不要用 write_file 整体覆盖。(2) 如果 replace 失败（old_string 不匹配），先 read_file 重新确认精确内容再重试，不要用 write_file 绕行。(3) 如果文件已损坏，`git checkout HEAD -- <file>` 可从最近提交恢复。

### Project
- [2026-08-17 01:26:32] [project] colab_listening_b: Colab 上运行的英语听力练习视频生成管线（方案 B）。流程：LLM 脚本(SenseNova)→frontier 图片+Kokoro/edge-tts TTS→方案 B 分组(连续行合并为一个视频片段)→Seedance2 视频→FFmpeg+Pillow 合成。18 行对话，繁体中文，720p 16:9。pipeline.py 已重构为步骤函数架构：main() 仅 ~75 行编排，_step0_script/_step1_mcp/_step2_images_tts/_step3_clips/_step4_timeline/_step45_thumbnail/_step5_compose/_step6_4k 独立可测；纯函数 _build_scene_clip_task/_build_group_clip_tasks/_enrich_timeline/_safe_dirname（新增步骤写新 _stepN_* 并在 main() 加一行调用）。**Why:** 用户要 Colab 版本脱离 Windows 依赖；巨型 main() 无法单测。**How to apply:** --mcp-tokens 传 token，--api-key 传 SenseNova key，--structure original|enhanced|static|quest，--model deepseek-v4-flash|glm-5.2。GitHub 公开仓库：https://github.com/collinsgraciano/colab_listening_b.git。四种结构：(1) original 4章节；(2) enhanced 7章节（词汇+慢速+测验，enhanced/ 目录）；(3) static 全图片模式（compose_static，dialogue_img_{i}.png 逐行图片，跳过视频生成）；(4) quest 任务钩子慢速听力（quest/ 目录，2026-08-17 新增）：旁白钩子(听力任务诱饵)→三幕对话 build up/core/review→闭环CTA，3角色(char_c=服务人员，core 行参考 char_scene_c.png)，48行默认(26/9/13)，TTS -25% 无中文，pad 默认5s，全图片零视频，场景无关可套任意 topic，--num-lines/--pad 缺省按模式自动(quest→48/5.0，其余→18/0.4)。




- [2026-08-13 15:18:38] colab_listening_b YouTube title format: 繁中+emoji, 【】bracket tag, ｜separator, 80-150 chars. Three patterns: (A) 【沉浸式英文動畫】hook+emoji+topic：skills，聽完就能說！｜English topic; (B) 【每天50句英文】emoji+topic情境對話｜🎬沉浸式英文動畫｜...; (C) 【🎬沉浸式英文動畫】emoji+topic英文｜✅skill｜🗣️small talk｜... End with ｜{English topic name}. **Why:** videos target overseas Chinese audience, need high-CTR 繁中 titles. **How to apply:** LLM prompt in llm_client.py and enhanced/llm_client_enhanced.py specifies these patterns.
- [2026-08-15 00:08:51] - colab_listening_b 多Token轮换+断点续传机制：(1) mcp_client.py 支持多token，API返回"积分"/"credit"/"余额"/"quota"时自动切换下一个token，全部耗尽抛ALL_MCP_TOKENS_EXHAUSTED（单token积分错误同样明确抛错）。(2) pipeline.py 每步完成后保存checkpoint.json，--resume跳过已完成步骤。步骤: step0_script→step2_images_tts→step3_video→step4_timeline→step4.5_thumbnail→step5_compose→step6_4k；运行完整性以 step6_4k 判定（4K崩溃可续传）；checkpoint 记录 _run_dir，扫描子目录时按 mtime 取最新。(3) 完成后自动删除checkpoint.json。Step3 逐clip恢复（已存在的 clip 文件直接复用，不重花积分）。(4) 每个视频产物存独立子文件夹(以YouTube title命名)。(5) 视频文件也以YouTube title命名。(6) used_topics.json 默认写在 output 目录（Drive持久化），topic 只在脚本成功保存后才标记 used。(7) Notebook单Cell，MCP_TOKENS数组格式，输出到Google Drive；--mcp-tokens 替代 --mcp-token；--no-4k 跳过4K；clip_0 场景空镜与TTS并行生成。



### Reference

