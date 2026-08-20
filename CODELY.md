

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
- [2026-08-18 17:12:57] FFmpeg crop 滤镜支持时间变量 `t` 实现动态裁剪窗口，可用于定格动画 landing transform。**Why:** image-motion-animation skill 的 render_semantic_cartoon.py 用 PIL 逐帧合成 landing transform（scale 衰减 + 位移 + 弹跳），但在 FFmpeg 管线中可用 crop 滤镜的 `t` 变量直接实现，无需 PIL 逐帧渲染。**How to apply:** VF 链：先 scale 放大 4%（如 1331×749），再 `crop=w=1280:h=720:x='(iw-1280)/2+{dir}*14*max(0,1-t/0.3)':y='(ih-720)/2-10*sin(min(1,t/0.3)*PI)'`。x 用 `max(0,1-t/0.3)` 做线性衰减位移，y 用 `sin(min(1,t/0.3)*PI)` 做半正弦弹跳，方向 dir 按 idx%2 交替 ±1。注意 crop 表达式内逗号必须用单引号包裹整个表达式。
- [2026-08-18 19:07:44] rembg (U2-Net) AI 语义抠图远优于亮度阈值白底移除。**Why:** 旧 remove_white_bg 用 RGB 亮度≥238 判定背景，对浅色衣服/皮肤高光/阴影边缘有大量误伤，pose_0_0 透明率仅 68.6%；rembg 用 U2-Net 语义分割，同一图透明率 83.8%，角色边缘干净无残留。**How to apply:** stop_motion.py remove_bg() 优先用 rembg，rembg 不可用时降级为 _remove_white_bg_fallback。rembg 首次运行需下载 bria-rmbg-2.0.onnx 模型（~44MB），用户可能需手动下载放入 ~/.u2net/ 目录。requirements.txt 已添加 rembg+onnxruntime。
- [2026-08-18 22:51:20] stop_motion pose 图片必须是半身大特写（half-body close-up, waist up），不生成道具/物品/场景。**Why:** 全身图含道具时 rembg 抠图透明率低（test3 仅 67-73%），且 image-to-image 模式下三视图参考图太强势会导致 pose 几乎复制参考图。**How to apply:** image_gen.py pose prompt 强制添加 "half-body close-up shot, waist up, no full body, no props, no objects, no scene"；normalize_pose 居中对齐（POSE_CENTER_Y=TARGET_H//2+60）而非 bottom 锚定。LLM poses 数组只描述表情和手势，禁止提及道具/场景。paste_with_shadow 新增 centered=True 参数（y 是垂直中心而非底部边缘），否则半身图 top=360-640=-280 会导致角色跳出画面顶部。
- [2026-08-19 23:08:04] Pose atlas 方案：所有角色统一 4×2 八宫格（8 poses，2560×1280），PIL crop 分割后逐张 rembg 抠图。所有图片 prompt 统一使用 _QUEST_STYLE（"3D cartoon style, Pixar-like, warm soft lighting, cel-shaded, vibrant saturated colors, smooth surfaces"）保证画面风格一致。**How to apply:** image_gen.py generate_quest_atlases() 生成 4 张 atlas（char_a/b/c/host 各 8 poses = 32 pose images）。pipeline.py quest 分支：char_pose_map 对所有角色读 range(8)。


- [2026-08-19 00:42:07] SenseNova DeepSeek V4 Flash API 在生成复杂 JSON 时会把所有输出放在 reasoning_content 字段，导致 content 为空字符串。**Why:** 模型的推理模式消耗了所有 max_tokens 配额用于 reasoning_content，留给 content 的 token 为 0。在 quest 模式 48 行复杂 prompt 下必现，20 行也会出现。**How to apply:** _chat() 请求体中添加 "reasoning_effort": "low" 参数，将推理开销降到最低，确保 content 有足够 token 输出完整 JSON。llm_client.py 的 _chat 函数已添加此参数。
- [2026-08-19 14:45:43] FFmpeg concat demuxer 的 concat.txt 中文件路径必须是绝对路径。**Why:** 当 work_dir 是相对路径（如 'test_output'）时，concat.txt 写入的相对路径在 FFmpeg 运行时 cwd 不同会导致 "Impossible to open" 错误。**How to apply:** media_utils.py concat_segments() 中用 Path(s).resolve() 写绝对路径到 concat.txt。
- [2026-08-19 14:45:45] Kokoro KPipeline.__init__() 不再接受 repo_id 参数。**Why:** kokoro 库更新后移除了该参数，传 repo_id 会 TypeError。**How to apply:** tts_engine.py 中 KPipeline(lang_code='a') 和 KPipeline(lang_code='z') 不传 repo_id。

### Project
- [2026-08-19 23:08:11] [project] colab_listening_b quest 模式最终结构（2026-08-19 阶段1更新）：(1) 4阶段对话 buildup→core→reveal→review。(2) 4角色：char_a/char_b + char_c（服务人员）+ host（节目主）。(3) 无 title_card，welcome/hook/outro 用字幕显示。burn_subtitles 的 subtitle_seg_types 包含 welcome/hook_intro/outro；长文本按句子分割逐句显示（_re.split 按句号/问号/感叹号分句，按字数比例分配时间）。(4) TTS 语速 0%，pad 0.4s。(5) Host 在 TV 演播室背景上出镜。所有角色统一 4×2 atlas（8 poses），统一 _QUEST_STYLE 画面风格。(6) Dialogue 用多场景背景（scene_images 3-5 个），每 5 行轮换。(7) on_screen 字段控制画面角色。(8) Phase 1 动画改进：① 光学流插帧 — pose 切换时调用 generate_morph_frames（5帧过渡，0.2s），不再硬切；② 倾听者微动 — 正弦波呼吸 ±2px（3.3s 周期）+ 偶尔眨眼（每 5-7s 短暂切换到 surprised pose）；③ 音频驱动切换 — _compute_audio_rms_segments 检测 TTS 音频低能量点（语句停顿），在停顿处切换 pose，不再随机 2-5s。(9) concat_segments 用 Path.resolve() 写绝对路径。(10) Kokoro KPipeline 不传 repo_id。











- [2026-08-13 15:18:38] colab_listening_b YouTube title format: 繁中+emoji, 【】bracket tag, ｜separator, 80-150 chars. Three patterns: (A) 【沉浸式英文動畫】hook+emoji+topic：skills，聽完就能說！｜English topic; (B) 【每天50句英文】emoji+topic情境對話｜🎬沉浸式英文動畫｜...; (C) 【🎬沉浸式英文動畫】emoji+topic英文｜✅skill｜🗣️small talk｜... End with ｜{English topic name}. **Why:** videos target overseas Chinese audience, need high-CTR 繁中 titles. **How to apply:** LLM prompt in llm_client.py and enhanced/llm_client_enhanced.py specifies these patterns.
- [2026-08-15 00:08:51] - colab_listening_b 多Token轮换+断点续传机制：(1) mcp_client.py 支持多token，API返回"积分"/"credit"/"余额"/"quota"时自动切换下一个token，全部耗尽抛ALL_MCP_TOKENS_EXHAUSTED（单token积分错误同样明确抛错）。(2) pipeline.py 每步完成后保存checkpoint.json，--resume跳过已完成步骤。步骤: step0_script→step2_images_tts→step3_video→step4_timeline→step4.5_thumbnail→step5_compose→step6_4k；运行完整性以 step6_4k 判定（4K崩溃可续传）；checkpoint 记录 _run_dir，扫描子目录时按 mtime 取最新。(3) 完成后自动删除checkpoint.json。Step3 逐clip恢复（已存在的 clip 文件直接复用，不重花积分）。(4) 每个视频产物存独立子文件夹(以YouTube title命名)。(5) 视频文件也以YouTube title命名。(6) used_topics.json 默认写在 output 目录（Drive持久化），topic 只在脚本成功保存后才标记 used。(7) Notebook单Cell，MCP_TOKENS数组格式，输出到Google Drive；--mcp-tokens 替代 --mcp-token；--no-4k 跳过4K；clip_0 场景空镜与TTS并行生成。



### Reference

