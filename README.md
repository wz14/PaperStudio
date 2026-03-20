# PaperStudio

面向**微信服务号**的科研助手基础版：后端使用 [Future-House/paper-qa](https://github.com/Future-House/paper-qa)（PaperQA）做文献 RAG 问答，支持**按微信用户隔离文献库**、**多用户对话记录**（SQLite）、**链接/图片入库**与**客服消息异步推送长回答**。

## 功能概览

| 能力 | 说明 |
|------|------|
| PaperQA RAG | 每用户独立 `papers/` 与向量索引目录，与 paper-qa 的 `IndexSettings` 对齐 |
| 多人隔离 | 使用 `openid` 的哈希作为数据目录名，互不影响 |
| 对话历史 | `chat_messages` 表记录用户与助手消息，便于审计与后续多轮扩展 |
| 微信幂等 | `MsgId` 去重，避免平台重试导致重复入库或重复扣费 |
| 文本问答 | 用户发问题 → 被动回复短确认 → **客服消息**推送完整回答（被动回复 5 秒限制） |
| 文献入库 | `添加 <URL>`、文本中带论文 URL、或发送「链接」消息 |
| 图片 | 下载临时素材保存到文献目录（复杂 PDF 仍建议用链接） |

### 关于「通过微信传文件」

微信公众平台对**服务号**接收消息类型以文本、图片、语音、链接等为主；**用户直接发送 PDF 文件**在多数场景下不可用。推荐方式：

1. 发送论文 **URL**（消息中带链接或使用「添加 https://…」）；  
2. 使用「分享链接」类消息（本服务会处理 `MsgType=link`）；  
3. 若后续开通企业微信 / 小程序，可再接专用上传通道。

## 快速开始

```bash
cd /workspace
pip install -r requirements.txt
cp .env.example .env
# 填写 OPENAI_API_KEY、WECHAT_TOKEN、WECHAT_APP_ID、WECHAT_APP_SECRET
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

健康检查：`GET /health`。

## 微信公众平台配置

1. **服务器 URL**：`https://你的域名/wechat`，**Token** 与 `.env` 中 `WECHAT_TOKEN` 一致。  
2. 明文模式即可（本版本未实现消息加解密；生产若开启安全模式需自行扩展 `WXBizMsgCrypt`）。  
3. 需配置 **AppID / AppSecret**，用于：  
   - `access_token` 拉取临时素材（图片）；  
   - **客服消息**接口推送长文本回答。  
4. 若未配置 AppID/Secret：仍可完成 URL 验证与短文本被动回复，但**无法**下发客服消息与下载图片。

## 环境变量

见 `.env.example`。PaperQA 通过 **LiteLLM** 调用模型，请至少设置 `OPENAI_API_KEY`（或与所选模型匹配的密钥）。

可选：`PAPERQA_LLM`、`PAPERQA_SUMMARY_LLM`、`PAPERQA_EMBEDDING`、`PAPERQA_AGENT_TIMEOUT`。

## 项目结构

```
app/
  main.py        # FastAPI、微信回调
  config.py      # 应用配置
  db.py          # SQLite 模型与幂等
  wechat.py      # 签名、XML、客服消息、素材下载
  rag_service.py # PaperQA 每用户 Settings 与文献路径
  handlers.py    # 命令解析与异步任务
data/            # 运行时数据（默认 gitignore）
```

## 用户侧命令（节选）

- `帮助`：说明  
- `文献`：列出当前文献文件名  
- `添加 https://arxiv.org/pdf/...`：下载并入库  
- 直接输入问题：基于已入库文献做 PaperQA 问答  

## 许可证

MIT（若上游 paper-qa 有额外要求，请以其仓库为准）。
