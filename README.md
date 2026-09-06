# Local AI Video Box

Self-hosted multi-engine AI video generation backend for a single NVIDIA RTX 5090.

This project recreates the AI video server stack used by the Frameforge frontend and provides one bootstrap script for provisioning a fresh GPU machine.

Current supported engines:

- LTX-2.5 22B Distilled
- Wan 2.2 A14B NVFP4
- SkyReels V2 DF 14B FP8

The goal is to make disposable GPU servers practical:

```text
Rent fresh RTX 5090 server
        ↓
git clone local-ai-video-box
        ↓
./bootstrap.sh
        ↓
Enter Hugging Face token
        ↓
Models + environments + kernels installed
        ↓
FastAPI starts
        ↓
Connect frontend
```

## Image references API

All API endpoints except `/api/health` require the configured bearer API
key. Upload one or more images first:

```bash
API_KEY="$(awk -F= '/^AI_MOVIE_API_KEY=/{print $2}' /opt/ai-movie/server/.env)"

START_UPLOAD_ID="$(
  curl -fsS http://127.0.0.1:11434/api/uploads \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@/path/to/start-image.jpg" \
  | jq -r .id
)"

STYLE_UPLOAD_ID="$(
  curl -fsS http://127.0.0.1:11434/api/uploads \
    -H "Authorization: Bearer $API_KEY" \
    -F "file=@/path/to/style-reference.webp" \
  | jq -r .id
)"
```

Create a generation with the normalized `references` array:

```bash
curl -fsS http://127.0.0.1:11434/api/generations \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  --data "{
    \"engine\": \"ltx\",
    \"prompt\": \"A cinematic slow camera move through a moonlit forest.\",
    \"references\": [
      {\"upload_id\": \"$START_UPLOAD_ID\", \"role\": \"start_image\"},
      {\"upload_id\": \"$STYLE_UPLOAD_ID\", \"role\": \"style\"}
    ]
  }"
```

`references` is optional and accepts up to eight unique upload IDs. Valid
roles are `reference` (the default when omitted), `start_image`, `character`,
`object`, `style`, `location`, and `final_target`. A generation accepts at
most one `final_target`; it is currently supported only by LTX.

All references are persisted and returned with generation records. The first
`start_image`, or else the first submitted reference, remains the native I2V
input. Every submitted reference is also analyzed locally by
`Qwen/Qwen2.5-VL-7B-Instruct`; its concise role-aware guidance is added only
to the engine's internal effective prompt. The original user prompt remains
unchanged in the API and database.

Analysis is cached by upload ID, role, and analyzer version. LTX, Wan, and
SkyReels therefore receive the same compact multi-reference guidance while
retaining their existing single-primary-image conditioning behavior.

For LTX, `final_target` additionally conditions frame 120 of the 121-frame
output. It is never selected as a fallback start image: LTX uses the first
`start_image`, then the first non-`final_target` reference, or no native start
image when neither exists.
