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
