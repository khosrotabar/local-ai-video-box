from __future__ import annotations

import codecs
import json
import os
import queue
import re
import secrets
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


ROOT = Path("/opt/ai-movie")
SERVER = ROOT / "server"
OUTPUT_DIR = ROOT / "outputs" / "api"
LOG_DIR = ROOT / "logs" / "api"
DB_PATH = SERVER / "jobs.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("AI_MOVIE_API_KEY", "").strip()

app = FastAPI(
    title="AI Movie Multi-Engine API",
    version="0.2.0",
)

security = HTTPBearer(auto_error=False)

# We keep one sequential GPU queue on the current 1x RTX 5090.
job_queue: queue.Queue[str] = queue.Queue()

# job_id -> subprocess.Popen
active_processes: dict[str, subprocess.Popen] = {}
active_processes_lock = threading.Lock()


class JobCancelled(Exception):
    pass


class GenerateRequest(BaseModel):
    engine: Literal["ltx", "wan", "skyreels"] = "ltx"
    prompt: str = Field(min_length=3, max_length=8000)
    negative_prompt: str | None = None
    seed: int = 42


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    return conn


def has_column(conn: sqlite3.Connection, name: str) -> bool:
    columns = conn.execute("PRAGMA table_info(jobs)").fetchall()
    return any(column["name"] == name for column in columns)


def init_db():
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                engine TEXT NOT NULL,
                prompt TEXT NOT NULL,
                negative_prompt TEXT,
                seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                output_file TEXT,
                error TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        migrations = {
            "stage": "TEXT",
            "started_at": "TEXT",
            "finished_at": "TEXT",
        }

        for column, sql_type in migrations.items():
            if not has_column(conn, column):
                conn.execute(
                    f"ALTER TABLE jobs ADD COLUMN {column} {sql_type}"
                )

        conn.execute(
            """
            UPDATE jobs
            SET
                status='failed',
                stage='failed',
                error='Backend restarted while generation was running.',
                finished_at=?,
                updated_at=?
            WHERE status='running'
            """,
            (now(), now()),
        )


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server API key is not configured.",
        )

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(
            credentials.credentials,
            API_KEY,
        )
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )


def get_job(job_id: str):
    with db() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE id=?",
            (job_id,),
        ).fetchone()


def get_status(job_id: str) -> str | None:
    row = get_job(job_id)
    return row["status"] if row else None


def update_job(job_id: str, **values):
    if not values:
        return

    values["updated_at"] = now()

    columns = ", ".join(
        f"{key}=?" for key in values
    )
    params = list(values.values()) + [job_id]

    with db() as conn:
        conn.execute(
            f"UPDATE jobs SET {columns} WHERE id=?",
            params,
        )


def row_to_dict(row: sqlite3.Row):
    result = {
        "id": row["id"],
        "engine": row["engine"],
        "prompt": row["prompt"],
        "seed": row["seed"],
        "status": row["status"],
        "progress": row["progress"],
        "stage": row["stage"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "video_url": (
            f"/api/videos/{row['id']}"
            if row["status"] == "completed"
            else None
        ),
    }

    if row["metadata"]:
        try:
            result["metadata"] = json.loads(row["metadata"])
        except Exception:
            result["metadata"] = None
    else:
        result["metadata"] = None

    return result


def common_env():
    env = os.environ.copy()

    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CC": "gcc",
            "CXX": "g++",
            "HF_HOME": "/opt/ai-movie/cache/huggingface",
            "TRANSFORMERS_CACHE": "/opt/ai-movie/cache/huggingface",
            "PYTHONUNBUFFERED": "1",
        }
    )

    return env


PERCENT_RE = re.compile(
    r"(?<!\d)(100|[1-9]?\d)\s*%\s*\|"
)

STEP_RE = re.compile(
    r"(?<!\d)(\d+)\s*/\s*(\d+)(?!\d)"
)


def detect_stage(line: str, current_stage: str) -> str:
    text = line.lower()

    if any(
        term in text
        for term in (
            "loading transformer",
            "loading model",
            "loading checkpoint",
            "loading weights",
            "loading vae",
            "loading text encoder",
            "load model",
        )
    ):
        return "loading_model"

    if any(
        term in text
        for term in (
            "encoding prompt",
            "encode prompt",
            "text encoding",
            "tokenizing",
        )
    ):
        return "encoding_prompt"

    if any(
        term in text
        for term in (
            "generating",
            "generation",
            "denois",
            "sampling",
            "diffusion",
            "step_index:",
            "start segment",
            "infer_main",
        )
    ):
        return "generating"

    if any(
        term in text
        for term in (
            "encoding mp4",
            "export_to_video",
            "saving video",
            "writing video",
            "ffmpeg",
            "muxing",
            "mux ",
        )
    ):
        return "encoding_video"

    return current_stage


def parse_progress(
    engine: str,
    line: str,
    current_stage: str,
) -> tuple[str, int | None]:
    stage = detect_stage(line, current_stage)

    percent_match = PERCENT_RE.search(line)

    if percent_match:
        percent = max(
            0,
            min(100, int(percent_match.group(1))),
        )

        # Do not accidentally treat model-download/loading bars as
        # generation percentage unless we already reached generation,
        # or the text itself clearly indicates inference/generation.
        lower = line.lower()

        generation_hint = any(
            x in lower
            for x in (
                "generat",
                "denois",
                "sampl",
                "diffusion",
                "infer",
            )
        )

        if stage == "generating" or generation_hint:
            return "generating", min(percent, 99)

        # Wan / LTX often emit tqdm without descriptive text once
        # the actual denoising loop starts. Their models are already
        # local, so step-style bars after startup are useful signals.
        step_match = STEP_RE.search(line)

        if step_match:
            current = int(step_match.group(1))
            total = int(step_match.group(2))

            if total > 0:
                # Known inference-loop sizes are small.
                # This avoids treating large checkpoint-shard bars
                # as generation progress.
                if total <= 100:
                    return "generating", min(percent, 99)

    # Some tools expose steps without an explicit percentage.
    step_match = STEP_RE.search(line)

    if step_match and stage == "generating":
        current = int(step_match.group(1))
        total = int(step_match.group(2))

        if total > 0:
            percentage = int((current / total) * 100)
            return stage, min(max(percentage, 0), 99)

    return stage, None


def register_process(
    job_id: str,
    process: subprocess.Popen,
):
    with active_processes_lock:
        active_processes[job_id] = process


def unregister_process(
    job_id: str,
    process: subprocess.Popen,
):
    with active_processes_lock:
        current = active_processes.get(job_id)

        if current is process:
            active_processes.pop(job_id, None)


def terminate_process_tree(
    process: subprocess.Popen,
    graceful_timeout: float = 5.0,
):
    if process.poll() is not None:
        return

    try:
        os.killpg(
            os.getpgid(process.pid),
            signal.SIGTERM,
        )
    except ProcessLookupError:
        return

    deadline = time.time() + graceful_timeout

    while time.time() < deadline:
        if process.poll() is not None:
            return

        time.sleep(0.1)

    try:
        os.killpg(
            os.getpgid(process.pid),
            signal.SIGKILL,
        )
    except ProcessLookupError:
        pass


def run_live_process(
    job_id: str,
    engine: str,
    command: list[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env=None,
):
    recent_lines: deque[str] = deque(maxlen=120)

    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )

    register_process(job_id, process)

    stage = "starting"
    last_progress: int | None = None
    last_stage = stage

    decoder = codecs.getincrementaldecoder("utf-8")(
        errors="replace"
    )
    buffer = ""

    try:
        update_job(
            job_id,
            stage="starting",
        )

        with log_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        ) as log:

            fd = process.stdout.fileno()

            while True:
                chunk = os.read(fd, 4096)

                if not chunk:
                    break

                text = decoder.decode(chunk)
                log.write(text)
                log.flush()

                buffer += text

                # tqdm commonly updates using carriage return rather
                # than newline, so handle both.
                parts = re.split(r"[\r\n]+", buffer)
                buffer = parts.pop() if parts else ""

                for line in parts:
                    clean = line.strip()

                    if not clean:
                        continue

                    recent_lines.append(clean)

                    stage, parsed_progress = parse_progress(
                        engine,
                        clean,
                        stage,
                    )

                    changes = {}

                    if stage != last_stage:
                        changes["stage"] = stage
                        last_stage = stage

                    if (
                        parsed_progress is not None
                        and parsed_progress != last_progress
                    ):
                        changes["progress"] = parsed_progress
                        last_progress = parsed_progress

                    if changes:
                        # Do not overwrite a cancellation that happened
                        # concurrently via the API.
                        if get_status(job_id) != "cancelled":
                            update_job(
                                job_id,
                                **changes,
                            )

                if get_status(job_id) == "cancelled":
                    terminate_process_tree(process)

            remaining = decoder.decode(b"", final=True)

            if remaining:
                buffer += remaining

            if buffer.strip():
                log.write("\n")
                recent_lines.append(buffer.strip())

        return_code = process.wait()

        if get_status(job_id) == "cancelled":
            raise JobCancelled()

        if return_code != 0:
            tail = "\n".join(recent_lines)

            raise RuntimeError(
                f"Engine exited with code {return_code}\n{tail}"
            )

    finally:
        unregister_process(job_id, process)


def run_ltx(
    job_id: str,
    job,
    output: Path,
    log_path: Path,
):
    repo = ROOT / "engines" / "ltx" / "LTX-2"
    py = repo / ".venv" / "bin" / "python"
    model = ROOT / "models" / "ltx-2.5"

    command = [
        str(py),
        "-m",
        "ltx_pipelines.distilled",

        "--transformer-path",
        str(
            model
            / "diffusion_models"
            / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
        ),

        "--text-encoder-path",
        str(
            model
            / "text_encoders"
            / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
        ),

        "--video-vae-path",
        str(
            model
            / "vae"
            / "ltx-2.5-video-vae-bf16.safetensors"
        ),

        "--audio-vae-path",
        str(
            model
            / "vae"
            / "ltx-2.5-audio-vae-bf16.safetensors"
        ),

        "--spatial-upsampler-path",
        str(
            model
            / "latent_upscale_models"
            / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
        ),

        "--height",
        "1024",

        "--width",
        "1536",

        "--num-frames",
        "121",

        "--frame-rate",
        "24",

        "--seed",
        str(job["seed"]),

        "--quantization",
        "fp8-cast",

        "--prompt",
        job["prompt"],

        "--output-path",
        str(output),
    ]

    run_live_process(
        job_id,
        "ltx",
        command,
        log_path,
        cwd=repo,
        env=common_env(),
    )


def run_wan(
    job_id: str,
    job,
    output: Path,
    log_path: Path,
):
    repo = ROOT / "engines" / "wan" / "LightX2V"

    env = common_env()

    env.update(
        {
            "CUDA_HOME": "/usr/local/cuda-13.0",

            "JOB_PROMPT": job["prompt"],

            "JOB_NEG": job["negative_prompt"]
            or (
                "overexposed, static, blurry details, subtitles, "
                "worst quality, low quality, jpeg artifacts, "
                "deformed, malformed, extra fingers"
            ),

            "JOB_OUT": str(output),
        }
    )

    env["PATH"] = (
        "/usr/local/cuda-13.0/bin:"
        + env.get("PATH", "")
    )

    env["LD_LIBRARY_PATH"] = (
        "/usr/local/cuda-13.0/lib64:"
        + env.get("LD_LIBRARY_PATH", "")
    )

    script = r"""
set -Eeuo pipefail

export PYTHONPATH="${PYTHONPATH:-}"

lightx2v_path=/opt/ai-movie/engines/wan/LightX2V
model_path=/opt/ai-movie/models/wan2.2-t2v-base

source \
  /opt/ai-movie/engines/wan/LightX2V/scripts/base/base.sh

exec \
  /opt/ai-movie/engines/wan/LightX2V/.venv/bin/python \
  -m lightx2v.infer \
  --model_cls wan2.2_moe \
  --task t2v \
  --model_path \
    /opt/ai-movie/models/wan2.2-t2v-base \
  --config_json \
    /opt/ai-movie/engines/wan/LightX2V/configs/wan22/extreme/wan_moe_t2v_5090.json \
  --prompt "$JOB_PROMPT" \
  --negative_prompt "$JOB_NEG" \
  --save_result_path "$JOB_OUT"
"""

    run_live_process(
        job_id,
        "wan",
        ["/bin/bash", "-lc", script],
        log_path,
        cwd=repo,
        env=env,
    )


def run_skyreels(
    job_id: str,
    job,
    output: Path,
    log_path: Path,
):
    repo = ROOT / "engines" / "skyreels-diffusers"
    py = repo / ".venv" / "bin" / "python"

    command = [
        str(py),
        str(SERVER / "skyreels_runner.py"),

        "--prompt",
        job["prompt"],

        "--output",
        str(output),

        "--seed",
        str(job["seed"]),
    ]

    env = common_env()

    env["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"
    )

    run_live_process(
        job_id,
        "skyreels",
        command,
        log_path,
        cwd=repo,
        env=env,
    )


def probe_video(path: Path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def process_job(job_id: str):
    row = get_job(job_id)

    if row is None:
        return

    # A queued job may have been cancelled while waiting.
    if row["status"] == "cancelled":
        return

    job = dict(row)

    output = OUTPUT_DIR / f"{job_id}.mp4"
    log_path = LOG_DIR / f"{job_id}.log"

    try:
        update_job(
            job_id,
            status="running",
            progress=0,
            stage="starting",
            error=None,
            started_at=now(),
            finished_at=None,
        )

        if output.exists():
            output.unlink()

        if job["engine"] == "ltx":
            run_ltx(
                job_id,
                job,
                output,
                log_path,
            )

        elif job["engine"] == "wan":
            run_wan(
                job_id,
                job,
                output,
                log_path,
            )

        elif job["engine"] == "skyreels":
            run_skyreels(
                job_id,
                job,
                output,
                log_path,
            )

        else:
            raise RuntimeError(
                f"Unknown engine: {job['engine']}"
            )

        if get_status(job_id) == "cancelled":
            raise JobCancelled()

        update_job(
            job_id,
            stage="encoding_video",
            progress=99,
        )

        if (
            not output.exists()
            or output.stat().st_size == 0
        ):
            raise RuntimeError(
                "Engine finished but no video was produced."
            )

        metadata = probe_video(output)

        update_job(
            job_id,
            status="completed",
            progress=100,
            stage="completed",
            output_file=str(output),
            metadata=json.dumps(metadata),
            error=None,
            finished_at=now(),
        )

    except JobCancelled:
        if output.exists():
            try:
                output.unlink()
            except OSError:
                pass

        # Cancellation endpoint normally set this already;
        # keep it idempotent.
        update_job(
            job_id,
            status="cancelled",
            stage="cancelled",
            finished_at=now(),
            error=None,
        )

    except Exception as exc:
        if get_status(job_id) == "cancelled":
            if output.exists():
                try:
                    output.unlink()
                except OSError:
                    pass

            update_job(
                job_id,
                status="cancelled",
                stage="cancelled",
                finished_at=now(),
                error=None,
            )

            return

        update_job(
            job_id,
            status="failed",
            progress=100,
            stage="failed",
            error=str(exc)[-12000:],
            finished_at=now(),
        )


def worker_loop():
    while True:
        job_id = job_queue.get()

        try:
            row = get_job(job_id)

            if row is None:
                continue

            if row["status"] == "cancelled":
                continue

            process_job(job_id)

        finally:
            job_queue.task_done()


@app.get("/api/health")
def health():
    with active_processes_lock:
        active_count = len(active_processes)

    return {
        "ok": True,
        "gpu_workers": 1,
        "active_jobs": active_count,
        "engines": [
            "ltx",
            "wan",
            "skyreels",
        ],
    }


@app.get(
    "/api/engines",
    dependencies=[Depends(require_auth)],
)
def engines():
    return {
        "engines": [
            {
                "id": "ltx",
                "name": "LTX-2.5 22B Distilled",
                "resolution": "1536x1024",
                "fps": 24,
                "frames": 121,
                "shot_seconds": 5.04,
                "direct_long": False,
            },
            {
                "id": "wan",
                "name": "Wan 2.2 A14B NVFP4",
                "resolution": "832x480",
                "fps": 16,
                "frames": 81,
                "shot_seconds": 5.06,
                "direct_long": False,
            },
            {
                "id": "skyreels",
                "name": "SkyReels V2 DF 14B FP8",
                "resolution": "960x544",
                "fps": 24,
                "frames": 57,
                "shot_seconds": 2.375,
                "direct_long": False,
            },
        ]
    }


@app.post(
    "/api/generations",
    status_code=202,
    dependencies=[Depends(require_auth)],
)
def create_generation(request: GenerateRequest):
    job_id = str(uuid.uuid4())
    timestamp = now()

    with db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                id,
                engine,
                prompt,
                negative_prompt,
                seed,
                status,
                progress,
                stage,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                request.engine,
                request.prompt,
                request.negative_prompt,
                request.seed,
                "queued",
                0,
                "queued",
                timestamp,
                timestamp,
            ),
        )

    job_queue.put(job_id)

    return {
        "id": job_id,
        "engine": request.engine,
        "status": "queued",
        "progress": 0,
        "stage": "queued",
    }


@app.get(
    "/api/generations",
    dependencies=[Depends(require_auth)],
)
def list_generations():
    with db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM jobs
            ORDER BY created_at DESC
            LIMIT 100
            """
        ).fetchall()

    return {
        "items": [
            row_to_dict(row)
            for row in rows
        ]
    }


@app.get(
    "/api/generations/{job_id}",
    dependencies=[Depends(require_auth)],
)
def generation(job_id: str):
    row = get_job(job_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Generation not found.",
        )

    return row_to_dict(row)


@app.post(
    "/api/generations/{job_id}/cancel",
    dependencies=[Depends(require_auth)],
)
def cancel_generation(job_id: str):
    row = get_job(job_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Generation not found.",
        )

    current_status = row["status"]

    if current_status in (
        "completed",
        "failed",
        "cancelled",
    ):
        return {
            "id": job_id,
            "status": current_status,
        }

    # Mark first. This prevents the worker from starting a queued
    # job and tells a running worker not to convert SIGTERM -> failed.
    update_job(
        job_id,
        status="cancelled",
        stage="cancelled",
        finished_at=now(),
        error=None,
    )

    with active_processes_lock:
        process = active_processes.get(job_id)

    if process is not None:
        terminate_process_tree(process)

    output = OUTPUT_DIR / f"{job_id}.mp4"

    if output.exists():
        try:
            output.unlink()
        except OSError:
            pass

    return {
        "id": job_id,
        "status": "cancelled",
    }


@app.get(
    "/api/videos/{job_id}",
    dependencies=[Depends(require_auth)],
)
def video(job_id: str):
    row = get_job(job_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Generation not found.",
        )

    if row["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Video is not ready.",
        )

    path = Path(row["output_file"])

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Video file is missing.",
        )

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


init_db()

worker = threading.Thread(
    target=worker_loop,
    name="gpu-worker-0",
    daemon=True,
)

worker.start()
