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

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field


ROOT = Path("/opt/ai-movie")
SERVER = ROOT / "server"
OUTPUT_DIR = ROOT / "outputs" / "api"
LOG_DIR = ROOT / "logs" / "api"
UPLOAD_DIR = ROOT / "uploads"
DB_PATH = SERVER / "jobs.db"

MAX_UPLOAD_SIZE = 10 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
IMAGE_MEDIA_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
MAX_REFERENCES = 8
ANALYZER_VERSION = "qwen2.5-vl-7b-instruct-v1"
MAX_REFERENCE_ANALYSIS_CHARS = 420
MAX_REFERENCE_GUIDANCE_CHARS = 2800

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

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


class GenerationReference(BaseModel):
    upload_id: str = Field(min_length=1, max_length=64)
    role: Literal[
        "reference",
        "start_image",
        "character",
        "object",
        "style",
        "location",
        "final_target",
    ] = "reference"


class GenerateRequest(BaseModel):
    engine: Literal["ltx", "wan", "skyreels"] = "ltx"
    prompt: str = Field(min_length=3, max_length=8000)
    negative_prompt: str | None = None
    seed: int = 42
    references: list[GenerationReference] = Field(
        default_factory=list
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def db():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def has_column(conn: sqlite3.Connection, name: str) -> bool:
    columns = conn.execute("PRAGMA table_info(jobs)").fetchall()
    return any(column["name"] == name for column in columns)


def ensure_job_references_table(conn: sqlite3.Connection):
    columns = conn.execute(
        "PRAGMA table_info(job_references)"
    ).fetchall()

    expected_primary_key = ["job_id", "upload_id"]
    primary_key = [
        column["name"]
        for column in sorted(
            columns,
            key=lambda column: column["pk"],
        )
        if column["pk"]
    ]
    has_position = any(
        column["name"] == "position"
        for column in columns
    )

    if columns and primary_key == expected_primary_key and has_position:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_job_references_job_position
            ON job_references (job_id, position)
            """
        )
        return

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS job_references_new (
            job_id TEXT NOT NULL,
            upload_id TEXT NOT NULL,
            role TEXT NOT NULL,
            position INTEGER NOT NULL,
            PRIMARY KEY (job_id, upload_id),
            FOREIGN KEY (job_id) REFERENCES jobs(id),
            FOREIGN KEY (upload_id) REFERENCES uploads(id)
        )
        """
    )

    if columns:
        conn.execute(
            """
            INSERT INTO job_references_new (
                job_id, upload_id, role, position
            )
            SELECT job_id, upload_id, role, 0
            FROM job_references
            """
        )
        conn.execute("DROP TABLE job_references")

    conn.execute(
        "ALTER TABLE job_references_new RENAME TO job_references"
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_job_references_job_position
        ON job_references (job_id, position)
        """
    )


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
            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY,
                stored_file TEXT NOT NULL,
                media_type TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        ensure_job_references_table(conn)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reference_analyses (
                upload_id TEXT NOT NULL,
                role TEXT NOT NULL,
                analyzer_version TEXT NOT NULL,
                analysis TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (upload_id, role, analyzer_version),
                FOREIGN KEY (upload_id) REFERENCES uploads(id)
            )
            """
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


def get_job_references(job_id: str):
    with db() as conn:
        rows = conn.execute(
            """
            SELECT upload_id, role
            FROM job_references
            WHERE job_id=?
            ORDER BY position
            """,
            (job_id,),
        ).fetchall()

    return [
        {
            "upload_id": row["upload_id"],
            "role": row["role"],
        }
        for row in rows
    ]


def get_primary_reference_path(job_id: str) -> Path | None:
    with db() as conn:
        row = conn.execute(
            """
            SELECT uploads.stored_file
            FROM job_references
            JOIN uploads ON uploads.id = job_references.upload_id
            WHERE job_references.job_id=?
              AND job_references.role != 'final_target'
            ORDER BY
                CASE job_references.role
                    WHEN 'start_image' THEN 0
                    ELSE 1
                END,
                job_references.position
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()

    if row is None:
        return None

    return UPLOAD_DIR / row["stored_file"]


def get_final_target_path(job_id: str) -> Path | None:
    with db() as conn:
        row = conn.execute(
            """
            SELECT uploads.stored_file
            FROM job_references
            JOIN uploads ON uploads.id = job_references.upload_id
            WHERE job_references.job_id=?
              AND job_references.role='final_target'
            LIMIT 1
            """,
            (job_id,),
        ).fetchone()

    if row is None:
        return None

    return UPLOAD_DIR / row["stored_file"]


def get_references_for_analysis(job_id: str):
    with db() as conn:
        rows = conn.execute(
            """
            SELECT job_references.upload_id, job_references.role,
                   uploads.stored_file
            FROM job_references
            JOIN uploads ON uploads.id = job_references.upload_id
            WHERE job_references.job_id=?
            ORDER BY job_references.position
            """,
            (job_id,),
        ).fetchall()

    return [
        {
            "upload_id": row["upload_id"],
            "role": row["role"],
            "path": UPLOAD_DIR / row["stored_file"],
        }
        for row in rows
    ]


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

    result["references"] = get_job_references(row["id"])

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

STAGE_ORDER = {
    "starting": 0,
    "loading_model": 1,
    "encoding_prompt": 2,
    "generating": 3,
    "encoding_video": 4,
    "completed": 5,
}


def advance_stage(current_stage: str, candidate_stage: str) -> str:
    current_order = STAGE_ORDER.get(current_stage)
    candidate_order = STAGE_ORDER.get(candidate_stage)

    if candidate_order is None:
        return current_stage

    if current_order is None or candidate_order > current_order:
        return candidate_stage

    return current_stage


def detect_stage(line: str, current_stage: str) -> str:
    text = line.lower()

    candidate_stage = current_stage

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
        candidate_stage = "loading_model"

    elif any(
        term in text
        for term in (
            "encoding prompt",
            "encode prompt",
            "text encoding",
            "tokenizing",
        )
    ):
        candidate_stage = "encoding_prompt"

    elif any(
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
        candidate_stage = "generating"

    elif any(
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
        candidate_stage = "encoding_video"

    return advance_stage(current_stage, candidate_stage)


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

    current_job = get_job(job_id)
    stage = (
        current_job["stage"]
        if current_job is not None and current_job["stage"]
        else "starting"
    )
    last_progress = (
        int(current_job["progress"] or 0)
        if current_job is not None
        else 0
    )
    last_stage = stage

    decoder = codecs.getincrementaldecoder("utf-8")(
        errors="replace"
    )
    buffer = ""

    try:
        with log_path.open(
            "a",
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

                    if parsed_progress is not None:
                        progress = max(last_progress, parsed_progress)

                        if progress != last_progress:
                            changes["progress"] = progress
                            last_progress = progress

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


def cached_reference_analysis(upload_id: str, role: str) -> str | None:
    with db() as conn:
        row = conn.execute(
            """
            SELECT analysis
            FROM reference_analyses
            WHERE upload_id=? AND role=? AND analyzer_version=?
            """,
            (upload_id, role, ANALYZER_VERSION),
        ).fetchone()

    return row["analysis"] if row is not None else None


def clean_reference_analysis(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Analyzer returned an invalid response.")

    analysis = " ".join(value.split())

    if not analysis:
        raise ValueError("Analyzer returned an empty response.")

    return analysis[:MAX_REFERENCE_ANALYSIS_CHARS]


def run_reference_analyzer(
    job_id: str,
    pending_references: list[dict],
    log_path: Path,
) -> list[dict]:
    analyzer = ROOT / "engines" / "reference-analyzer"
    command_input = ROOT / "temp" / f"{job_id}-references.json"
    command_output = ROOT / "temp" / f"{job_id}-reference-analysis.json"

    payload = []

    for reference in pending_references:
        path = reference["path"]

        if not path.is_file():
            raise RuntimeError("Referenced image is unavailable.")

        payload.append(
            {
                "upload_id": reference["upload_id"],
                "role": reference["role"],
                "image_path": str(path),
            }
        )

    command_input.write_text(json.dumps(payload), encoding="utf-8")

    command = [
        str(analyzer / ".venv" / "bin" / "python"),
        str(SERVER / "reference_analyzer.py"),
        "--input",
        str(command_input),
        "--output",
        str(command_output),
    ]

    process = None

    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            log.write("Starting local reference analysis.\n")
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=common_env(),
            )
            register_process(job_id, process)

            while process.poll() is None:
                if get_status(job_id) == "cancelled":
                    terminate_process_tree(process)
                    raise JobCancelled()

                time.sleep(0.2)

        if get_status(job_id) == "cancelled":
            raise JobCancelled()

        if process.returncode != 0:
            raise RuntimeError("Reference image analysis failed.")

        try:
            results = json.loads(command_output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Reference image analysis failed.") from exc

        if not isinstance(results, list) or len(results) != len(payload):
            raise RuntimeError("Reference image analysis failed.")

        expected = {
            (reference["upload_id"], reference["role"])
            for reference in pending_references
        }
        analyses = []

        for result in results:
            if not isinstance(result, dict):
                raise RuntimeError("Reference image analysis failed.")

            upload_id = result.get("upload_id")
            role = result.get("role")

            if (upload_id, role) not in expected:
                raise RuntimeError("Reference image analysis failed.")

            analyses.append(
                {
                    "upload_id": upload_id,
                    "role": role,
                    "analysis": clean_reference_analysis(
                        result.get("analysis")
                    ),
                }
            )

        if len({(item["upload_id"], item["role"]) for item in analyses}) != len(expected):
            raise RuntimeError("Reference image analysis failed.")

        with db() as conn:
            for item in analyses:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO reference_analyses (
                        upload_id, role, analyzer_version, analysis,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        item["upload_id"],
                        item["role"],
                        ANALYZER_VERSION,
                        item["analysis"],
                        now(),
                    ),
                )

        return analyses

    finally:
        if process is not None:
            unregister_process(job_id, process)

        for path in (command_input, command_output):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def analyze_job_references(job_id: str, log_path: Path) -> list[dict]:
    references = get_references_for_analysis(job_id)

    if not references:
        return []

    update_job(job_id, stage="encoding_prompt", progress=5)

    analyses = []
    pending = []

    for reference in references:
        analysis = cached_reference_analysis(
            reference["upload_id"],
            reference["role"],
        )

        if analysis is None:
            pending.append(reference)
        else:
            analyses.append(
                {
                    "upload_id": reference["upload_id"],
                    "role": reference["role"],
                    "analysis": analysis,
                }
            )

    if pending:
        try:
            generated = run_reference_analyzer(
                job_id,
                pending,
                log_path,
            )
        except JobCancelled:
            raise
        except Exception as exc:
            raise RuntimeError("Reference image analysis failed.") from exc

        analyses.extend(generated)

    analyses_by_key = {
        (item["upload_id"], item["role"]): item
        for item in analyses
    }

    return [
        analyses_by_key[(reference["upload_id"], reference["role"])]
        for reference in references
    ]


def build_effective_prompt(user_prompt: str, analyses: list[dict]) -> str:
    if not analyses:
        return user_prompt

    final_target_constraint = (
        "The final frame must match the uploaded target image exactly. "
        "Do not substitute, redesign, reinterpret, or replace its symbol. "
    )
    label_characters = sum(
        len(item["role"]) + 2
        for item in analyses
    )
    constraint_characters = sum(
        len(final_target_constraint)
        for item in analyses
        if item["role"] == "final_target"
    )
    analysis_budget = max(
        1,
        (
            MAX_REFERENCE_GUIDANCE_CHARS
            - label_characters
            - len(analyses)
            - constraint_characters
        ) // len(analyses),
    )
    guidance_lines = []

    for item in analyses:
        if item["role"] == "final_target":
            analysis = (
                final_target_constraint
                +
                f"{item['analysis'][:analysis_budget]}"
            )
        else:
            analysis = item["analysis"][:analysis_budget]

        guidance_lines.append(
            f"{item['role'].upper()}: "
            f"{analysis}"
        )

    guidance = "\n".join(guidance_lines)

    return (
        f"{user_prompt}\n\n"
        "REFERENCE GUIDANCE:\n"
        f"{guidance}\n\n"
        "Treat reference guidance as visual constraints. Preserve "
        "distinctive appearance where possible. Do not create a collage "
        "or multiple panels."
    )


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

    start_image = get_primary_reference_path(job_id)

    if start_image is not None:
        if not start_image.is_file():
            raise RuntimeError("Referenced start image is unavailable.")

        command.extend(
            [
                "--image",
                str(start_image),
                "0",
                "1.0",
            ]
        )

    final_target = get_final_target_path(job_id)

    if final_target is not None:
        if not final_target.is_file():
            raise RuntimeError("Referenced final target image is unavailable.")

        command.extend(
            [
                "--image",
                str(final_target),
                "120",
                "1.0",
            ]
        )

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

    start_image = get_primary_reference_path(job_id)

    if start_image is not None:
        if not start_image.is_file():
            raise RuntimeError("Referenced start image is unavailable.")

        env["JOB_IMAGE"] = str(start_image)

        script = r"""
set -Eeuo pipefail

export PYTHONPATH="${PYTHONPATH:-}"

lightx2v_path=/opt/ai-movie/engines/wan/LightX2V
model_path=/opt/ai-movie/models/wan2.2-i2v-base

source \
  /opt/ai-movie/engines/wan/LightX2V/scripts/base/base.sh

exec \
  /opt/ai-movie/engines/wan/LightX2V/.venv/bin/python \
  -m lightx2v.infer \
  --model_cls wan2.2_moe \
  --task i2v \
  --model_path \
    /opt/ai-movie/models/wan2.2-i2v-base \
  --config_json \
    /opt/ai-movie/engines/wan/LightX2V/configs/wan22/extreme/wan_moe_i2v_5090.json \
  --image_path "$JOB_IMAGE" \
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

    start_image = get_primary_reference_path(job_id)

    if start_image is not None:
        if not start_image.is_file():
            raise RuntimeError("Referenced start image is unavailable.")

        command.extend(
            [
                "--image",
                str(start_image),
            ]
        )

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

        reference_guidance = analyze_job_references(job_id, log_path)
        effective_job = dict(job)
        effective_job["prompt"] = build_effective_prompt(
            job["prompt"],
            reference_guidance,
        )

        if job["engine"] == "ltx":
            run_ltx(
                job_id,
                effective_job,
                output,
                log_path,
            )

        elif job["engine"] == "wan":
            run_wan(
                job_id,
                effective_job,
                output,
                log_path,
            )

        elif job["engine"] == "skyreels":
            run_skyreels(
                job_id,
                effective_job,
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
        metadata["references"] = get_job_references(job_id)
        metadata["reference_guidance"] = reference_guidance
        metadata["reference_analyzer"] = (
            "Qwen/Qwen2.5-VL-7B-Instruct"
            if reference_guidance
            else None
        )
        metadata["native_final_target"] = (
            job["engine"] == "ltx"
            and any(
                reference["role"] == "final_target"
                for reference in get_job_references(job_id)
            )
        )

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
                "supports_start_image": True,
                "supports_final_target": True,
                "supports_references": True,
                "max_references": MAX_REFERENCES,
            },
            {
                "id": "wan",
                "name": "Wan 2.2 A14B NVFP4",
                "resolution": "832x480",
                "fps": 16,
                "frames": 81,
                "shot_seconds": 5.06,
                "direct_long": False,
                "supports_start_image": True,
                "supports_final_target": False,
                "supports_references": True,
                "max_references": MAX_REFERENCES,
            },
            {
                "id": "skyreels",
                "name": "SkyReels V2 DF 14B FP8",
                "resolution": "960x544",
                "fps": 24,
                "frames": 57,
                "shot_seconds": 2.375,
                "direct_long": False,
                "supports_start_image": True,
                "supports_final_target": False,
                "supports_references": True,
                "max_references": MAX_REFERENCES,
            },
        ]
    }


@app.post(
    "/api/uploads",
    status_code=201,
    dependencies=[Depends(require_auth)],
)
async def upload_image(file: UploadFile = File(...)):
    upload_id = str(uuid.uuid4())
    temporary_file = UPLOAD_DIR / f"{upload_id}.uploading"
    stored_path: Path | None = None
    size = 0

    try:
        with temporary_file.open("xb") as destination:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Image upload must not exceed 10 MB.",
                    )

                destination.write(chunk)

        try:
            with Image.open(temporary_file) as image:
                image_format = image.format
                image.verify()
        except (
            Image.DecompressionBombError,
            OSError,
            SyntaxError,
            UnidentifiedImageError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail="Upload must be a valid PNG, JPEG, or WebP image.",
            ) from exc

        media_type = IMAGE_MEDIA_TYPES.get(image_format)

        if media_type is None:
            raise HTTPException(
                status_code=422,
                detail="Upload must be a PNG, JPEG, or WebP image.",
            )

        extension = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
        }[media_type]
        stored_file = f"{upload_id}.{extension}"
        stored_path = UPLOAD_DIR / stored_file
        os.replace(temporary_file, stored_path)

        created_at = now()

        with db() as conn:
            conn.execute(
                """
                INSERT INTO uploads (
                    id, stored_file, media_type, size, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    upload_id,
                    stored_file,
                    media_type,
                    size,
                    created_at,
                ),
            )

        return {
            "id": upload_id,
            "media_type": media_type,
            "size": size,
            "created_at": created_at,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to store image upload.",
        ) from exc
    finally:
        await file.close()

        if temporary_file.exists():
            temporary_file.unlink()

        if stored_path is not None and stored_path.exists():
            with db() as conn:
                persisted = conn.execute(
                    "SELECT 1 FROM uploads WHERE id=?",
                    (upload_id,),
                ).fetchone()

            if persisted is None:
                stored_path.unlink()


@app.post(
    "/api/generations",
    status_code=202,
    dependencies=[Depends(require_auth)],
)
def create_generation(request: GenerateRequest):
    if len(request.references) > MAX_REFERENCES:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_REFERENCES} references are supported.",
        )

    reference_ids = [
        reference.upload_id
        for reference in request.references
    ]

    if len(set(reference_ids)) != len(reference_ids):
        raise HTTPException(
            status_code=422,
            detail="Duplicate upload references are not supported.",
        )

    final_target_count = sum(
        reference.role == "final_target"
        for reference in request.references
    )

    if final_target_count > 1:
        raise HTTPException(
            status_code=422,
            detail="At most one final_target reference is supported.",
        )

    if final_target_count and request.engine != "ltx":
        raise HTTPException(
            status_code=422,
            detail="final_target is currently supported only by LTX.",
        )

    job_id = str(uuid.uuid4())
    timestamp = now()

    with db() as conn:
        if reference_ids:
            placeholders = ", ".join("?" for _ in reference_ids)
            uploads = conn.execute(
                f"SELECT id FROM uploads WHERE id IN ({placeholders})",
                reference_ids,
            ).fetchall()

            if len(uploads) != len(reference_ids):
                raise HTTPException(
                    status_code=422,
                    detail="One or more upload IDs are invalid.",
                )

        normalized_references = [
            {
                "upload_id": reference.upload_id,
                "role": reference.role,
            }
            for reference in request.references
        ]

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
                metadata,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps({"references": normalized_references}),
                timestamp,
                timestamp,
            ),
        )

        for position, reference in enumerate(request.references):
            conn.execute(
                """
                INSERT INTO job_references (
                    job_id, upload_id, role, position
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    job_id,
                    reference.upload_id,
                    reference.role,
                    position,
                ),
            )

    job_queue.put(job_id)

    return {
        "id": job_id,
        "engine": request.engine,
        "status": "queued",
        "progress": 0,
        "stage": "queued",
        "references": normalized_references,
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
