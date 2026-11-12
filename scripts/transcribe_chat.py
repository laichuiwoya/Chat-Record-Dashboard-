"""Add local Whisper transcripts to copies of exported chat Markdown."""

import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import unquote


VOICE = re.compile(r"(?m)^[ \t]*\[语音\]\(<?([^\r\n)]+?)>?\)[ \t]*$")


def transcribe(audio, cli, model, scratch):
    # Stage the file under a short filename for the native tools.
    source = scratch / ("input" + audio.suffix)
    shutil.copyfile(audio, source)
    wav = scratch / "audio.wav"
    transcript = scratch / "result.txt"
    transcript.unlink(missing_ok=True)
    commands = [
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)],
        [str(cli), "-m", str(model), "-f", str(wav), "-l", "zh", "-ng",
         "-otxt", "-of", str(transcript.with_suffix(""))],
    ]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, timeout=180)
    return " ".join(transcript.read_text(encoding="utf-8-sig").split())


def cloud_transcriber(provider, config_path):
    if not config_path.is_file():
        raise ValueError(f"请先按语音转写配置说明创建 {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    providers = config.get("providers") if isinstance(config, dict) else None
    settings = providers.get(provider) if isinstance(providers, dict) else None
    if not isinstance(settings, dict):
        raise ValueError(f"请在 {config_path} 的 providers 中配置 {provider}")

    if provider == "tencent":
        engine = settings.get("engine", "16k_zh")
        names = [settings.get("secret_id_env", "TENCENTCLOUD_SECRET_ID"),
                 settings.get("secret_key_env", "TENCENTCLOUD_SECRET_KEY")]
        if not all(isinstance(name, str) and name for name in names):
            raise ValueError("腾讯云配置须填写有效的凭据环境变量名")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise ValueError("请先设置环境变量 " + "、".join(missing))
        try:
            from tencentcloud.common import credential
            from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
            from tencentcloud.asr.v20190614 import asr_client, models
        except ImportError as exc:
            raise ValueError("请先安装腾讯云 SDK：python -m pip install tencentcloud-sdk-python-asr") from exc
        client = asr_client.AsrClient(credential.Credential(*(os.environ[n] for n in names)), "")

        def recognize(audio):
            size = audio.stat().st_size
            if 4 * ((size + 2) // 3) > 3_000_000:
                raise ValueError("音频超出腾讯云短语音接口的大小限制")
            payload = audio.read_bytes()
            request = models.SentenceRecognitionRequest()
            request.from_json_string(json.dumps({
                "EngSerViceType": engine, "SourceType": 1,
                "VoiceFormat": audio.suffix.lower().lstrip("."),
                "Data": base64.b64encode(payload).decode("ascii"), "DataLen": len(payload),
            }))
            try:
                return client.SentenceRecognition(request).Result or ""
            except TencentCloudSDKException as exc:
                raise ValueError("腾讯云转写请求失败，请检查账号、接口权限和音频限制") from exc

        return recognize

    command = settings.get("command")
    if (not isinstance(command, list) or not command
            or not all(isinstance(part, str) and part for part in command)
            or "{audio}" not in command):
        raise ValueError(f"请为 {provider} 配置转写适配命令，参数中须单独包含 {{audio}}")

    def recognize(audio):
        args = [str(audio) if part == "{audio}" else part for part in command]
        result = subprocess.run(args, check=True, capture_output=True, timeout=180)
        return result.stdout.decode("utf-8-sig")

    return recognize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Archive folder with Markdown and media")
    parser.add_argument("output", type=Path, help="Separate folder for enriched Markdown")
    parser.add_argument("--whisper", type=Path, help="Path to whisper-cli.exe")
    parser.add_argument("--model", type=Path, help="Path to the multilingual GGML model")
    parser.add_argument("--provider", default="local", help="local, tencent, or a configured provider name")
    parser.add_argument("--config", type=Path, help="Override the cloud configuration path")
    args = parser.parse_args()

    root, output = args.input.resolve(), args.output.resolve()
    if not root.is_dir() or output.is_relative_to(root) or root.is_relative_to(output):
        parser.error("Provide an existing archive folder and a separate output folder.")
    files = sorted(root.rglob("*.md"))
    if not files:
        parser.error("No Markdown files found.")
    runtime = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "ChatVoiceTranscription"
    cli = args.whisper or next((runtime / "runtime").rglob("whisper-cli.exe"), None)
    model = (args.model or runtime / "base-complete.bin").resolve()
    cloud = None
    if args.provider != "local":
        try:
            cloud = cloud_transcriber(args.provider, args.config or runtime / "asr-config.json")
        except ValueError as exc:
            parser.error(str(exc))
    cache = {}
    counts = {"files": len(files), "recognized": 0, "unavailable": 0}

    with TemporaryDirectory(prefix="chat-voice-") as temp:
        for md in files:
            relative = md.relative_to(root)
            original = md.read_text(encoding="utf-8-sig")

            def enrich(match):
                reference = unquote(match.group(1))
                audio = (md.parent / reference).resolve()
                if audio not in cache:
                    if not audio.is_relative_to(root) or not audio.is_file():
                        cache[audio] = ("", "归档内未找到语音文件")
                    elif cloud is None and (not cli or not cli.is_file() or not model.is_file() or not shutil.which("ffmpeg")):
                        cache[audio] = ("", "本地转写工具或模型未配置")
                    else:
                        try:
                            text = cloud(audio) if cloud else transcribe(audio, cli.resolve(), model, Path(temp))
                            text = " ".join(text.split())
                            cache[audio] = (text, "未识别到文字" if not text else "")
                        except (OSError, subprocess.SubprocessError, ValueError):
                            cache[audio] = ("", f"{args.provider} 转写失败")
                text, reason = cache[audio]
                counts["recognized" if text else "unavailable"] += 1
                label = "本地语音转写" if cloud is None else "云端语音转写"
                annotation = f"{label}（待核对）：{text}" if text else f"语音未转写：{reason}"
                return f"{match.group(0)}\n\n{annotation}"

            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(VOICE.sub(enrich, original), encoding="utf-8")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
