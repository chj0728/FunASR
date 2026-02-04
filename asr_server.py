import os
import shutil
import uuid
import argparse
from fastapi import FastAPI, File, UploadFile
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# ===============================
# 1. FastAPI 初始化
# ===============================
app = FastAPI(title="FunASR SenseVoice ASR Server")

TMP_DIR = "/tmp/asr_upload"
os.makedirs(TMP_DIR, exist_ok=True)


# ===============================
# 4. ASR 接口
# ===============================
@app.post("/asr")
async def asr(file: UploadFile = File(...)):
    """
    接收 wav 文件，返回识别文本
    """
    # 1. 保存临时文件
    filename = file.filename or ""
    suffix = os.path.splitext(filename)[-1]
    wav_path = os.path.join(TMP_DIR, f"{uuid.uuid4()}{suffix}")

    with open(wav_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. ASR 推理
        res = model.generate(
            input=wav_path,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            # batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
            hotword="智己",
            spk_diarization=True,  # 启用说话人分离
            max_speakers=3,  # 根据实际人数设置
            chunk_size=500,  # 增大推理块提升速度
            batch_size_s=300,  # 批量处理优化
        )
        # for r in res[0]["sentence_info"]:
        #     print(r)
        #     print("-----")
        # for speaker_text in res[0]["speaker_texts"]:
        #     print(f"Speaker {speaker_text['speaker_id']}: {speaker_text['text']}")

        key = res[0].get("key", None)
        text = rich_transcription_postprocess(res[0]["text"])
        sentences = []
        for sentence_info in res[0]["sentence_info"]:
            sentence = {
                "start": sentence_info["start"],
                "end": sentence_info["end"],
                "timestamp": sentence_info["timestamp"],
                "text": rich_transcription_postprocess(sentence_info["text"]),
                "speaker_id": sentence_info.get("spk", None),
            }
            sentences.append(sentence)

        return {
            "code": 0,
            "msg": "success",
            "key": key,
            "text": text,
            "sentences": sentences,
        }

    except Exception as e:
        return {
            "code": -1,
            "msg": str(e),
            "key": None,
            "text": "",
            "sentences": [],
        }

    finally:
        # 3. 清理临时文件
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ===============================
# 5. 启动服务
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=2002)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/workspace/ASR/pretrained_models/paraformer-zh",
        help="local path or modelscope repo id",
    )
    parser.add_argument(
        "--vad_model_dir",
        type=str,
        default="/workspace/ASR/pretrained_models/speech_fsmn_vad",
        help="local path or modelscope repo id",
    )
    parser.add_argument(
        "--spk_model_dir",
        type=str,
        default="/workspace/ASR/pretrained_models/cam++",
        help="local path or modelscope repo id",
    )
    parser.add_argument(
        "--punc_model_dir",
        type=str,
        default="/workspace/ASR/pretrained_models/punc_ct",
        help="local path or modelscope repo id",
    )
    parser.add_argument(
        "--max_single_segment_time",
        type=int,
        default=30000,
        help="max_single_segment_time",
    )

    args = parser.parse_args()

    model = AutoModel(
        model=args.model_dir,
        vad_model=args.vad_model_dir,
        vad_kwargs={"max_single_segment_time": args.max_single_segment_time},
        spk_model=args.spk_model_dir,
        punc_model=args.punc_model_dir,
        disable_update=True,
        device="cuda:0",
    )

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port)
