from flask import Flask, request, jsonify, Response, render_template
from faster_whisper import WhisperModel
import os, time, uuid
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import threading

app = Flask(__name__)

UPLOAD_DIR = "temp_sounds"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Loading model...")

model = WhisperModel(
    "model_small",
    device="cpu",
    compute_type="int8",
    cpu_threads=2,             # تحسين السرعة
    num_workers=2            # معالجة متوازية أسرع             

)

print("Model loaded.")

progress = {"value": 0, "started": False}
progress_lock = threading.Lock()


def get_audio_duration(path):
    """قراءة مدة الصوت لجميع الصيغ (mp3, wav, ogg...)"""
    try:
        audio = AudioSegment.from_file(path)
        return len(audio) / 1000.0
    except Exception:
        return 0


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/progress')
def progress_stream():
    def generate():
        yield "data:start\n\n"
        while True:
            with progress_lock:
                value = progress["value"]

            yield f"data:{value}\n\n"
            time.sleep(0.1)   # تحديث أسرع

            if value >= 100:
                break

    response = Response(generate(), mimetype='text/event-stream')
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route('/transcribe', methods=['POST'])
def transcribe():
    with progress_lock:
        progress["value"] = 0
        progress["started"] = True

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{filename}")
    file.save(filepath)

    total_duration = get_audio_duration(filepath)

    text = ""
    try:
        segments, info = model.transcribe(
            filepath,
            language="ar",
            beam_size=3,
            best_of=1,
            vad_filter=True,
            temperature=0,
            condition_on_previous_text=True,
            vad_parameters={"min_silence_duration_ms": 250}
        )

        last_percent = 0

        for seg in segments:
            text += seg.text.strip() + " "

            if total_duration > 0:
                percent = int((seg.end / total_duration) * 100)
                last_percent = max(last_percent, percent)

                with progress_lock:
                    progress["value"] = min(percent, 100)

        # ضمان الوصول لـ 100%
        if last_percent < 98:
            with progress_lock:
                progress["value"] = 100

        return jsonify({"text": text.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            os.remove(filepath)
        except:
            pass



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)