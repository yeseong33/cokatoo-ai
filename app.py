from flask import Flask, request, jsonify
from flask_cors import CORS
import torchaudio
import torch
from speechbrain.pretrained import SpeakerRecognition
import os
from pydub import AudioSegment
import tempfile

# Flask 앱 생성
app = Flask(__name__)
CORS(app, supports_credentials=True)

# SpeakerRecognition 모델 로드
recognizer = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

def convert_to_wav(input_file_path):
    try:
        audio = AudioSegment.from_file(input_file_path)
        output_file_path = f"{os.path.splitext(input_file_path)[0]}.wav"
        audio.export(output_file_path, format="wav")
        print(f"Converted {input_file_path} to {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

def save_file_to_user_folder(file_path, user_id, sound_id):
    try:
        user_sound_dir = 'userSound'
        os.makedirs(user_sound_dir, exist_ok=True)
        _, file_extension = os.path.splitext(file_path)

        # 파일 이름에 음원 아이디와 유저 아이디를 반영
        output_file_path = os.path.join(user_sound_dir, f"{sound_id}_{user_id}{file_extension or '.wav'}")
        os.rename(file_path, output_file_path)
        print(f"Saved file to {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def prepare_audio_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            file_path = tmp.name

        if file_path.endswith(('.m4a', '.webm')):
            file_path = convert_to_wav(file_path)
        
        if not file_path:
            raise ValueError("File conversion failed")
        
        return file_path
    except Exception as e:
        print(f"Error preparing audio file: {e}")
        return None

@app.route('/analyze-similarity', methods=['POST'])
def analyze_similarity():
    try:
        # 파일 및 사용자 ID 확인
        file1 = request.files.get('file1')
        file2 = request.files.get('file2')
        sound_id = request.form.get('sound_id')
        user_id = request.form.get('user_id')

        if not file1 or not file2 or not user_id or not sound_id:
            return jsonify({'error': 'file1, file2, user_id, sound_id는 필수 항목입니다.'}), 400

        # 파일 준비
        file1_path = prepare_audio_file(file1)
        file2_path = prepare_audio_file(file2)
        if not file1_path or not file2_path:
            return jsonify({'error': '파일 준비 중 오류 발생'}), 500

        # file2 저장
        file2_path = save_file_to_user_folder(file2_path, user_id, sound_id)
        if not file2_path:
            return jsonify({'error': 'file2 저장 실패'}), 500

        # 음원 파일 로드
        signal1, fs1 = torchaudio.load(file1_path)
        signal2, fs2 = torchaudio.load(file2_path)
        print(f"Loaded file1 (fs={fs1}), file2 (fs={fs2})")

        # 채널을 모노로 변환
        signal1_mono = torch.mean(signal1, dim=0, keepdim=True)
        signal2_mono = torch.mean(signal2, dim=0, keepdim=True)

        # 유사도 분석
        score, prediction = recognizer.verify_batch(signal1_mono, signal2_mono)
        print(f"Similarity score: {score.item()}, Is same: {bool(prediction.item())}")

        # 결과 반환
        return jsonify({
            'similarity_score': score.item(),
            'is_same': bool(prediction.item())
        })
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
