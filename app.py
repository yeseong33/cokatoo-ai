from flask import Flask, request, jsonify
from flask_cors import CORS
import torchaudio
import torch
from speechbrain.pretrained import SpeakerRecognition
import os
from pydub import AudioSegment  # pydub 라이브러리 추가
import tempfile  # 임시 파일 생성용 라이브러리

# Flask 앱 생성
app = Flask(__name__)
CORS(app, supports_credentials=True)

# SpeakerRecognition 모델 로드
recognizer = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

def convert_to_wav(input_file_path, output_format='wav'):
    """webm 또는 m4a 파일을 wav로 변환"""
    try:
        audio = AudioSegment.from_file(input_file_path)
        output_file_path = input_file_path.rsplit('.', 1)[0] + f'.{output_format}'
        audio.export(output_file_path, format=output_format)
        print(f"Converted {input_file_path} to {output_file_path}")
        return output_file_path
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

# 음원 유사도 분석 API
@app.route('/analyze-similarity', methods=['POST'])
def analyze_similarity():
    print("Request received!", request)
    
    try:
        # 클라이언트로부터 파일 수신
        if 'file1' not in request.files or 'file2' not in request.files:
            return jsonify({'error': '파일이 없습니다.'}), 400

        file1 = request.files['file1']
        file2 = request.files['file2']

        print('file1:', file1, 'file2:', file2)

        # 파일의 내용이 비어있으면 오류 반환
        if file1.filename == '' or file2.filename == '':
            return jsonify({'error': '파일 이름이 비어 있습니다.'}), 400

        # 임시 폴더 생성
        os.makedirs('tmp', exist_ok=True)

        # 임시 파일 저장 (업로드된 파일을 임시 파일에 저장)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file1.filename)[1]) as tmp1:
            file1.save(tmp1.name)
            file1_path = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file2.filename)[1] or '.webm') as tmp2:
            file2.save(tmp2.name)
            file2_path = tmp2.name

        print(f"Saved file1 at {file1_path}")
        print(f"Saved file2 at {file2_path}")

        # 파일 형식 변환 (m4a, webm -> wav)
        if file1_path.endswith('.m4a') or file1_path.endswith('.webm'):
            file1_path = convert_to_wav(file1_path)
            if not file1_path:
                return jsonify({'error': 'file1 변환 실패'}), 500

        if file2_path.endswith('.m4a') or file2_path.endswith('.webm'):
            file2_path = convert_to_wav(file2_path)
            if not file2_path:
                return jsonify({'error': 'file2 변환 실패'}), 500

        print(f"Converted file1 to {file1_path}")
        print(f"Converted file2 to {file2_path}")

        # 음원 파일 로드
        try:
            signal1, fs1 = torchaudio.load(file1_path)
            signal2, fs2 = torchaudio.load(file2_path)
            print(f"Loaded file1 with sample rate {fs1}")
            print(f"Loaded file2 with sample rate {fs2}")
        except Exception as e:
            return jsonify({'error': f'음원 파일 로드 실패: {str(e)}'}), 500

        # 채널을 모노로 변환
        signal1_mono = torch.mean(signal1, dim=0, keepdim=True)
        signal2_mono = torch.mean(signal2, dim=0, keepdim=True)

        # 유사도 분석 수행
        score, prediction = recognizer.verify_batch(signal1_mono, signal2_mono)

        print('similarity_score:', score.item(), 'is_same:', bool(prediction.item()))

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
