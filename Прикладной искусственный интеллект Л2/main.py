
import sys
import json
from googletrans import Translator

def translate_with_transcription(text, source='en', target='zh'):
    translator = Translator()
    try:
        result = translator.translate(text, src=source, dest=target)
        print(result.extra_data)
        return {
            'translation': result.text,
            'pronunciation': result.pronunciation,
            'source': source,
            'target': target
        }
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # Получаем аргументы
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No text provided'}))
        sys.exit(1)

    text = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else 'en'
    target = sys.argv[3] if len(sys.argv) > 3 else 'zh'

    result = translate_with_transcription(text, source, target)
    print(json.dumps(result))