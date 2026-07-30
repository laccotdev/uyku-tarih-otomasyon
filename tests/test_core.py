import importlib.util
import sys
import types
from pathlib import Path

# External API SDK stubs for offline unit tests.
google = types.ModuleType('google')
genai = types.ModuleType('google.genai')
genai_types = types.ModuleType('google.genai.types')

class Dummy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

genai.Client = object
genai_types.GenerateContentConfig = Dummy
genai_types.SpeechConfig = Dummy
genai_types.VoiceConfig = Dummy
genai_types.PrebuiltVoiceConfig = Dummy
genai.types = genai_types
google.genai = genai
sys.modules['google'] = google
sys.modules['google.genai'] = genai
sys.modules['google.genai.types'] = genai_types

MODULE_PATH = Path(__file__).resolve().parents[1] / 'src' / 'agent.py'
spec = importlib.util.spec_from_file_location('agent', MODULE_PATH)
agent = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(agent)


def test_slugify():
    assert agent.slugify("İstanbul'un Fethi") == 'istanbul-un-fethi'


def test_project_id_stable():
    assert agent.project_id('Odysseus yolculuğu', 60) == agent.project_id('Odysseus yolculuğu', 60)


def test_scene_chunks():
    text = 'Birinci olay başladı. İkinci karar alındı. Üçüncü aşama tamamlandı. Dördüncü sonuç ortaya çıktı.'
    chunks = agent.scene_chunks(text, 4)
    assert len(chunks) == 4
    assert 'Birinci' in ' '.join(chunks)
    assert 'Dördüncü' in ' '.join(chunks)


def test_durations():
    scenes = [{'narration':'bir iki üç'}, {'narration':'dört beş altı yedi'}]
    values = agent.scene_durations(scenes, 100.0)
    assert abs(sum(values) - 100.0) < 0.001
