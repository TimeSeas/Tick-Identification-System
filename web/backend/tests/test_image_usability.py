"""Input safety and annotation regressions, without models or database writes."""
import base64
import io
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
from app.routers.analyze import analyze, MAX_FILE_SIZE
from app.routers.analyze import INFERENCE_LOCK
from app.utils.images import load_rgb_image, ImageValidationError
from app.utils.visualize import draw_label, draw_result, load_font


class ImageUsabilityTests(unittest.TestCase):
    def test_shared_model_inference_is_serialized(self):
        entered, release, queued = Event(), Event(), Event()
        counter_lock = Lock()
        active, maximum, calls = 0, 0, 0

        def infer(data):
            nonlocal active, maximum, calls
            with counter_lock:
                active += 1
                maximum = max(maximum, active)
                calls += 1
                first = calls == 1
            if first:
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test did not release first inference')
            with counter_lock:
                active -= 1
            return {'detections': [], 'risk_advice': ''}

        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            pipeline=SimpleNamespace(analyze=infer))))

        class ObservedLock:
            def __enter__(self):
                if INFERENCE_LOCK.locked():
                    queued.set()
                INFERENCE_LOCK.acquire()

            def __exit__(self, *args):
                INFERENCE_LOCK.release()

        def upload():
            file = SimpleNamespace(content_type='image/png', filename='test.png',
                                   file=io.BytesIO(b'image'))
            return analyze(request, file, Mock())

        with patch('app.routers.analyze.INFERENCE_LOCK', ObservedLock()), patch(
                'app.routers.analyze.load_rgb_image'), patch(
                'app.routers.analyze.visualize.draw_result', return_value='image'):
            with ThreadPoolExecutor(max_workers=2) as workers:
                first = workers.submit(upload)
                try:
                    self.assertTrue(entered.wait(2))
                    second = workers.submit(upload)
                    self.assertTrue(queued.wait(2))
                    # First inference remains held while another worker tries to enter.
                    self.assertFalse(second.done())
                    with counter_lock:
                        self.assertEqual(calls, 1)
                finally:
                    release.set()
                self.assertEqual(first.result(timeout=2)['code'], 200)
                self.assertEqual(second.result(timeout=2)['code'], 200)
        self.assertEqual(calls, 2)
        self.assertEqual(maximum, 1)

    def test_large_upload_reads_only_limit_plus_one(self):
        stream = Mock()
        stream.read.return_value = b'x' * (MAX_FILE_SIZE + 1)
        file = SimpleNamespace(content_type='image/jpeg', file=stream)
        request, db = Mock(), Mock()
        self.assertEqual(analyze(request, file, db)['code'], 400)
        stream.read.assert_called_once_with(MAX_FILE_SIZE + 1)
        request.app.state.pipeline.analyze.assert_not_called()
        db.add.assert_not_called()

    def test_invalid_image_returns_friendly_error_without_inference(self):
        file = SimpleNamespace(content_type='image/png', file=io.BytesIO(b'broken image'))
        request, db = Mock(), Mock()
        result = analyze(request, file, db)
        self.assertEqual(result['code'], 400)
        self.assertIn('图片无法解码', result['msg'])
        request.app.state.pipeline.analyze.assert_not_called()
        db.add.assert_not_called()

    def test_exif_orientation_matches_annotation(self):
        image = Image.new('RGB', (120, 80), 'white')
        exif = image.getexif()
        exif[274] = 6
        source = io.BytesIO()
        image.save(source, 'JPEG', exif=exif)
        raw = source.getvalue()
        self.assertEqual(load_rgb_image(raw).size, (80, 120))
        result = draw_result(raw, [{'bbox': [0, 0, 60, 100], 'score': .9,
            'species_cn': '无法确定品种', 'species_slug': 'uncertain',
            'candidate_cn': '美洲花蜱', 'confidence': 75, 'status': 'uncertain'}])
        decoded = Image.open(io.BytesIO(base64.b64decode(result.split(',', 1)[1])))
        self.assertEqual(decoded.size, (80, 120))

    def test_long_candidate_wraps_inside_right_edge(self):
        image = Image.new('RGB', (160, 120), 'white')
        draw = Mock(wraps=ImageDraw.Draw(image))
        font = load_font(16)
        draw_label(draw, '疑似美洲花蜱蜱种尚待人工确认 75.0%', font,
                   (150, 90), image.size, 'black')
        rectangle = draw.rectangle.call_args.args[0]
        self.assertGreaterEqual(rectangle[0], 0)
        self.assertLess(rectangle[2], image.width)
        self.assertLess(rectangle[3], image.height)
        self.assertGreater(draw.text.call_count, 1)
        for call in draw.text.call_args_list:
            (x, _), line = call.args[:2]
            self.assertLessEqual(x + draw.textlength(line, font=font), image.width)

    def test_pixel_limit_prevents_full_decode(self):
        image = Mock(width=6000, height=4000)
        opened = Mock()
        opened.__enter__ = Mock(return_value=image)
        opened.__exit__ = Mock(return_value=False)
        with patch('app.utils.images.Image.open', return_value=opened):
            with self.assertRaisesRegex(ImageValidationError, '2000万'):
                load_rgb_image(b'header')
        image.load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
