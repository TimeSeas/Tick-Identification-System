import base64
import io
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from smoke_recognition_service import decode_result_image,validate_smoke_database,ROOT


class ServiceImageTests(unittest.TestCase):
    def test_new_experiment_database_is_allowed_but_user_database_is_rejected(self):
        validate_smoke_database(ROOT/'output/optimization_20261015/species_20261027/http_test.db')
        with self.assertRaises(ValueError):validate_smoke_database(ROOT/'web/backend/data.db')
        with self.assertRaises(ValueError):
            validate_smoke_database(ROOT/'output/optimization_20261015/../../user.db')

    def test_export_removes_data_uri_prefix_and_verifies_image(self):
        buffer=io.BytesIO(); Image.new('RGB',(2,2)).save(buffer,format='JPEG')
        raw=buffer.getvalue()
        self.assertEqual(decode_result_image('data:image/jpeg;base64,'+base64.b64encode(raw).decode()),raw)
        with self.assertRaises((ValueError,OSError)):
            decode_result_image('data:image/jpeg;base64,bm90IGFuIGltYWdl')


if __name__=='__main__':
    unittest.main()
