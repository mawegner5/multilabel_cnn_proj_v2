# tests/test_main.py
import unittest
from main import load_data, preprocess_data

class TestMainFunctions(unittest.TestCase):

    def test_load_data(self):
        data = load_data()
        self.assertIn('train', data)
        self.assertIn('val', data)
        self.assertIn('test', data)
        self.assertIn('images', data['train'])
        self.assertIn('labels', data['train'])

    def test_preprocess_data(self):
        data = load_data()
        processed_data = preprocess_data(data['train'], data['val'], data['test'])
        self.assertIn('train', processed_data)
        self.assertIn('val', processed_data)
        self.assertIn('test', processed_data)
        self.assertIn('images', processed_data['train'])
        self.assertIn('labels', processed_data['train'])
        self.assertEqual(processed_data['train']['images'].shape[-1], 3)  # Check image channels

if __name__ == '__main__':
    unittest.main()
