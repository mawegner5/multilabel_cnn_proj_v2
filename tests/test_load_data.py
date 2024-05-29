import unittest

class TestOutput(unittest.TestCase): 
    # test function to test if values are present
    def test_load_data(self): 
        result = load_data()
        # error message in case if test case got failed 
        self.assertNotEqual(result, None, "Data did not load correctly.") 
        self.assertNotEqual(result['train'], None, "Training data did not load.")
        self.assertNotEqual(result['val'], None, "Value data did not load.")
        self.assertNotEqual(result['test'], None, "Test data did not load.")
  
if __name__ == '__main__': 
    unittest.main()