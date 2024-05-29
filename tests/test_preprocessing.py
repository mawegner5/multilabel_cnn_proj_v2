import unittest

class TestOutput(unittest.TestCase): 


    def test_preprocess_images(images, self):
        new_images = []
        old_images = []
        for img in processed_images:
            new_img = img * 255.0
            new_images.append(new_img)
        for image in images:
            image = Image.fromarray(image).resize((224, 224))
            image = img_to_array(image)
            old_images.append(image)

        self.assertEqual(new_images, old_images, "Error in image transformations.")
    
    def test_preprocess_labels(labels, label_map, self):
        label_return = preprocess_labels(labels, label_map)
        self.assertIn(1, label_return, "Labels not binarized.")
        

  
if __name__ == '__main__': 
    unittest.main() 