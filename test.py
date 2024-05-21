import os
import unittest
import tempfile
import json
import requests

class TestApp(unittest.TestCase):
    def setUp(self):
        self.app_url = "http://localhost:8081"  
        self.headers = {'Content-Type': 'application/json'}

    def tearDown(self):
        pass

    def test_index(self):
        # Test index route
        response = requests.post(self.app_url + '/', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue('book_name' in data)
        self.assertTrue('author' in data)
        self.assertTrue('image' in data)
        self.assertTrue('votes' in data)
        self.assertTrue('rating' in data)

    def test_recommend_books(self):
        # Test recommend_books route with hardcoded data
        data = {'user_input': '1984'}  # Hardcoded input
        response = requests.post(self.app_url + '/recommend_books', data=json.dumps(data), headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue('book_name' in data)
        self.assertTrue('author' in data)
        self.assertTrue('image' in data)

if __name__ == '__main__':
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        print("Tests successful")
    else:
        print("Test failed")
