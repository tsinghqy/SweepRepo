import unittest
from unittest.mock import MagicMock, patch

from sweeplocal import (calculate_average_embedding, find_relevant_chunks,
                        generate_embedding_for_request, process_user_request)


class TestSweepLocal(unittest.TestCase):

    @patch('sweeplocal.word_tokenize')
    @patch('sweeplocal.calculate_average_embedding')
    def test_generate_embedding_for_request(self, mock_calculate_average_embedding, mock_word_tokenize):
        mock_word_tokenize.return_value = ['dummy', 'tokens']
        mock_calculate_average_embedding.return_value = [0.5] * 100
        word2vec_model = MagicMock()
        word2vec_model.vector_size = 100
        text_splitter = MagicMock()
        text_splitter.split_text.return_value = ['dummy text']
        result = generate_embedding_for_request('dummy request', word2vec_model, text_splitter)
        self.assertEqual(len(result), 100)

    def test_calculate_average_embedding(self):
        embeddings = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        word2vec_model = MagicMock()
        word2vec_model.vector_size = 3
        result = calculate_average_embedding(embeddings, word2vec_model)
        self.assertEqual(result, [4, 5, 6])

    @patch('sweeplocal.NearestNeighbors')
    def test_find_relevant_chunks(self, mock_NearestNeighbors):
        mock_knn_model = MagicMock()
        mock_NearestNeighbors.return_value = mock_knn_model
        mock_knn_model.kneighbors.return_value = ([], [[0, 1, 2, 3, 4]])
        embedding_store = {'file1': ['chunk1', 'chunk2'], 'file2': ['chunk3', 'chunk4', 'chunk5']}
        user_embedding = [0.5] * 100
        result = find_relevant_chunks(user_embedding, embedding_store, k=5)
        self.assertEqual(len(result), 5)

    @patch('sweeplocal.find_relevant_chunks')
    @patch('sweeplocal.generate_modified_code')
    def test_process_user_request(self, mock_generate_modified_code, mock_find_relevant_chunks):
        mock_find_relevant_chunks.return_value = ['relevant chunk']
        mock_generate_modified_code.return_value = 'modified code'
        embedding_store = {'file1': ['chunk1', 'chunk2']}
        result = process_user_request('dummy request', embedding_store)
        self.assertIn('modified code', result)
        mock_find_relevant_chunks.return_value = []
        result = process_user_request('dummy request', embedding_store)
        self.assertIn('No relevant code chunks found', result)

if __name__ == '__main__':
    unittest.main()
