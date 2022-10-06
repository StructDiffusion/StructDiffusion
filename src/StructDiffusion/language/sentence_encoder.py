from sentence_transformers import SentenceTransformer

class SentenceBertEncoder:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, sentences):
        #Our sentences we like to encode
        # sentences = ['This framework generates embeddings for each input sentence',
        #     'Sentences are passed as a list of string.',
        #     'The quick brown fox jumps over the lazy dog.']
        #Sentences are encoded by calling model.encode()

        embeddings = self.model.encode(sentences)
        # print(embeddings.shape)
        return embeddings


if __name__ == "__main__":
    sentence_encoder = SentenceBertEncoder()
    embedding = sentence_encoder.encode(["this is cool!"])
    print(embedding.shape)