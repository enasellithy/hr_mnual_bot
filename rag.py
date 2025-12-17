from pathlib import Path
import numpy as np
import faiss
import os
from langchain_google_genai import GoogleGenerativeAI
from google import genai
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from TextClean import TextCleaner
from DatasetPrep import DatasetPrParation  # <-- تم التصحيح
load_dotenv()

# 🔴🔴🔴 FIX: تصحيح اسم النموذج (هذا سبب الخطأ 404)
model_name = "models/gemma-3-4b-it"  # أو "gemini-1.5-flash"
embedding_model = "all-MiniLM-L6-v2"
data_path = "hr_manual.txt"
GEMINI_KEY=""
client = genai.Client(api_key=GEMINI_KEY)
model = client.models.list()
for model in client.models.list():
    print(f"Name: {model.name}")
    print(f"Display Name: {model.display_name}")
    print(f"Description: {model.description}")
    print(f"Input Token Limit: {model.input_token_limit}")
    print(f"Output Token Limit: {model.output_token_limit}")
    print(f"Supported Actions: {model.supported_actions}")
    print("-" * 50)

llm = GoogleGenerativeAI(
    model=model_name,
    google_api_key=GEMINI_KEY
)

class Chunker:
    def __init__(self, max_words: int = 50, overlap: int = 10):
        self.max_words = max_words
        self.overlap = overlap
    
    def chunks(self, text: str):
        words = text.split()
        out = []
        i = 0
        while i < len(words):
            end = min(i + self.max_words, len(words))
            chunk = " ".join(words[i:end])
            out.append(chunk)
            i += (self.max_words - self.overlap)
        return out

class Embedder:
    def __init__(self, model: str = embedding_model):
        self.model = SentenceTransformer(model, device="cpu")
    
    def fit_transform(self, chunks):
        self.chunks = chunks
        self.emb_norm = self.model.encode(chunks, normalize_embeddings=True)
        return self.emb_norm.astype("float32")
    
    def transform(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).astype("float32")
    
class BM25Retriever:
    def __init__(self, chunks):
        tokenized = [c.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.chunks = chunks
    
    def search(self, query: str, top_k: int = 3):
        scores = self.bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i]), self.chunks[int(i)]) for i in top_idx]
    
class Retriever:
    def __init__(self, emb_norm):
        dim = emb_norm.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb_norm)
    
    def search(self, query_vec, top_k: int = 10):
        query_vec = np.array(query_vec).astype("float32")
        if len(query_vec.shape) == 1:
            query_vec = query_vec.reshape(1, -1)
        sims, idx = self.index.search(query_vec, top_k)
        return sims, idx

class RAGPipeline:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.chunker = Chunker()
        self.embedder = Embedder()
        # ✅ تصحيح: استخدام DatasetPrParation بدلاً من DatasetPreparation
        self.dataset_prep = DatasetPrParation(  # <-- تم التصحيح
            chunk_size=50,
            chunk_overlap=10,
            keep_numbers=False
        )
        self.chunks = []
        self.chunks_metadata = []
        self.retriever = None
    
    def build(self):
        prepared_data = self.dataset_prep.prepare_from_file(str(self.data_path))
        
        # تأكد من أن prepared_data يحتوي على (chunk, metadata)
        if prepared_data and isinstance(prepared_data[0], dict):
            # إذا كانت بيانات فقط بدون chunks
            self.chunks_metadata = prepared_data
            # نحتاج لإنشاء chunks من البيانات المتاحة
            self.chunks = [f"Chunk {meta['chunk_id']}" for meta in prepared_data]
        else:
            # إذا كانت (chunk, metadata)
            self.chunks = [chunk for chunk, _ in prepared_data]
            self.chunks_metadata = [metadata for _, metadata in prepared_data]
        
        emb_norm = self.embedder.fit_transform(self.chunks)
        self.retriever = Retriever(emb_norm)
    
    def query(self, text: str, top_k: int = 10):
        cleaned_query = self.dataset_prep.text_cleaner.clean(text)
        q_vec = self.embedder.transform([cleaned_query])
        sims, idx = self.retriever.search(q_vec, top_k)
        
        results = []
        for j, i in enumerate(idx[0]):
            if i >= 0 and i < len(self.chunks):
                results.append((int(i), float(sims[0, j]), self.chunks[int(i)]))
        return results
    
    def get_chunk_info(self, chunk_id: int) -> dict:
        if 0 <= chunk_id < len(self.chunks_metadata):
            return self.chunks_metadata[chunk_id]
        return {}

def build_prompt(question: str, top_chunks):
    chunk_texts = [chunk for _, _, chunk in top_chunks]
    context = "\n\n".join(chunk_texts)
    
    lang = "ar" if any("\u0600" <= c <= "\u06FF" for c in question) else "en"
    not_found = ("الإجابة غير موجودة في الدليل." if lang == "ar" else
                 "This information is not available in the manual.")
    
    return f"""You are a helpful HR assistant.
- Answer concisely in the SAME language the user used ({'Arabic' if lang=='ar' else 'English'}).
- If the context lacks the exact information, reply ONLY with: "{not_found}"
Context from HR manual:
{context}
User question: {question}
Answer:"""

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    rag = RAGPipeline(data_path)
    print("Building RAG pipeline...")
    rag.build()
    print("RAG pipeline built successfully!")
    
    print("\nBot: Hi, I'm an HR assistant bot. How can I help you? (Type 'exit' to quit)")
    print("Commands: 'info <chunk_id>' to get chunk info, 'stats' to show dataset stats")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        elif user_input.lower() in ["hello", "hi", "hey"]:
            print(f"Bot: Hello! How can I assist you today?")
        elif user_input.lower() == "stats":
            print(f"\n📊 System Statistics:")
            print(f"   Total chunks: {len(rag.chunks)}")
            print(f"   Total metadata entries: {len(rag.chunks_metadata)}")
            if rag.chunks_metadata:
                languages = {}
                for meta in rag.chunks_metadata:
                    lang = meta.get("language", "unknown")
                    languages[lang] = languages.get(lang, 0) + 1
                print(f"   Languages: {languages}")
        elif user_input.lower().startswith("info "):
            try:
                chunk_id = int(user_input.split()[1])
                info = rag.get_chunk_info(chunk_id)
                if info:
                    print(f"\n📄 Chunk {chunk_id} Info:")
                    for key, value in info.items():
                        print(f"   {key}: {value}")
                    if chunk_id < len(rag.chunks):
                        print(f"   Content: {rag.chunks[chunk_id]}")
                else:
                    print(f"Bot: Chunk {chunk_id} not found.")
            except:
                print("Bot: Usage: 'info <chunk_id>'")
        else:
            results = rag.query(user_input, top_k=3)
            
            if not results:
                print("Bot: I couldn't find relevant information in the HR manual.")
                continue
            
            prompt = build_prompt(user_input, results)
            try:
                answer = llm.invoke(prompt)
                print(f"\nBot: {answer.strip()}")
                
                print(f"\n📚 Sources used:")
                for i, (chunk_id, score, chunk) in enumerate(results):
                    print(f"   {i+1}. Chunk {chunk_id} (Score: {score:.3f})")
            except Exception as e:
                print(f"Error: {e}")
                print("Please check your Gemini API key and model name.")