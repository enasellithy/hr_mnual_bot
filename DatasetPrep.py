import re 
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from TextClean import TextCleaner

class DatasetPrParation:
    
    def __init__(self,
                    chunk_size: int = 50,
                    chunk_overlap: int = 10,
                    keep_numbers: bool = False,
                    custom_stopwords: Optional[dict] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_numbers = keep_numbers
        self.custom_stopwords = custom_stopwords
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_cleaner = TextCleaner(keep_numbers=keep_numbers)
       
        if custom_stopwords:
            for lang, words in custom_stopwords.items():
                if hasattr(self.text_cleaner, "base_stopwords") and lang in self.text_cleaner.base_stopwords:
                    self.text_cleaner.base_stopwords[lang].extend(words)
    
    def load_document(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileExistsError("File not found")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    
    def clean_document(self, text: str, lang: Optional[str] = None) -> str:
        return self.text_cleaner.clean(text, lang)
    
    def chunk_document(self, text: str) -> List[str]: 
        words, chunks, i = text.split(), [], 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunks.append(" ".join(words[i:end]))
            i += (self.chunk_size - self.chunk_overlap)
        return chunks 
    
    def prepare_chunks_with_metadata(self, text: str) -> List[Tuple[str, dict]]:
        cleaned_text = self.clean_document(text)
        chunks = self.chunk_document(cleaned_text)
        chunks_with_metadata = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                "chunk_id": idx,
                "word_count": len(chunk.split()),
                "char_count": len(chunk),
                "language": self.text_cleaner.detect_language(chunk),
                "source": "hr_manual"
            }
            chunks_with_metadata.append((chunk, metadata)) 
        return chunks_with_metadata
    
    def prepare_from_file(self, file_path: str) -> List[Tuple[str, dict]]:
        print("Prepare file")
        raw_text = self.load_document(file_path)
        cleaned_text = self.clean_document(raw_text)
        chunks = self.chunk_document(cleaned_text)
        prepared_chunks = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                "chunk_id": idx,
                "word_count": len(chunk.split()),
                "char_count": len(chunk),
                "language": self.text_cleaner.detect_language(chunk),
                "source": Path(file_path).name
            }
            prepared_chunks.append((chunk, metadata))
        self.analyze_dataset(prepared_chunks)
        return prepared_chunks
    
    def analyze_dataset(self, prepared_chunks: List[Tuple[str, dict]]):
        if not prepared_chunks:
            print("Not found data")
        total_chunks = len(prepared_chunks)
        total_words = sum(meta["word_count"] for _, meta in prepared_chunks)
        avg_words = total_words / total_chunks if total_chunks > 0 else 0
        languages = {}
        for _, meta in prepared_chunks:
            lang = meta["language"]
            languages[lang] = languages.get(lang, 0) + 1
        
        print("\n📊 تحليل مجموعة البيانات:")
        print(f"   العدد الإجمالي للقطع: {total_chunks}")
        print(f"   العدد الإجمالي للكلمات: {total_words}")
        print(f"   متوسط الكلمات في القطعة: {avg_words:.1f}")
        print(f"   توزيع اللغات: {languages}")
        
        # عرض عينة من القطع
        print(f"\n🔍 عينة من القطع (الـ 3 الأولى):")
        for i, (chunk, meta) in enumerate(prepared_chunks[:3]):
            print(f"   [{meta['chunk_id']}] {chunk[:80]}...")
    
    def save_prepared_chunks(self, prepared_chunks: List[Tuple[str, dict]], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk, metadata in prepared_chunks:
                f.write(f"=== Chunk {metadata['chunk_id']} ===\n")
                f.write(f"Language: {metadata['language']}\n")
                f.write(f"Words: {metadata['word_count']}\n")
                f.write(f"Source: {metadata['source']}\n")
                f.write(f"{chunk}\n\n")
        print(f"💾 تم حفظ {len(prepared_chunks)} قطعة في: {output_file}")
        
        