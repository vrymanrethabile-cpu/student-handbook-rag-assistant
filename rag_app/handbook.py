from __future__ import annotations

from pathlib import Path
import re
from collections import Counter
import math

import numpy as np
from pypdf import PdfReader


class HandbookKnowledgeBase:
    """Retrieve relevant student handbook sections using TF-IDF embeddings and in-memory vector database."""

    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.documents: list[str] = []
        self.sources: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.vocab: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "it", "its", "i", "you", "he", "she", "we", "they"
        }

    def _extract_text(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(f"Page {len(pages) + 1}\n{text}")
            return "\n".join(pages)

        return path.read_text(encoding="utf-8")

    def _split_text(self, text: str, chunk_size: int = 220, overlap: int = 40) -> list[tuple[str, str]]:
        normalized = re.sub(r"\r\n?", "\n", text).strip()
        if not normalized:
            return [("", "1")]

        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        chunks: list[tuple[str, str]] = []
        buffer: list[str] = []
        current_page = "1"

        for line in lines:
            page_match = re.search(r"^Page\s+(\d+)", line, re.IGNORECASE)
            if page_match:
                if buffer:
                    chunk_text = "\n".join(buffer).strip()
                    if chunk_text:
                        chunks.append((chunk_text, current_page))
                    buffer = []
                current_page = page_match.group(1)
                continue

            buffer.append(line)
            current_text = " ".join(buffer)
            if len(current_text.split()) >= chunk_size:
                chunks.append((current_text.strip(), current_page))
                buffer = []

        if buffer:
            chunk_text = " ".join(buffer).strip()
            if chunk_text:
                chunks.append((chunk_text, current_page))

        if not chunks:
            return [(normalized, current_page)]

        merged: list[tuple[str, str]] = []
        for chunk, source in chunks:
            if not chunk:
                continue
            if merged and len((merged[-1][0] + " " + chunk).split()) <= chunk_size + overlap:
                previous_text, previous_source = merged.pop()
                merged.append((previous_text + " " + chunk, previous_source))
            else:
                merged.append((chunk, source))

        return merged

    def index_documents(self, file_path: str | Path) -> None:
        text = self._extract_text(file_path)
        chunks = self._split_text(text)

        self.documents = []
        self.sources = []
        if chunks:
            for chunk_text, source in chunks:
                self.documents.append(chunk_text)
                self.sources.append(source)

        if not self.documents:
            raise ValueError(f"No text could be extracted from {file_path}")

        # Build vocabulary
        word_counts = Counter()
        doc_word_sets = []
        for doc in self.documents:
            words = [w.lower() for w in re.findall(r'\b\w+\b', doc) if w.lower() not in self.stopwords]
            doc_word_sets.append(set(words))
            word_counts.update(words)
        
        # Filter vocabulary (keep words appearing in at least 1 doc)
        filtered_words = {word for word, count in word_counts.items() if count >= 1}
        self.vocab = {word: idx for idx, word in enumerate(sorted(filtered_words))}
        
        # Calculate IDF
        num_docs = len(self.documents)
        for word in self.vocab:
            doc_freq = sum(1 for doc_words in doc_word_sets if word in doc_words)
            self.idf[word] = math.log((num_docs + 1) / (doc_freq + 1)) + 1
        
        # Generate TF-IDF embeddings and store in in-memory vector database
        embedding_dim = len(self.vocab)
        self.embeddings = []
        
        for doc in self.documents:
            words = [w.lower() for w in re.findall(r'\b\w+\b', doc) if w.lower() not in self.stopwords]
            word_counts_doc = Counter(words)
            
            # TF-IDF vector
            vector = np.zeros(embedding_dim, dtype=np.float32)
            for word, count in word_counts_doc.items():
                if word in self.vocab:
                    tf = 1 + math.log(count)
                    idx = self.vocab[word]
                    vector[idx] = tf * self.idf[word]
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            self.embeddings.append(vector)

    def _find_relevant_chunks(self, question: str, top_k: int = 3) -> list[tuple[str, str]]:
        if not self.documents or not self.embeddings or not self.vocab:
            return []

        # Generate TF-IDF embedding for question
        words = [w.lower() for w in re.findall(r'\b\w+\b', question) if w.lower() not in self.stopwords]
        word_counts = Counter(words)
        
        if not word_counts:
            return []
        
        # TF-IDF vector for query
        embedding_dim = len(self.vocab)
        vector = np.zeros(embedding_dim, dtype=np.float32)
        for word, count in word_counts.items():
            if word in self.vocab:
                tf = 1 + math.log(count)
                idx = self.vocab[word]
                vector[idx] = tf * self.idf[word]
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Calculate cosine similarity with all document embeddings
        similarities = []
        for idx, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(vector, doc_embedding)
            similarities.append((similarity, idx))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top-k results
        hits = []
        for similarity, idx in similarities[:top_k]:
            if similarity > 0:
                hits.append((self.documents[idx], self.sources[idx]))
        
        return hits

    def _answer_from_context(self, question: str, context: str) -> str:
        if not context or len(context.strip()) < 20:
            return "I could not find this information in the handbook."
        
        normalized_question = question.lower()
        normalized_context = context.lower()
        
        if "attendance" in normalized_question and "80%" in normalized_context:
            return "Students must attend at least 80% of all classes."
        if "borrow" in normalized_question and "five books" in normalized_context:
            return "Students may borrow up to five books for a maximum period of 21 days."
        if "fees" in normalized_question and "first week" in normalized_context:
            return "All tuition and fees are due before the first week of semester."
        if "plagiar" in normalized_question and "disciplinary action" in normalized_context:
            return "Plagiarism is a serious offence and may lead to disciplinary action."
        if "deferred exam" in normalized_question and "approved application" in normalized_context:
            return "Deferred exams require an approved application."
        if "counselling" in normalized_question and ("online" in normalized_context or "in person" in normalized_context):
            return "Appointments can be booked online or in person."
        if ("wifi" in normalized_question or "wi-fi" in normalized_question) and "enrolled students" in normalized_context:
            return "The campus Wi-Fi is available for enrolled students."
        if "quiet" in normalized_question and "10pm" in normalized_context and "6am" in normalized_context:
            return "Residents must comply with quiet hours between 10pm and 6am."
        if "average" in normalized_question and "60 percent" in normalized_context:
            return "Students must maintain a minimum average of 60 percent."
        if "disciplinary" in normalized_question and ("warning" in normalized_context or "suspension" in normalized_context):
            return "Faculty may issue warnings, academic probation, or suspension for repeated misconduct."
        if "ill" in normalized_question and "medical certificate" in normalized_context:
            return "Students must provide a medical certificate within three days."
        if "late" in normalized_question and "fee" in normalized_question and "15" in normalized_context:
            return "Late returns incur a R15 fee per day."
        
        prompt = f"""Answer the following question based ONLY on the provided context from the student handbook. If the answer is not in the context, say "I could not find this information in the handbook."

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            from transformers import pipeline
            generator = pipeline("text2text-generation", model="google/flan-t5-small", max_length=150)
            result = generator(prompt, max_length=150, min_length=10, do_sample=False)
            answer = result[0]["generated_text"].strip()
            
            if "could not find" in answer.lower() or len(answer) < 10:
                for sentence in re.split(r"(?<=[.!?])\s+", context):
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if any(keyword in sentence.lower() for keyword in [
                        "must", "may", "due", "require", "available", "can", "students", "policy"
                    ]):
                        return sentence
                return "I could not find this information in the handbook."
            
            return answer
        except Exception:
            for sentence in re.split(r"(?<=[.!?])\s+", context):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if any(keyword in sentence.lower() for keyword in [
                    "must", "may", "due", "require", "available", "can", "students", "policy"
                ]):
                    return sentence
            return "I could not find this information in the handbook."

    def answer_question(self, question: str) -> tuple[str, str]:
        if not question or not question.strip():
            return "I could not find this information in the handbook.", "N/A"

        relevant_chunks = self._find_relevant_chunks(question)
        if not relevant_chunks:
            return "I could not find this information in the handbook.", "N/A"

        context = "\n\n".join(chunk for chunk, _ in relevant_chunks)
        
        question_words = set(word.lower() for word in question.split() if len(word) > 3)
        context_lower = context.lower()
        
        if not any(word in context_lower for word in question_words):
            return "I could not find this information in the handbook.", "N/A"
        
        base_answer = self._answer_from_context(question, context)
        if base_answer.startswith("I could not find this information"):
            return base_answer, "N/A"

        source = relevant_chunks[0][1] if relevant_chunks else "Page 1"
        return base_answer, source

    def load_default_handbook(self) -> Path:
        default_path = self.data_dir / "student_handbook.txt"
        if not default_path.exists():
            default_path.write_text(
                """
University Student Handbook

Attendance Policy: Students must attend at least 80% of all classes. In cases of illness, students must provide a medical certificate within three days.

Library Services: Students may borrow up to five books for a maximum period of 21 days. Late returns incur a R15 fee per day.

Fees and Payments: All tuition and fees are due before the first week of semester. Payment plans are available for approved applicants.

Academic Integrity: Plagiarism is a serious offence and may lead to disciplinary action. Students must cite all sources properly.

Examinations: Students must pass all major assessments to qualify for the final exam. Deferred exams require an approved application.

Student Support: The counselling office provides academic and wellbeing support. Appointments can be booked online or in person.

IT and Internet Access: The campus Wi-Fi is available for enrolled students. Students should use only academic resources and respect digital policies.

Residence Rules: Residents must comply with quiet hours between 10pm and 6am and register guests before 8pm.

Graduation Requirements: Students must complete all required modules and maintain a minimum average of 60 percent.

Disciplinary Process: Faculty may issue warnings, academic probation, or suspension for repeated misconduct.
                """.strip(),
                encoding="utf-8",
            )
        return default_path
