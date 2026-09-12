use std::collections::{HashMap, HashSet};

use crate::{
    document::DocumentChunk,
    error::{Result, TinyError},
};

use super::{RetrievedChunk, Retriever};

const DEFAULT_K1: f32 = 1.2;
const DEFAULT_B: f32 = 0.75;

#[derive(Debug, Clone)]
struct IndexedChunk {
    chunk: DocumentChunk,
    term_frequencies: HashMap<String, usize>,
    term_count: usize,
}

#[derive(Debug, Clone)]
pub struct LexicalRetriever {
    chunks: Vec<IndexedChunk>,
    document_frequencies: HashMap<String, usize>,
    average_document_length: f32,
    k1: f32,
    b: f32,
}

impl LexicalRetriever {
    pub fn new(chunks: Vec<DocumentChunk>) -> Result<Self> {
        if chunks.is_empty() {
            return Err(TinyError::InvalidArgument(
                "cannot create retriever from empty chunk list".to_string(),
            ));
        }

        let mut indexed_chunks = Vec::with_capacity(chunks.len());

        let mut document_frequencies = HashMap::<String, usize>::new();

        let mut total_terms = 0usize;

        for chunk in chunks {
            let terms = tokenize(chunk.text());

            let term_count = terms.len();

            total_terms += term_count;

            let mut term_frequencies = HashMap::<String, usize>::new();

            for term in terms {
                *term_frequencies.entry(term).or_insert(0) += 1;
            }

            /*
             * document frequency는 한 문서 안에서
             * 단어가 몇 번 나왔는지가 아니라,
             *
             * 몇 개 chunk에 등장했는지를 센다.
             */
            for term in term_frequencies.keys() {
                *document_frequencies.entry(term.clone()).or_insert(0) += 1;
            }

            indexed_chunks.push(IndexedChunk {
                chunk,
                term_frequencies,
                term_count,
            });
        }

        let average_document_length = total_terms as f32 / indexed_chunks.len() as f32;

        Ok(Self {
            chunks: indexed_chunks,
            document_frequencies,
            average_document_length,
            k1: DEFAULT_K1,
            b: DEFAULT_B,
        })
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn average_document_length(&self) -> f32 {
        self.average_document_length
    }

    fn score_chunk(&self, chunk: &IndexedChunk, query_terms: &HashSet<String>) -> f32 {
        let document_count = self.chunks.len() as f32;

        let document_length = chunk.term_count as f32;

        let mut score = 0.0f32;

        for term in query_terms {
            let Some(&term_frequency) = chunk.term_frequencies.get(term) else {
                continue;
            };

            let document_frequency = *self.document_frequencies.get(term).unwrap_or(&0) as f32;

            /*
             * BM25 IDF
             *
             * rare term일수록 점수가 높다.
             */
            let idf = (1.0
                + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
                .ln();

            let tf = term_frequency as f32;

            let length_normalization =
                1.0 - self.b + self.b * (document_length / self.average_document_length);

            let numerator = tf * (self.k1 + 1.0);

            let denominator = tf + self.k1 * length_normalization;

            score += idf * (numerator / denominator);
        }

        score
    }
}

impl Retriever for LexicalRetriever {
    fn retrieve(&self, query: &str, top_k: usize) -> Result<Vec<RetrievedChunk>> {
        if top_k == 0 {
            return Err(TinyError::InvalidArgument(
                "retrieval top_k must be greater than zero".to_string(),
            ));
        }

        let query_terms = tokenize(query).into_iter().collect::<HashSet<_>>();

        if query_terms.is_empty() {
            return Err(TinyError::InvalidArgument(
                "retrieval query produced no searchable terms".to_string(),
            ));
        }

        let mut results = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                let score = self.score_chunk(chunk, &query_terms);

                /*
                 * query term이 하나도 없는 chunk는
                 * 결과에서 제거.
                 */
                if score <= 0.0 {
                    return None;
                }

                Some(RetrievedChunk::new(chunk.chunk.clone(), score))
            })
            .collect::<Vec<_>>();

        results.sort_by(|left, right| right.score().total_cmp(&left.score()));

        results.truncate(top_k);

        Ok(results)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut terms = Vec::new();

    let mut current = String::new();

    for character in text.chars() {
        if character.is_alphanumeric() {
            /*
             * char::to_lowercase()는 한 char에서
             * 여러 Unicode char가 나올 수도 있으므로
             * extend 형태로 처리.
             */
            current.extend(character.to_lowercase());
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        terms.push(current);
    }

    terms
}

#[cfg(test)]
mod tests {
    use super::{LexicalRetriever, tokenize};

    use crate::{
        document::{ChunkConfig, Document, chunk_document},
        retrieval::Retriever,
    };

    fn test_chunks() -> Vec<crate::document::DocumentChunk> {
        let documents = [
            ("rust.txt", "Rust uses ownership to manage memory safely."),
            (
                "metal.txt",
                "Metal is Apple's low-level GPU programming API.",
            ),
            (
                "cargo.txt",
                "Cargo is Rust's package manager and build tool.",
            ),
        ];

        documents
            .into_iter()
            .flat_map(|(source, text)| {
                let document = Document::new(source, text);

                chunk_document(&document, &ChunkConfig::new(512, 0).unwrap()).unwrap()
            })
            .collect()
    }

    #[test]
    fn tokenizes_ascii_text() {
        assert_eq!(
            tokenize("Rust uses Ownership."),
            vec!["rust", "uses", "ownership",],
        );
    }

    #[test]
    fn tokenizes_korean_without_utf8_failure() {
        assert_eq!(
            tokenize("러스트는 소유권을 사용합니다."),
            vec!["러스트는", "소유권을", "사용합니다",],
        );
    }

    #[test]
    fn retrieves_most_relevant_chunk() {
        let retriever = LexicalRetriever::new(test_chunks()).unwrap();

        let results = retriever.retrieve("Rust ownership memory", 2).unwrap();

        assert!(!results.is_empty());

        assert_eq!(results[0].chunk().source().to_str(), Some("rust.txt"),);

        assert!(results[0].chunk().text().contains("ownership"));

        assert!(results[0].score() > 0.0);
    }

    #[test]
    fn retrieves_metal_chunk_for_gpu_query() {
        let retriever = LexicalRetriever::new(test_chunks()).unwrap();

        let results = retriever.retrieve("Metal GPU API", 1).unwrap();

        assert_eq!(results.len(), 1,);

        assert_eq!(results[0].chunk().source().to_str(), Some("metal.txt"),);
    }

    #[test]
    fn respects_top_k() {
        let retriever = LexicalRetriever::new(test_chunks()).unwrap();

        let results = retriever.retrieve("Rust", 1).unwrap();

        assert_eq!(results.len(), 1,);
    }

    #[test]
    fn rejects_zero_top_k() {
        let retriever = LexicalRetriever::new(test_chunks()).unwrap();

        let result = retriever.retrieve("Rust", 0);

        assert!(result.is_err());
    }

    #[test]
    fn query_without_matches_returns_empty_results() {
        let retriever = LexicalRetriever::new(test_chunks()).unwrap();

        let results = retriever.retrieve("elephant banana telescope", 5).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn rejects_empty_index() {
        let result = LexicalRetriever::new(Vec::new());

        assert!(result.is_err());
    }
}
