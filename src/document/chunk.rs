use std::path::{Path, PathBuf};

use crate::error::{Result, TinyError};

use super::document::Document;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl ChunkConfig {
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self> {
        let config = Self {
            chunk_size,
            overlap,
        };

        config.validate()?;

        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.chunk_size == 0 {
            return Err(TinyError::InvalidArgument(
                "document chunk_size must be greater than zero".to_string(),
            ));
        }

        if self.overlap >= self.chunk_size {
            return Err(TinyError::InvalidArgument(format!(
                "document overlap must be smaller than chunk_size: overlap={}, chunk_size={}",
                self.overlap, self.chunk_size,
            )));
        }

        Ok(())
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentChunk {
    index: usize,
    source: PathBuf,
    start_char: usize,
    end_char: usize,
    text: String,
}

impl DocumentChunk {
    pub(crate) fn new(
        index: usize,
        source: impl Into<PathBuf>,
        start_char: usize,
        end_char: usize,
        text: impl Into<String>,
    ) -> Self {
        Self {
            index,
            source: source.into(),
            start_char,
            end_char,
            text: text.into(),
        }
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn start_char(&self) -> usize {
        self.start_char
    }

    pub fn end_char(&self) -> usize {
        self.end_char
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn len(&self) -> usize {
        self.end_char - self.start_char
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

pub fn chunk_document(document: &Document, config: &ChunkConfig) -> Result<Vec<DocumentChunk>> {
    config.validate()?;

    if document.is_empty() {
        return Err(TinyError::InvalidArgument(
            "cannot chunk an empty document".to_string(),
        ));
    }

    let text = document.text();

    /*
     * Rust String은 UTF-8이므로 byte index로 아무 위치나
     * 자를 수 없다.
     *
     * 예:
     *
     * "가나다"
     *
     * char index:
     * 0 1 2 3
     *
     * byte index:
     * 0 3 6 9
     *
     * 따라서 먼저 모든 char boundary의 byte offset을 만든다.
     */
    let mut char_boundaries = text
        .char_indices()
        .map(|(byte_index, _)| byte_index)
        .collect::<Vec<_>>();

    /*
     * 마지막 문자의 끝 위치도 boundary로 넣는다.
     */
    char_boundaries.push(text.len());

    let char_count = char_boundaries.len() - 1;

    let step = config.chunk_size - config.overlap;

    let mut chunks = Vec::new();

    let mut start_char = 0;
    let mut index = 0;

    while start_char < char_count {
        let end_char = (start_char + config.chunk_size).min(char_count);

        let start_byte = char_boundaries[start_char];

        let end_byte = char_boundaries[end_char];

        let chunk_text = &text[start_byte..end_byte];

        chunks.push(DocumentChunk::new(
            index,
            document.source(),
            start_char,
            end_char,
            chunk_text,
        ));

        /*
         * 마지막 chunk까지 도달했으면 끝.
         *
         * 이 체크가 없으면 overlap 때문에 마지막 부분에서
         * 불필요한 짧은 chunk가 하나 더 생길 수 있다.
         */
        if end_char == char_count {
            break;
        }

        start_char += step;
        index += 1;
    }

    Ok(chunks)
}
