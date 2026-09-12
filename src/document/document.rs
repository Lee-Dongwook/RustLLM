use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    source: PathBuf,
    text: String,
}

impl Document {
    pub fn new(source: impl Into<PathBuf>, text: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            text: text.into(),
        }
    }

    pub fn source(&self) -> &Path {
        &self.source
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::Document;

    #[test]
    fn creates_document() {
        let document = Document::new("notes.md", "# Rust\nOwnership is important.");

        assert_eq!(document.source().to_str(), Some("notes.md"),);

        assert_eq!(document.text(), "# Rust\nOwnership is important.",);

        assert!(!document.is_empty());
    }
}
