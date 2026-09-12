use std::{fs, path::Path};

use crate::error::{Result, TinyError};

use super::Document;

pub fn load_document(path: impl AsRef<Path>) -> Result<Document> {
    let path = path.as_ref();

    validate_extension(path)?;

    let text = fs::read_to_string(path)?;

    if text.trim().is_empty() {
        return Err(TinyError::InvalidArgument(format!(
            "document is empty: {}",
            path.display(),
        )));
    }

    Ok(Document::new(path, text))
}

fn validate_extension(path: &Path) -> Result<()> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase());

    match extension.as_deref() {
        Some("txt" | "md") => Ok(()),

        Some(extension) => Err(TinyError::InvalidArgument(format!(
            "unsupported document format `.{extension}`: \
                         supported formats are .txt and .md"
        ))),

        None => Err(TinyError::InvalidArgument(format!(
            "document has no file extension: {}",
            path.display(),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::load_document;

    fn temp_path(extension: &str) -> std::path::PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!(
            "rustllm-document-{unique}-{}.{}",
            std::process::id(),
            extension,
        ))
    }

    #[test]
    fn loads_text_document() {
        let path = temp_path("txt");

        fs::write(&path, "Rust uses ownership.").unwrap();

        let document = load_document(&path).unwrap();

        assert_eq!(document.text(), "Rust uses ownership.",);

        assert_eq!(document.source(), path.as_path(),);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_markdown_document() {
        let path = temp_path("md");

        fs::write(&path, "# Rust\n\nOwnership").unwrap();

        let document = load_document(&path).unwrap();

        assert!(document.text().contains("Ownership"));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_unsupported_format() {
        let path = temp_path("pdf");

        fs::write(&path, "not really a pdf").unwrap();

        let result = load_document(&path);

        assert!(result.is_err());

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_empty_document() {
        let path = temp_path("txt");

        fs::write(&path, "   \n").unwrap();

        let result = load_document(&path);

        assert!(result.is_err());

        fs::remove_file(path).unwrap();
    }
}
