#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
)]
pub enum DType {
    F32,
    F16,
}

impl DType {
    pub fn size_in_bytes(
        self,
    ) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
        }
    }
}
