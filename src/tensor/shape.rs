use crate::error::{
    Result,
    TinyError,
};

#[derive(
    Debug,
    Clone,
    PartialEq,
    Eq,
)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(
        dims: &[usize],
    ) -> Result<Self> {
        if dims.is_empty() {
            return Err(
                TinyError::InvalidShape(
                    "shape cannot be empty"
                        .to_string(),
                ),
            );
        }

        if dims.contains(&0) {
            return Err(
                TinyError::InvalidShape(
                    "shape dimensions must be greater than zero"
                        .to_string(),
                ),
            );
        }

        Ok(Self {
            dims: dims.to_vec(),
        })
    }

    pub fn dims(
        &self,
    ) -> &[usize] {
        &self.dims
    }

    pub fn rank(
        &self,
    ) -> usize {
        self.dims.len()
    }

    pub fn numel(
        &self,
    ) -> usize {
        self.dims
            .iter()
            .product()
    }

    pub fn dim(
        &self,
        index: usize,
    ) -> Result<usize> {
        self.dims
            .get(index)
            .copied()
            .ok_or_else(|| {
                TinyError::InvalidDimension(
                    format!(
                        "dimension {index} does not exist for shape {:?}",
                        self.dims
                    ),
                )
            })
    }
}
