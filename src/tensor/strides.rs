use super::Shape;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides {
    values: Vec<usize>,
}

impl Strides {
    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();

        let mut values = vec![0; dims.len()];

        let mut stride = 1;

        for index in (0..dims.len()).rev() {
            values[index] = stride;
            stride *= dims[index];
        }

        Self { values }
    }

    pub(crate) fn from_values(values: Vec<usize>) -> Self {
        Self { values }
    }

    pub fn values(&self) -> &[usize] {
        &self.values
    }

    pub fn get(&self, index: usize) -> Option<usize> {
        self.values.get(index).copied()
    }
}
