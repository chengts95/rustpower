use nalgebra::Complex;
use serde::{Deserialize, Serialize};

/// Represents an admittance value in a power system.
///
/// `Admittance` is a wrapper around a complex number representing the admittance value.
#[derive(Clone, Default, PartialEq, Debug)]
#[cfg_attr(feature = "archive", derive(serde::Serialize, serde::Deserialize))]
pub struct Admittance(pub Complex<f64>);

/// Represents a port with two integer values.
///
/// `Port2` is a structure holding two integer values typically used to denote a port in a system.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Port2(pub nalgebra::Vector2<i32>);
