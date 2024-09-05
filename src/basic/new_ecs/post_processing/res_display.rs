use tabled::Tabled;
use std::fmt;
/// A wrapper around a float that limits the number of decimal places when printed.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub(crate) struct FloatWrapper {
    pub(crate) value: f64,
    pub(crate) precision: usize, // Number of decimal places to display
}

impl FloatWrapper {
    /// Creates a new `FloatWrapper` with the given value and precision.
    pub fn new(value: f64, precision: usize) -> Self {
        FloatWrapper { value, precision }
    }
}
impl Default for FloatWrapper {
    fn default() -> Self {
        Self { value: Default::default(), precision: 3 }
    }
}

impl fmt::Display for FloatWrapper {
    /// Formats the float with the specified precision.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the value with the specified number of decimal places.
        write!(f, "{:.1$}", self.value, self.precision)
    }
}

impl fmt::Debug for FloatWrapper {
    /// Formats the float for debugging with the specified precision.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display with the same precision for debug output.
        write!(f, "{:.1$}", self.value, self.precision)
    }
}

/// Table row for display Bus results.
#[derive(Debug, Tabled)]
#[allow(non_snake_case)]
pub(crate) struct BusResTable {
    pub(crate) Bus: i32,
    pub(crate) Vm: FloatWrapper,
    pub(crate) Va: FloatWrapper,
    pub(crate) P_mw: FloatWrapper,
    pub(crate) Q_mvar: FloatWrapper,
}

/// Data structure for storing results of power flow calculations for a line, with limited decimal precision for output.
#[derive(Debug, Default, Tabled)]
#[allow(non_snake_case)] 
pub struct LineResTable {
    pub(crate) from: i64,
    pub(crate) to: i64,
    pub(crate) p_from_mw: FloatWrapper, // Active power from the 'from' bus (MW)
    pub(crate) q_from_mvar: FloatWrapper, // Reactive power from the 'from' bus (MVAr)
    pub(crate) p_to_mw: FloatWrapper, // Active power to the 'to' bus (MW)
    pub(crate) q_to_mvar: FloatWrapper, // Reactive power to the 'to' bus (MVAr)
    pub(crate) pl_mw: FloatWrapper, // Line active power loss (MW)
    pub(crate) ql_mvar: FloatWrapper, // Line reactive power loss (MVAr)
    pub(crate) i_from_ka: FloatWrapper, // Current from the 'from' bus (kA)
    pub(crate) i_to_ka: FloatWrapper, // Current to the 'to' bus (kA)
    pub(crate) i_ka: FloatWrapper, // Line current (kA)
    pub(crate) vm_from_pu: FloatWrapper, // Voltage magnitude at the 'from' bus (p.u.)
    pub(crate) va_from_degree: FloatWrapper, // Voltage angle at the 'from' bus (degrees)
    pub(crate) vm_to_pu: FloatWrapper, // Voltage magnitude at the 'to' bus (p.u.)
    pub(crate) va_to_degree: FloatWrapper, // Voltage angle at the 'to' bus (degrees)
    pub(crate) loading_percent: FloatWrapper, // Line loading percentage (%)
}
