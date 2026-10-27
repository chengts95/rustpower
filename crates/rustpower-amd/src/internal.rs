// Flip is a "negation about -1", and is used to mark an integer i that is
// normally non-negative. flip(EMPTY) is empty. Flip of a number > EMPTY
// is negative, and flip of a number < EMTPY is positive. flip(flip(i)) = i
// for all integers i. unflip(i) is >= EMPTY.
pub const EMPTY: isize = -1;

pub fn flip(i: isize) -> isize {
    -i - 2
}

// Logical expression of p implies q:
pub fn implies(p: bool, q: bool) -> bool {
    !p || q
}

// Feature: debug1

#[cfg(feature = "debug1")]
macro_rules! debug1_print {
    ($( $args:expr ),*) => { print!( $( $args ),* ); }
}

#[cfg(not(feature = "debug1"))]
macro_rules! debug1_print {
    ($( $args:expr ),*) => {};
}

#[cfg(feature = "debug1")]
macro_rules! debug1_println {
    ($( $args:expr ),*) => { println!( $( $args ),* ); }
}

#[cfg(not(feature = "debug1"))]
macro_rules! debug1_println {
    ($( $args:expr ),*) => {};
}

// Feature: debug2

#[cfg(feature = "debug2")]
macro_rules! debug2_print {
    ($( $args:expr ),*) => { print!( $( $args ),* ); }
}

#[cfg(not(feature = "debug2"))]
macro_rules! debug2_print {
    ($( $args:expr ),*) => {};
}

#[cfg(feature = "debug2")]
macro_rules! debug2_println {
    ($( $args:expr ),*) => { println!( $( $args ),* ); }
}

#[cfg(not(feature = "debug2"))]
macro_rules! debug2_println {
    ($( $args:expr ),*) => {};
}

// Feature: debug3

#[cfg(feature = "debug3")]
macro_rules! debug3_print {
    ($( $args:expr ),*) => { print!( $( $args ),* ); }
}

#[cfg(not(feature = "debug3"))]
macro_rules! debug3_print {
    ($( $args:expr ),*) => {};
}

#[cfg(feature = "debug3")]
macro_rules! debug3_println {
    ($( $args:expr ),*) => { println!( $( $args ),* ); }
}

#[cfg(not(feature = "debug3"))]
macro_rules! debug3_println {
    ($( $args:expr ),*) => {};
}

// Feature: debug4

#[cfg(feature = "debug4")]
macro_rules! debug4_print {
    ($( $args:expr ),*) => { print!( $( $args ),* ); }
}

#[cfg(not(feature = "debug4"))]
macro_rules! debug4_print {
    ($( $args:expr ),*) => {};
}

#[cfg(feature = "debug4")]
macro_rules! debug4_println {
    ($( $args:expr ),*) => { println!( $( $args ),* ); }
}

#[cfg(not(feature = "debug4"))]
macro_rules! debug4_println {
    ($( $args:expr ),*) => {};
}

pub(crate) use {
    debug1_print, debug1_println, debug2_print, debug2_println, debug3_print, debug3_println,
    debug4_print, debug4_println,
};
