use crate::amd::Control;
use std::mem::size_of;

pub fn control(control: &Control) {
    let alpha = control.dense;

    print!(
        "\nAMD: approximate minimum degree ordering
    dense row parameter: {}\n",
        alpha
    );

    if alpha < 0.0 {
        println!("    no rows treated as dense")
    } else {
        print!(
            "    (rows with more than max ({} * sqrt(n), 16) entries are
    considered \"dense\", and placed last in output permutation)\n",
            alpha
        );
    }

    if control.aggressive {
        println!("    aggressive absorption:  yes");
    } else {
        println!("    aggressive absorption:  no");
    }

    print!("    size of AMD integer: {}\n\n", size_of::<isize>());
}
