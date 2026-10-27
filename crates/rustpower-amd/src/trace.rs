//! Opt-in observations only. No tracing exists in non-trace builds.
use std::cell::RefCell;
#[derive(Debug, PartialEq, Eq)]
pub struct Pivot {
    pub representative: usize,
    pub degree: usize,
    pub weight: usize,
    pub eliminated: usize,
    /// Current principal variable states for the first8 pivots only: (id,degree,weight).
    pub candidates: Vec<(usize, usize, usize)>,
    pub updates: Vec<Update>,
    pub merges: Vec<(usize, usize)>,
    pub ordered_front: Vec<isize>,
    pub element_absorptions: Vec<(usize, usize)>,
    pub bucket_insertions: Vec<(usize, usize, isize)>,
}
thread_local! {static PIVOTS:RefCell<Option<Vec<Pivot>>>=const{RefCell::new(None)};}
struct Restore(Option<Vec<Pivot>>);
impl Drop for Restore {
    fn drop(&mut self) {
        PIVOTS.with(|p| *p.borrow_mut() = self.0.take());
    }
}
pub fn capture<T>(f: impl FnOnce() -> T) -> (T, Vec<Pivot>) {
    let guard = Restore(PIVOTS.with(|p| p.borrow_mut().replace(vec![])));
    let value = f();
    let pivots = PIVOTS.with(|p| p.borrow_mut().take().unwrap());
    drop(guard);
    (value, pivots)
}
pub(crate) fn record(
    representative: usize,
    degree: usize,
    weight: usize,
    eliminated: usize,
    degrees: &[usize],
    weights: &[isize],
    elen: &[isize],
) {
    PIVOTS.with(|p| {
        let mut p = p.borrow_mut();
        if let Some(events) = p.as_mut() {
            let candidates = if events.len() < 8 {
                (0..degrees.len())
                    .filter(|&i| weights[i] > 0 && elen[i] >= 0)
                    .map(|i| (i, degrees[i], weights[i] as usize))
                    .collect()
            } else {
                vec![]
            };
            events.push(Pivot {
                representative,
                degree,
                weight,
                eliminated,
                candidates,
                updates: vec![],
                merges: vec![],
                ordered_front: vec![],
                element_absorptions: vec![],
                bucket_insertions: vec![],
            });
        }
    });
}

#[derive(Debug, PartialEq, Eq)]
pub struct Update {
    id: usize,
    degree: usize,
    weight: isize,
    element_count: isize,
    parent: Option<isize>,
    ordered_list: Vec<isize>,
    hash: u64,
}
pub(crate) fn update(
    id: usize,
    degree: usize,
    weight: isize,
    element_count: isize,
    parent: isize,
    list: &[isize],
    hash: u64,
) {
    PIVOTS.with(|p| {
        if let Some(events) = p.borrow_mut().as_mut() {
            if let Some(p) = events.last_mut() {
                p.updates.push(Update {
                    id,
                    degree,
                    weight,
                    element_count,
                    parent: if weight == 0 { Some(parent) } else { None },
                    ordered_list: if weight != 0 { list.to_vec() } else { vec![] },
                    hash,
                });
            }
        }
    });
}
pub(crate) fn merge(into: usize, from: usize) {
    PIVOTS.with(|p| {
        if let Some(events) = p.borrow_mut().as_mut() {
            events.last_mut().unwrap().merges.push((into, from));
        }
    });
}

pub(crate) fn front(values: Vec<isize>) {
    PIVOTS.with(|p| {
        if let Some(e) = p.borrow_mut().as_mut() {
            e.last_mut().unwrap().ordered_front = values;
        }
    });
}
pub(crate) fn absorb_element(from: usize, into: usize) {
    PIVOTS.with(|p| {
        if let Some(e) = p.borrow_mut().as_mut() {
            e.last_mut().unwrap().element_absorptions.push((from, into));
        }
    });
}
pub(crate) fn insert_bucket(variable: usize, degree: usize, old_head: isize) {
    PIVOTS.with(|p| {
        if let Some(e) = p.borrow_mut().as_mut() {
            e.last_mut()
                .unwrap()
                .bucket_insertions
                .push((variable, degree, old_head));
        }
    });
}
