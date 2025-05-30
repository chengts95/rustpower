use std::collections::VecDeque;

#[derive(Component, Debug, Clone)]
pub struct ScheduledInjection {
    pub target_p: VecDeque<(f64, f64)>, // (time, value) in seconds, pu
    pub target_q: VecDeque<(f64, f64)>,
}
pub fn scheduled_injection_system(
    time: Res<Time>,
    mut query: Query<(&mut TargetP, &mut TargetQ, &mut ScheduledInjection)>,
) {
    let now = time.0;
    for (mut tp, mut tq, mut sched) in &mut query {
        while let Some(&(t, p)) = sched.target_p.front() {
            if now >= t {
                tp.0 = p;
                sched.target_p.pop_front();
            } else {
                break;
            }
        }
        while let Some(&(t, q)) = sched.target_q.front() {
            if now >= t {
                tq.0 = q;
                sched.target_q.pop_front();
            } else {
                break;
            }
        }
    }
}
