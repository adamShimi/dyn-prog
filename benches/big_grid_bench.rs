use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dyn_prog::{find_optimal,GPIVersion};
use dyn_prog::mdp::MDP;
use dyn_prog::mdp::grid_world::{GridWorld,GridState};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mdp = GridWorld::new(10,10,GridState{row:0,col:0},GridState{row:8,col:4},0.9);
    let gpi = GPIVersion::PolicyIteration {thresh : std::f64::EPSILON};
    c.bench_function("grid 10x10", |b| b.iter(|| find_optimal(black_box(&mdp),black_box(gpi))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
