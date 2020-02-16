use criterion::{black_box, criterion_group, criterion_main, Criterion};
use dyn_prog::{find_optimal,GPIVersion};
use dyn_prog::mdp::MDP;
use dyn_prog::mdp::grid_world::{GridWorld,GridState};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mdp = GridWorld::new(10,10,GridState{row:0,col:0},GridState{row:8,col:4},0.9);
    let gpi_pol = GPIVersion::PolicyIteration {thresh : std::f64::EPSILON};
    let gpi_val = GPIVersion::ValueIteration;
    c.bench_function("grid polIter 10x10",
                     |b| b.iter(|| find_optimal(black_box(&mdp),black_box(gpi_pol))));
    c.bench_function("grid valIter 10x10",
                     |b| b.iter(|| find_optimal(black_box(&mdp),black_box(gpi_val))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
