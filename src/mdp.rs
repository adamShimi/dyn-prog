use std::collections::HashMap;

pub trait State {}
pub trait Action {}

// Deterministic MDP
pub struct MDP<'a, S : State, A : Action> {
  pub states : &'a [S],
  pub actions : &'a [A],
  pub discount : f64,
  // Link a tuple (index state,index action) to the list of
  // triples (proba, reward, index next state),
  // where the proba is the numerator of the fraction
  // giving the probability of the state, where the denominator
  // is the number of states.
  pub dynamics : HashMap<(usize,usize),&'a [(usize,isize,usize)]>,
}
