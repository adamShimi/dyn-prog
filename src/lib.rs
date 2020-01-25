use std::collections::HashMap;

use std::hash::Hash;

pub fn find_optimal<'a,S,A,M>(_prob : &M) -> Policy<'a,A>
  where S : State,
        A : Action,
        M : MDP<'a,S,A> {
  unimplemented!("Still everything to do!");
}

fn policy_evaluation<'a,S,A,M>(prob : &'a M,
                               pol : &'a Policy<'a,A>,
                               thresh : f64) -> StateValue
  where S : State,
        A : Action,
        M : MDP<'a,S,A> {

  let mut max_diff = thresh;
  let mut value = vec![0.0; prob.states().len()];

  while max_diff >= thresh - std::f64::EPSILON {
    for (index,state) in prob.states().iter().enumerate() {
      let update : f64 =
        prob.rewards().iter()
                    .zip(prob.states().iter().enumerate())
                    .map(|(reward,(index_next,next_state))|
                      prob.dynamics(state,
                                    &pol.choice[index],
                                    *reward,
                                    next_state)
                      *((*reward as f64)+prob.discount()*value[index_next])
                    )
                    .sum();
      max_diff = max_diff.max((update-value[index]).abs());
      value[index] = update;
    }
  }
  StateValue {value}
}

fn policy_improvement<'a,S,A,M>(_prob : &M,
                                _pol : &Policy<'a,A>,
                                _val : &StateValue) -> Policy<'a,A>
  where S : State,
        A : Action,
        M : MDP<'a,S,A> {

  unimplemented!("Second block of General Policy Iteration");
}

pub trait State : Eq + Hash + Copy + 'static {}
pub trait Action : Eq + Hash + Copy + 'static {}

// Deterministic policy
pub struct Policy<'a, A : Action> {
  choice : &'a [A],
}

pub struct StateValue {
  value : Vec<f64>,
}

pub trait MDP<'a,S,A> {
  fn states(&self) -> &'a [S];
  fn actions(&self) -> &'a [A];
  fn rewards(&self) -> &'a [isize];
  fn discount(&self) -> f64;
  fn dynamics(&self, state : &S, action : &A, reward : isize, next_state : &S) -> f64;
}

pub struct TabMDP<'a, S : State, A : Action> {
  pub states : &'a [S],
  pub actions : &'a [A],
  pub rewards : &'a [isize],
  pub discount : f64,
  // Link a 4-tuple (state,action,reward,next state) with
  // the probability to get reward and next state when
  // using action in state.
  pub dynamics : HashMap<(S,A,isize,S),usize>,
}

impl<'a,S,A> MDP<'a,S,A> for TabMDP<'a,S,A>
  where S : State,
        A : Action {

  fn states(&self) -> &'a [S] {
    self.states
  }
  fn actions(&self) -> &'a [A] {
    self.actions
  }
  fn rewards(&self) -> &'a [isize] {
    self.rewards
  }
  fn discount(&self) -> f64 {
    self.discount
  }
  fn dynamics(&self, state : &S, action : &A, reward : isize, next_state : &S) -> f64 {
    (*self.dynamics.get(&(*state,*action,reward,*next_state)).unwrap() as f64)/
      (self.actions.len() as f64)
  }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
