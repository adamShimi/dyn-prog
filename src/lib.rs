pub mod mdp;
pub mod dyn_prog;

use mdp::{State, Action, MDP, Policy};
use dyn_prog::{run_policy_iteration,run_value_iteration};

pub fn find_optimal<S,A,M>(prob : &M,
                           gpi : GPIVersion) -> Policy
  where S : State,
        A : Action,
        M : MDP<S,A> {

  match gpi {
    GPIVersion::PolicyIteration { thresh } =>
      run_policy_iteration(prob,thresh),
    GPIVersion::ValueIteration =>
      run_value_iteration(prob),
  }
}

#[derive(Clone,Copy)]
pub enum GPIVersion {
  PolicyIteration { thresh : f64},
  ValueIteration,
}
