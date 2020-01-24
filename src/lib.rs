use std::collections::HashMap;
use std::collections::HashSet;

pub fn find_optimal<S,A>(_prob : &MDP<S,A>) -> Policy<S,A>
  where S : State,
        A : Action {
  unimplemented!("Still everything to do!");
}

fn policy_evaluation<S,A>(_prob : &MDP<S,A>,
                          _pol : &Policy<S,A>,
                          _thresh : f64) -> StateValue<S>
  where S : State,
        A : Action {

  unimplemented!("First block of General Policy Iteration");
}

fn policy_improvement<S,A>(_prob : &MDP<S,A>,
                           _pol : &Policy<S,A>,
                           _val : &StateValue<S>) -> Policy<S,A>
  where S : State,
        A : Action {

  unimplemented!("Second block of General Policy Iteration");
}

pub trait State {}
pub trait Action {}

// Deterministic policy
pub struct Policy<S : State, A : Action> {
  choice : HashMap<S,A>,
}

pub struct StateValue<S : State> {
  value : HashMap<S,f64>,
}

pub struct MDP<S : State, A : Action> {
  states : HashSet<S>,
  actions : HashSet<A>,
  // Link a 4-tuple (state,action,reward,next state) with
  // the probability to get reward and next state when
  // using action in state.
  dynamics : HashSet<(S,A,f64,S),f64>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
