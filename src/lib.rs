use std::collections::HashMap;
use std::collections::HashSet;

pub fn find_optimal<S,A>(_prob : MDP<S,A>) -> Policy<S,A>
  where S : State,
        A : Action {
  unimplemented!("Still everything to do!");
}

pub trait State {}
pub trait Action {}

// Deterministic policy
type Policy<S : State, A : Action> = HashMap<S,A>;

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
