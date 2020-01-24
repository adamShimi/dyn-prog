use std::collections::HashMap;
use std::collections::HashSet;

pub fn find_optimal(_prob : MDP) -> Policy {
  unimplemented!("Still everything to do!");
}

pub struct State;
pub struct Action;

// Deterministic policy
type Policy = HashMap<State,Action>;

pub struct MDP {
  states : HashSet<State>,
  actions : HashSet<Action>,
  // Link a 4-tuple (state,action,reward,next state) with
  // the probability to get reward and next state when
  // using action in state.
  dynamics : HashSet<(State,Action,f64,State),f64>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
