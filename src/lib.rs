use std::collections::BTreeMap;

pub fn find_optimal<'a,S,A>(_prob : &MDP<'a,S,A>) -> Policy<'a,A>
  where S : State,
        A : Action {
  unimplemented!("Still everything to do!");
}

fn policy_evaluation<'a,S,A>(_prob : &MDP<'a,S,A>,
                             _pol : &Policy<'a,A>,
                             _thresh : f64) -> StateValue<'a>
  where S : State,
        A : Action {

  unimplemented!("First block of General Policy Iteration");
}

fn policy_improvement<'a,S,A>(_prob : &MDP<'a,S,A>,
                              _pol : &Policy<'a,A>,
                              _val : &StateValue<'a>) -> Policy<'a,A>
  where S : State,
        A : Action {

  unimplemented!("Second block of General Policy Iteration");
}

pub trait State {}
pub trait Action {}

// Deterministic policy
pub struct Policy<'a, A : Action> {
  choice : &'a [A],
}

pub struct StateValue<'a> {
  value : &'a [f64],
}

pub struct MDP<'a, S : State, A : Action> {
  pub states : &'a [S],
  pub actions : &'a [A],
  pub rewards : &'a [isize],
  pub discount : f64,
  // Link a 4-tuple (state,action,reward,next state) with
  // the probability to get reward and next state when
  // using action in state.
  pub dynamics : BTreeMap<(S,A,isize,S),f64>,
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
