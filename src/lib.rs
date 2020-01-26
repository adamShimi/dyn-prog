use std::collections::HashMap;

pub fn find_optimal<'a,S,A>(_prob : &MDP<'a,S,A>) -> Policy
  where S : State,
        A : Action {
  unimplemented!("Still everything to do!");
}

fn policy_evaluation<'a,S,A>(prob : &MDP<'a,S,A>,
                               pol : &Policy,
                               thresh : f64) -> StateValue
  where S : State,
        A : Action {

  let mut max_diff : f64 = 0.0;
  let mut value = vec![0.0; prob.states.len()];

  loop  {
    for index in 0..prob.states.len() {
      let (reward,index_next) = prob.dynamics.get(&(index,
                                                    *pol.choice.get(index).unwrap()
                                                   )
                                             )
                                             .unwrap();
      let update : f64 = (*reward as f64)+prob.discount*value[*index_next];
      max_diff = max_diff.max((update-value[index]).abs());
      println!("{}",max_diff);
      value[index] = update;
    }

    if (max_diff - thresh).abs() <= std::f64::EPSILON {
      break;
    }
    max_diff = 0.0;
  }
  StateValue {value}
}

fn policy_improvement<'a,S,A,M>(_prob : &MDP<'a,S,A>,
                                _pol : &Policy,
                                _val : &StateValue) -> Policy
  where S : State,
        A : Action {

  unimplemented!("Second block of General Policy Iteration");
}

pub trait State {}
pub trait Action {}

// Deterministic policy that gives the index of the chosen action
// for the indexed state.
pub struct Policy {
  choice : Vec<usize>,
}

pub struct StateValue {
  value : Vec<f64>,
}

// Deterministic MDP
pub struct MDP<'a, S : State, A : Action> {
  pub states : &'a [S],
  pub actions : &'a [A],
  pub discount : f64,
  // Link a tuple (index state,index action) deterministically
  // to the tuple (reward, index next state).
  pub dynamics : HashMap<(usize,usize),(isize,usize)>,
}

fn eq_slice_f64(slice1 : &[f64], slice2 : &[f64]) -> bool {

  slice1.iter()
        .zip(slice2.iter())
        .find(|(f1,f2)| (*f1 - *f2).abs() >= std::f64::EPSILON)
        == None
}

#[cfg(test)]
mod tests {

  use super::*;

  struct GridState {
    abs : usize,
    ord : usize,
  }

  impl State for GridState {}

  enum GridAction {
    Up,
    Down,
    Left,
    Right,
  }

  impl Action for GridAction {}

  static EX_STATES : &[GridState] = &[GridState {abs : 0, ord: 0},
                                      GridState {abs : 0, ord: 1},
                                      GridState {abs : 1, ord: 0},
                                      GridState {abs : 1, ord: 1}];
  static EX_ACTIONS : &[GridAction] = &[GridAction::Up,
                        GridAction::Down,
                        GridAction::Left,
                        GridAction::Right];
  static EX_DISC : f64 = 1.0;

  static EX_DYNAMICS : [((usize,usize),(isize,usize));16] =
    [((0,0),(-1,2)),
     ((0,1),(-1,0)),
     ((0,2),(-1,0)),
     ((0,3),(-1,1)),

     ((1,0),(-1,1)),
     ((1,1),(-1,0)),
     ((1,2),(-1,1)),
     ((1,3),(0,3)),

     ((2,0),(0,3)),
     ((2,1),(-1,2)),
     ((2,2),(-1,0)),
     ((2,3),(-1,2)),

     ((3,0),(0,3)),
     ((3,1),(0,3)),
     ((3,2),(0,3)),
     ((3,3),(0,3))];

  #[test]
  fn grid_eval() {
    let mdp = MDP { states : EX_STATES,
                    actions : EX_ACTIONS,
                    discount : EX_DISC,
                    dynamics : EX_DYNAMICS.iter().cloned().collect()};
    let optimal_pol = Policy { choice : vec![3,3,0,0] };
    let optimal_val = StateValue { value : vec![-1.0,0.0,0.0,0.0]};

    assert!(eq_slice_f64(&optimal_val.value,
                         &policy_evaluation(&mdp,
                                            &optimal_pol,
                                            std::f64::EPSILON).value
                        )
           );

  }
}
