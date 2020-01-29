use std::collections::HashMap;

pub fn find_optimal<'a,S,A>(prob : &MDP<'a,S,A>) -> Policy
  where S : State,
        A : Action {

  let mut pol = Policy { choice : vec![0; prob.states.len()] };
  let mut new_pol = Policy { choice : vec![0; prob.states.len()] };

  loop {
    let pol =
      std::mem::replace(&mut new_pol,
                        policy_improvement(prob,
                                           &policy_evaluation(prob,
                                                              &pol,
                                                              0.0000001)));
    if new_pol == pol {
      break;
    }
  }
  new_pol
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
      value[index] = update;
    }

    if max_diff.abs() <= thresh {
      break;
    }
    max_diff = 0.0;
  }
  StateValue {value}
}

fn policy_improvement<'a,S,A>(prob : &MDP<'a,S,A>,
                              val : &StateValue) -> Policy
  where S : State,
        A : Action {

  let mut new_pol = vec![0; prob.states.len()];

  let mut max_val : f64 = 0.0;
  let mut max_index : usize = 0;
  for index in 0..prob.states.len() {
    for index_action in 0..prob.actions.len() {
      let (reward,index_next) =
        prob.dynamics.get(&(index, index_action)).unwrap();
      let ret = (*reward as f64) + prob.discount*val.value[*index_next];
      if index_action == 0 || (ret >= max_val + std::f64::EPSILON) {
        max_val = ret;
        max_index = index_action;
      }
    }
    max_val = 0.0;
    new_pol[index] = max_index;
  }
  Policy {choice : new_pol}
}

pub trait State {}
pub trait Action {}

// Deterministic policy that gives the index of the chosen action
// for the indexed state.
#[derive(PartialEq)]
pub struct Policy {
  pub choice : Vec<usize>,
}

pub struct StateValue {
  pub value : Vec<f64>,
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
  static EX_DISC : f64 = 0.9;

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

  fn eq_slice_f64(slice1 : &[f64], slice2 : &[f64]) -> bool {

    slice1.iter()
          .zip(slice2.iter())
          .find(|(f1,f2)| (*f1 - *f2).abs() >= std::f64::EPSILON)
          == None
  }

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

  #[test]
  fn grid_improv() {

    let mdp = MDP { states : EX_STATES,
                    actions : EX_ACTIONS,
                    discount : EX_DISC,
                    dynamics : EX_DYNAMICS.iter().cloned().collect()};
    let optimal_pol = Policy { choice : vec![0,3,0,0] };
    let optimal_val = StateValue { value : vec![-1.0,0.0,0.0,0.0]};

    println!("{:?}", policy_improvement(&mdp,&optimal_val).choice);
    assert!( optimal_pol.choice == policy_improvement(&mdp,&optimal_val).choice);
  }
}
