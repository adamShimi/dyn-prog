use std::collections::HashMap;

pub fn find_optimal<'a,S,A>(prob : &MDP<'a,S,A>,
                            gpi : GPIVersion) -> Policy
  where S : State,
        A : Action {

  let pol = Policy { choice : vec![(0..prob.actions.len()).collect::<Vec<usize>>();
                                   prob.states.len()] };
  let mut new_pol = Policy { choice : vec![(0..prob.actions.len()).collect::<Vec<usize>>();
                                           prob.states.len()] };

  match gpi {
    GPIVersion::PolicyIteration { thresh } => {
      loop {
        let pol =
          std::mem::replace(&mut new_pol,
                            policy_improvement(prob,
                                               &policy_evaluation(prob,
                                                                  &pol,
                                                                  thresh)));
        if new_pol == pol {
          break;
        }
      }
    },
    GPIVersion::ValueIteration => {
      loop {
        let pol =
          std::mem::replace(&mut new_pol,
                            value_iteration(prob,
                                            &pol));
        if new_pol == pol {
          break;
        }
      }
    },
  }
  new_pol
}

pub enum GPIVersion {
  PolicyIteration { thresh : f64},
  ValueIteration,
}

fn policy_evaluation<'a,S,A>(prob : &MDP<'a,S,A>,
                             pol : &Policy,
                             thresh : f64) -> StateValue
  where S : State,
        A : Action {

  let mut value = vec![0.0; prob.states.len()];

  loop  {
    let (val, max_diff) = sweep(prob,pol,&StateValue {value});

    value = val.value;

    if max_diff.abs() <= thresh {
      break;
    }
  }
  StateValue {value}
}

fn sweep<'a,S,A>(prob : &MDP<'a,S,A>,
                 pol : &Policy,
                 val : &StateValue) -> (StateValue,f64)
  where S : State,
        A : Action {


  let mut max_diff : f64 = 0.0;
  let mut value = vec![0.0; prob.states.len()];
  let mut update : f64 = 0.0;
  for index in 0..prob.states.len() {
    let factor = 1.0/(pol.choice.get(index).unwrap().len() as f64);
    for index_action in pol.choice.get(index).unwrap() {
      let (reward,index_next) = prob.dynamics.get(&(index,
                                                    *index_action
                                                   )
                                             )
                                             .unwrap();
      update += factor*((*reward as f64)+prob.discount*val.value[*index_next]);
    }
    max_diff = max_diff.max((update-val.value[index]).abs());
    value[index] = update;
    update = 0.0;
  }
  (StateValue{value},max_diff)
}

fn policy_improvement<'a,S,A>(prob : &MDP<'a,S,A>,
                              val : &StateValue) -> Policy
  where S : State,
        A : Action {

  let mut new_pol = vec![Vec::new(); prob.states.len()];

  let mut max_val : f64 = 0.0;
  let mut max_indexes : Vec<usize> = Vec::new();
  for index in 0..prob.states.len() {
    for index_action in 0..prob.actions.len() {
      let (reward,index_next) =
        prob.dynamics.get(&(index, index_action)).unwrap();
      let ret = (*reward as f64) + prob.discount*val.value[*index_next];
      if index_action == 0 || (ret > max_val + std::f64::EPSILON) {
        max_val = ret;
        max_indexes.clear();
        max_indexes.push(index_action);
      } else if (ret-max_val).abs() <= std::f64::EPSILON {
        max_indexes.push(index_action);
      }
    }
    max_val = 0.0;
    new_pol[index] = std::mem::replace(&mut max_indexes,Vec::new());
  }
  Policy {choice : new_pol}
}

fn value_iteration<'a,S,A>(prob : &MDP<'a,S,A>,
                           pol : &Policy) -> Policy
  where S : State,
        A : Action {

  let (val,_) = sweep(prob,pol,&StateValue {value : vec![0.0;prob.states.len()]});

  policy_improvement(prob,&val)
}



pub trait State {}
pub trait Action {}

// Stochastic policy that gives the list of index of optimal actions.
#[derive(PartialEq,Debug)]
pub struct Policy {
  pub choice : Vec<Vec<usize>>,
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

    let optimal_pol = Policy { choice : vec![vec![3],vec![3],vec![0],vec![0]] };
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
    let optimal_pol = Policy { choice : vec![vec![0,3],vec![3],vec![0],vec![0,1,2,3]] };
    let optimal_val = StateValue { value : vec![-1.0,0.0,0.0,0.0]};

    assert_eq!(optimal_pol.choice,policy_improvement(&mdp,&optimal_val).choice);
  }

  #[test]
  fn grid_optimal() {

    let mdp = MDP { states : EX_STATES,
                    actions : EX_ACTIONS,
                    discount : EX_DISC,
                    dynamics : EX_DYNAMICS.iter().cloned().collect()};
    let optimal_pol = Policy { choice : vec![vec![0,3],vec![3],vec![0],vec![0,1,2,3]] };

    assert_eq!(optimal_pol, find_optimal(&mdp,
                                         GPIVersion::PolicyIteration { thresh : 0.1_000_000}));
  }

  #[test]
  fn grid_optimal_value() {

    let mdp = MDP { states : EX_STATES,
                    actions : EX_ACTIONS,
                    discount : EX_DISC,
                    dynamics : EX_DYNAMICS.iter().cloned().collect()};
    let optimal_pol = Policy { choice : vec![vec![0,3],vec![3],vec![0],vec![0,1,2,3]] };

    assert_eq!(optimal_pol,find_optimal(&mdp,
                                        GPIVersion::ValueIteration));
  }
}
