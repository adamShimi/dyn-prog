use crate::mdp::{State, Action, MDP, StateValue, Policy};

pub fn run_policy_iteration<S,A,M>(prob : &M,
                               thresh : f64) -> Policy
  where S : State,
        A : Action,
        M : MDP<S,A> {

  let mut pol = Policy { choice : vec![(0..prob.nb_actions()).collect::<Vec<usize>>();
                                       prob.nb_states()] };
  let mut new_pol = Policy { choice : vec![(0..prob.nb_actions()).collect::<Vec<usize>>();
                                           prob.nb_states()] };
  let mut val = StateValue { value : vec![0.0; prob.nb_states()] };

  loop {
    new_pol = std::mem::replace(&mut pol,new_pol);
    std::mem::replace(&mut new_pol,
                      policy_improvement(prob,
                                         &policy_evaluation(prob,
                                                            &pol,
                                                            thresh)));
    if new_pol == pol {
      break;
    }
  }
  new_pol
}

pub fn run_value_iteration<S,A,M>(prob : &M) -> Policy
  where S : State,
        A : Action,
        M : MDP<S,A> {

  let mut pol = Policy { choice : vec![(0..prob.nb_actions()).collect::<Vec<usize>>();
                                       prob.nb_states()] };
  let mut new_pol = Policy { choice : vec![(0..prob.nb_actions()).collect::<Vec<usize>>();
                                           prob.nb_states()] };
  let mut val = StateValue { value : vec![0.0; prob.nb_states()] };

  loop {
    new_pol = std::mem::replace(&mut pol,new_pol);
    let (pol_next, new_val) = value_iteration(prob, &pol, &val);
    std::mem::replace(&mut new_pol, pol_next);
    std::mem::replace(&mut val, new_val);

    if new_pol.choice == pol.choice {
      break;
    }
  }
  new_pol
}

fn policy_evaluation<S,A,M>(prob : &M,
                            pol : &Policy,
                            thresh : f64) -> StateValue
  where S : State,
        A : Action,
        M : MDP<S,A> {

  let mut value = vec![0.0; prob.nb_states()];

  loop  {
    let (val, max_diff) = sweep(prob,pol,&StateValue {value});

    value = val.value;

    if max_diff.abs() <= thresh {
      break;
    }
  }
  StateValue {value}
}

fn sweep<S,A,M>(prob : &M,
                pol : &Policy,
                val : &StateValue) -> (StateValue,f64)
  where S : State,
        A : Action,
        M : MDP<S,A> {


  let mut max_diff : f64 = 0.0;
  let mut value = vec![0.0; prob.nb_states()];
  let mut update : f64 = 0.0;
  for index in 0..prob.nb_states() {
    let factor = 1.0/(pol.choice.get(index).unwrap().len() as f64);
    for index_action in pol.choice.get(index).unwrap() {
      let dyna = prob.dynamics(index, *index_action);
      let nb_choice = dyna.len();
      update += dyna.iter()
                    .map(|(proba,reward,index_next)|
                      (*proba as f64)
                        *((*reward as f64)+prob.discount()*val.value[*index_next])
                    )
                    .sum::<f64>()*(factor/nb_choice as f64);
    }
    max_diff = max_diff.max((update-val.value[index]).abs());
    value[index] = update;
    update = 0.0;
  }
  (StateValue{value},max_diff)
}

fn policy_improvement<S,A,M>(prob : &M,
                             val : &StateValue) -> Policy
  where S : State,
        A : Action,
        M : MDP<S,A> {

  let mut new_pol = vec![Vec::new(); prob.nb_states()];

  let mut max_val : f64 = 0.0;
  let mut max_indexes : Vec<usize> = Vec::new();
  for index in 0..prob.nb_states() {
    for index_action in 0..prob.nb_actions() {
      let dyna = prob.dynamics(index, index_action);
      let nb_choice = dyna.len();
      let ret = dyna.iter()
                    .map(|(proba,reward,index_next)|
                      (*proba as f64)
                        *((*reward as f64)+prob.discount()*val.value[*index_next])
                    )
                    .sum::<f64>()/(nb_choice as f64);
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

fn value_iteration<S,A,M>(prob : &M,
                          pol : &Policy,
                          val : &StateValue) -> (Policy,StateValue)
  where S : State,
        A : Action,
        M : MDP<S,A> {

  let (new_val,_) = sweep(prob,pol,val);

  (policy_improvement(prob,&new_val),new_val)
}


#[cfg(test)]
mod tests {

  use super::*;
  use crate::mdp::grid_world::{GridWorld,GridState};
  use crate::mdp::{Policy,StateValue};

  static EX_DISC : f64 = 0.9;

  fn eq_slice_f64(slice1 : &[f64], slice2 : &[f64]) -> bool {

    slice1.iter()
          .zip(slice2.iter())
          .find(|(f1,f2)| (*f1 - *f2).abs() >= std::f64::EPSILON)
          == None
  }

  #[test]
  fn grid_eval() {

    let mdp = GridWorld::new(2,2,GridState{row:0,col:0},GridState{row:1,col:1},EX_DISC);
    let optimal_pol = Policy { choice : vec![vec![3],vec![0],vec![3],vec![0]] };
    let val = StateValue { value : vec![-1.9,-1.0,-1.0,0.0]};

    assert!(eq_slice_f64(&val.value,
                         &policy_evaluation(&mdp,
                                            &optimal_pol,
                                            std::f64::EPSILON).value
                        ));

  }

  #[test]
  fn grid_improv() {

    let mdp = GridWorld::new(2,2,GridState{row:0,col:0},GridState{row:1,col:1},EX_DISC);
    let improved_pol = Policy { choice : vec![vec![0,3],vec![0,1,3],vec![0,2,3],vec![0,1,2,3]] };
    let val = StateValue { value : vec![-1.0,0.0,0.0,0.0]};

    assert_eq!(improved_pol.choice,policy_improvement(&mdp,&val).choice);
  }

  #[test]
  fn grid_optimal() {

    let mdp = GridWorld::new(2,2,GridState{row:0,col:0},GridState{row:1,col:1},EX_DISC);
    let optimal_pol = Policy { choice : vec![vec![0,3],vec![0],vec![3],vec![0,1,2,3]] };

    assert_eq!(optimal_pol, run_policy_iteration(&mdp, 0.1_000_000));
  }

  #[test]
  fn grid_optimal_value() {

    let mdp = GridWorld::new(2,2,GridState{row:0,col:0},GridState{row:1,col:1},EX_DISC);
    let optimal_pol = Policy { choice : vec![vec![0,3],vec![0],vec![3],vec![0,1,2,3]] };

    assert_eq!(optimal_pol, run_value_iteration(&mdp));
  }
}
