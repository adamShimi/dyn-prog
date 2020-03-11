use crate::mdp::{State, Action, ActionValue, SampleMDP, Policy};
use std::collections::{HashMap,HashSet};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::SliceRandom;

// Public API

pub fn run_monte_carlo_first_visit<S,A,M>(prob : &M) -> Policy
  where S : State,
        A : Action,
        M : SampleMDP<S,A> {

  let mut pol = Policy { choice : vec![(0..prob.nb_actions()).collect::<Vec<usize>>();
                                       prob.nb_states()] };
  let mut new_pol;
  let mut action_value =
    ActionValue { value : HashMap::new() };
  let mut episode;

  let starts = Uniform::from(0..prob.nb_states());
  let mut rng = rand::thread_rng();

  loop {
    episode = get_episode(prob, &pol, starts.sample(&mut rng));
    new_pol = update_first_visit(prob, &mut action_value, episode);

    // Maybe not adapted to Monte Carlo
    if new_pol == pol {
      break;
    } else {
      std::mem::replace(&mut pol, new_pol);
    }
  }
  new_pol
}

struct Episode {
  events : Vec<(usize,usize,f64)>
}

fn get_episode<S,A,M>(prob : &M, pol : &Policy, start_index : usize) -> Episode
  where S : State,
        A : Action,
        M : SampleMDP<S,A> {

  let rng = &mut rand::thread_rng();

  let mut episode = Episode { events : Vec::new() };
  let mut index = start_index;
  let mut index_action = pol.choice[index].choose(rng).unwrap();

  while let Some ((reward,index_next)) = prob.sample(index,*index_action) {
    episode.events.push((index,*index_action,reward));
    index = index_next;
    index_action = pol.choice[index].choose(rng).unwrap();
  }
  episode
}

fn update_first_visit<S,A,M>(prob : &M,
                             action_value: &mut ActionValue,
                             episode: Episode) -> Policy
  where S : State,
        A : Action,
        M : SampleMDP<S,A> {


  let mut new_pol = Policy { choice : vec![(0..prob.nb_actions()).collect::<Vec<usize>>();
                                           prob.nb_states()] };
  let mut partial_return : f64 = 0.0;
  let mut seen : HashSet<(usize,usize)> = HashSet::new();
  let discount = prob.discount();

  for (index,index_action,reward) in episode.events.iter().rev() {
    partial_return = reward + discount*partial_return;
    if !seen.contains(&(*index,*index_action)) {
      seen.insert((*index,*index_action));
      let (count,q) = action_value.value
                                  .entry((*index,*index_action))
                                  .or_insert((0,0.0));
      *count += 1;
      *q += (partial_return-*q)/(*count as f64);
      new_pol.choice[*index] = max_action(prob,*index,action_value);
    }
  }
  new_pol
}

fn max_action<S,A,M>(prob : &M,
                     index : usize,
                     action_value : &ActionValue) -> Vec<usize>
  where S : State,
        A : Action,
        M : SampleMDP<S,A> {

  let mut max_val : f64 = std::f64::MIN;
  let mut max_indexes : Vec<usize> = Vec::new();

  for index_action in 0..prob.nb_actions() {
    let (_,q) = action_value.value.get(&(index,index_action)).unwrap();
    if *q > max_val + std::f64::EPSILON {
      max_val = *q;
      max_indexes.clear();
      max_indexes.push(index_action);
    } else if (*q-max_val).abs() <= std::f64::EPSILON {
      max_indexes.push(index_action);
    }
  }
  max_indexes
}
